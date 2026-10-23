// Molecular Dynamics GPU kernel for MEC phase interactions
//
// ASSUMPTIONS:
// - Particle positions/velocities stored as float3 arrays
// - MAX_PARTICLES = 10000 (system size limit)
// - MAX_NEIGHBORS = 128 (interaction radius cutoff)
// - Precision: f32 for performance, f64 available for accuracy
// - Block size: 256 threads (optimal for force calculation)
// - Grid size: ceil(num_particles / threads_per_block)
// - Requires: sm_70+ for efficient atomic operations
// REFERENCE: PRISM Spec Section 6.2 "MEC Molecular Dynamics"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

// Configuration constants
constexpr int MAX_PARTICLES = 10000;
constexpr int MAX_NEIGHBORS = 128;
constexpr int THREADS_PER_BLOCK = 256;
constexpr float CUTOFF_RADIUS = 10.0f;
constexpr float CUTOFF_RADIUS_SQ = CUTOFF_RADIUS * CUTOFF_RADIUS;
constexpr float EPSILON = 1e-10f;

// Molecular dynamics parameters
struct MDParams {
    int num_particles;         // Number of particles in system
    float timestep;            // Integration timestep (dt)
    float temperature;         // Target temperature for thermostat
    float box_size;           // Cubic box dimension for PBC
    float epsilon;            // Lennard-Jones well depth
    float sigma;              // Lennard-Jones distance parameter
    float damping;            // Langevin damping coefficient
    float coupling_strength;  // Inter-particle coupling for MEC
    int integration_steps;    // Steps per kernel call
    unsigned long seed;       // Random seed for Langevin
};

// Particle data structure
struct Particle {
    float3 position;
    float3 velocity;
    float3 force;
    float mass;
    int type;        // Particle type for heterogeneous systems
    float charge;    // For electrostatic interactions
    float radius;    // Collision radius
};

// Device function: Apply periodic boundary conditions
__device__ float3 apply_pbc(float3 pos, float box_size) {
    pos.x = pos.x - box_size * floorf(pos.x / box_size);
    pos.y = pos.y - box_size * floorf(pos.y / box_size);
    pos.z = pos.z - box_size * floorf(pos.z / box_size);
    return pos;
}

// Device function: Minimum image distance with PBC
__device__ float3 minimum_image(float3 dr, float box_size) {
    float half_box = box_size * 0.5f;

    if (dr.x > half_box) dr.x -= box_size;
    else if (dr.x < -half_box) dr.x += box_size;

    if (dr.y > half_box) dr.y -= box_size;
    else if (dr.y < -half_box) dr.y += box_size;

    if (dr.z > half_box) dr.z -= box_size;
    else if (dr.z < -half_box) dr.z += box_size;

    return dr;
}

// Device function: Lennard-Jones force calculation
__device__ float3 lennard_jones_force(
    float3 dr,
    float r2,
    float epsilon,
    float sigma
) {
    float sigma2 = sigma * sigma;
    float sigma6 = sigma2 * sigma2 * sigma2;
    float sigma12 = sigma6 * sigma6;

    float r2_inv = 1.0f / (r2 + EPSILON);
    float r6_inv = r2_inv * r2_inv * r2_inv;
    float r12_inv = r6_inv * r6_inv;

    // F = 24 * epsilon * (2 * sigma^12 / r^13 - sigma^6 / r^7)
    float force_mag = 24.0f * epsilon * r2_inv *
                     (2.0f * sigma12 * r12_inv - sigma6 * r6_inv);

    return make_float3(force_mag * dr.x, force_mag * dr.y, force_mag * dr.z);
}

// Device function: Coulomb force for charged particles
__device__ float3 coulomb_force(
    float3 dr,
    float r2,
    float q1,
    float q2
) {
    const float ke = 8.99e9f; // Coulomb constant (simplified)
    float r = sqrtf(r2 + EPSILON);
    float force_mag = ke * q1 * q2 / (r2 * r + EPSILON);

    return make_float3(force_mag * dr.x / r,
                      force_mag * dr.y / r,
                      force_mag * dr.z / r);
}

// Device function: MEC coupling force (metaphysical interaction)
__device__ float3 mec_coupling_force(
    float3 dr,
    float r2,
    float coupling_strength,
    int type1,
    int type2
) {
    // Special coupling between different particle types
    float type_factor = (type1 != type2) ? 2.0f : 1.0f;

    // Oscillating potential for quantum-like behavior
    float r = sqrtf(r2 + EPSILON);
    float phase = r * 2.0f * M_PI / 5.0f; // 5 Angstrom wavelength
    float force_mag = coupling_strength * type_factor *
                     sinf(phase) * expf(-r / 10.0f) / (r + EPSILON);

    return make_float3(force_mag * dr.x / r,
                      force_mag * dr.y / r,
                      force_mag * dr.z / r);
}

// Main force calculation kernel with neighbor list
extern "C" __global__ void calculate_forces_kernel(
    Particle* __restrict__ particles,
    float* __restrict__ potential_energy,
    int* __restrict__ neighbor_list,  // [num_particles][MAX_NEIGHBORS]
    int* __restrict__ neighbor_counts, // Number of neighbors per particle
    int num_particles,
    float timestep,
    float temperature,
    float box_size,
    float epsilon,
    float sigma,
    float damping,
    float coupling_strength
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;

    Particle p_i = particles[tid];
    float3 force_total = make_float3(0.0f, 0.0f, 0.0f);
    float energy = 0.0f;

    // Get neighbor count for this particle
    int num_neighbors = neighbor_counts[tid];
    int neighbor_offset = tid * MAX_NEIGHBORS;

    // Loop over neighbors
    for (int n = 0; n < num_neighbors; ++n) {
        int j = neighbor_list[neighbor_offset + n];
        if (j <= tid) continue; // Avoid double counting

        Particle p_j = particles[j];

        // Compute distance vector with PBC
        float3 dr = make_float3(p_j.position.x - p_i.position.x,
                               p_j.position.y - p_i.position.y,
                               p_j.position.z - p_i.position.z);
        dr = minimum_image(dr, box_size);

        float r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

        // Skip if beyond cutoff
        if (r2 > CUTOFF_RADIUS_SQ) continue;

        // Lennard-Jones interaction
        float3 f_lj = lennard_jones_force(dr, r2, epsilon, sigma);
        force_total.x += f_lj.x;
        force_total.y += f_lj.y;
        force_total.z += f_lj.z;

        // Electrostatic interaction
        if (fabsf(p_i.charge) > EPSILON && fabsf(p_j.charge) > EPSILON) {
            float3 f_coulomb = coulomb_force(dr, r2, p_i.charge, p_j.charge);
            force_total.x += f_coulomb.x;
            force_total.y += f_coulomb.y;
            force_total.z += f_coulomb.z;
        }

        // MEC coupling force
        float3 f_mec = mec_coupling_force(dr, r2, coupling_strength,
                                         p_i.type, p_j.type);
        force_total.x += f_mec.x;
        force_total.y += f_mec.y;
        force_total.z += f_mec.z;

        // Update Newton's 3rd law pair force
        atomicAdd(&particles[j].force.x, -force_total.x);
        atomicAdd(&particles[j].force.y, -force_total.y);
        atomicAdd(&particles[j].force.z, -force_total.z);

        // Compute potential energy (LJ only for now)
        float r6_inv = 1.0f / (r2 * r2 * r2 + EPSILON);
        float sigma6 = sigma * sigma * sigma;
        sigma6 = sigma6 * sigma6;
        energy += 4.0f * epsilon * sigma6 * r6_inv *
                 (sigma6 * r6_inv - 1.0f);
    }

    // Store force
    particles[tid].force = force_total;

    // Store potential energy
    atomicAdd(potential_energy, energy);
}

// Velocity Verlet integration kernel
extern "C" __global__ void integrate_verlet_kernel(
    Particle* __restrict__ particles,
    float* __restrict__ kinetic_energy,
    int num_particles,
    float timestep,
    float temperature,
    float box_size,
    float epsilon,
    float sigma,
    float damping,
    float coupling_strength
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;

    Particle p = particles[tid];
    float dt = timestep;
    float dt2 = dt * dt;

    // Velocity Verlet step 1: Update positions
    p.position.x += p.velocity.x * dt + 0.5f * p.force.x * dt2 / p.mass;
    p.position.y += p.velocity.y * dt + 0.5f * p.force.y * dt2 / p.mass;
    p.position.z += p.velocity.z * dt + 0.5f * p.force.z * dt2 / p.mass;

    // Apply PBC
    p.position = apply_pbc(p.position, box_size);

    // Store old force for velocity update
    float3 old_force = p.force;

    // Force will be recalculated
    p.force = make_float3(0.0f, 0.0f, 0.0f);

    // Update half-step velocity (will complete after force calculation)
    p.velocity.x += 0.5f * old_force.x * dt / p.mass;
    p.velocity.y += 0.5f * old_force.y * dt / p.mass;
    p.velocity.z += 0.5f * old_force.z * dt / p.mass;

    // Store updated particle
    particles[tid] = p;

    // Calculate kinetic energy
    float ke = 0.5f * p.mass * (p.velocity.x * p.velocity.x +
                                p.velocity.y * p.velocity.y +
                                p.velocity.z * p.velocity.z);
    atomicAdd(kinetic_energy, ke);
}

// Complete velocity Verlet step after force calculation
extern "C" __global__ void complete_verlet_kernel(
    Particle* __restrict__ particles,
    int num_particles,
    float timestep
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;

    Particle p = particles[tid];
    float dt = timestep;

    // Velocity Verlet step 2: Complete velocity update
    p.velocity.x += 0.5f * p.force.x * dt / p.mass;
    p.velocity.y += 0.5f * p.force.y * dt / p.mass;
    p.velocity.z += 0.5f * p.force.z * dt / p.mass;

    particles[tid] = p;
}

// Langevin thermostat kernel
extern "C" __global__ void langevin_thermostat_kernel(
    Particle* __restrict__ particles,
    float current_temperature,
    MDParams params
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.num_particles) return;

    // Initialize random state
    curandState rand_state;
    curand_init(params.seed, tid, 0, &rand_state);

    Particle p = particles[tid];

    // Langevin dynamics: F = -γ*v + sqrt(2*γ*kT/m)*η
    float gamma = params.damping;
    float kT = params.temperature * 1.38e-23f; // Boltzmann constant
    float noise_amp = sqrtf(2.0f * gamma * kT / (p.mass * params.timestep));

    // Apply friction
    p.force.x -= gamma * p.velocity.x;
    p.force.y -= gamma * p.velocity.y;
    p.force.z -= gamma * p.velocity.z;

    // Add random force
    p.force.x += noise_amp * curand_normal(&rand_state);
    p.force.y += noise_amp * curand_normal(&rand_state);
    p.force.z += noise_amp * curand_normal(&rand_state);

    particles[tid] = p;
}

// Berendsen thermostat kernel (velocity rescaling)
extern "C" __global__ void berendsen_thermostat_kernel(
    Particle* __restrict__ particles,
    float current_temperature,
    float target_temperature,
    float tau_t,
    float dt,
    int num_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;

    // Calculate scaling factor
    float lambda = sqrtf(1.0f + (dt / tau_t) *
                        (target_temperature / current_temperature - 1.0f));

    // Scale velocities
    Particle p = particles[tid];
    p.velocity.x *= lambda;
    p.velocity.y *= lambda;
    p.velocity.z *= lambda;

    particles[tid] = p;
}

// Build neighbor list using cell lists
extern "C" __global__ void build_neighbor_list_kernel(
    const Particle* __restrict__ particles,
    int* __restrict__ neighbor_list,
    int* __restrict__ neighbor_counts,
    int* __restrict__ cell_list,      // Particle indices in cells
    int* __restrict__ cell_counts,    // Number of particles per cell
    int cells_per_dim,
    MDParams params
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.num_particles) return;

    Particle p_i = particles[tid];
    float cell_size = params.box_size / cells_per_dim;

    // Find cell indices for particle i
    int cx = (int)(p_i.position.x / cell_size) % cells_per_dim;
    int cy = (int)(p_i.position.y / cell_size) % cells_per_dim;
    int cz = (int)(p_i.position.z / cell_size) % cells_per_dim;

    int neighbor_count = 0;
    int neighbor_offset = tid * MAX_NEIGHBORS;

    // Check neighboring cells (3x3x3 = 27 cells)
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                // Get neighbor cell with PBC
                int ncx = (cx + dx + cells_per_dim) % cells_per_dim;
                int ncy = (cy + dy + cells_per_dim) % cells_per_dim;
                int ncz = (cz + dz + cells_per_dim) % cells_per_dim;

                int cell_idx = ncx + ncy * cells_per_dim +
                              ncz * cells_per_dim * cells_per_dim;

                // Check all particles in this cell
                int cell_offset = cell_idx * MAX_PARTICLES;
                int num_in_cell = cell_counts[cell_idx];

                for (int p = 0; p < num_in_cell && neighbor_count < MAX_NEIGHBORS; ++p) {
                    int j = cell_list[cell_offset + p];
                    if (j == tid) continue;

                    Particle p_j = particles[j];

                    // Compute distance
                    float3 dr = make_float3(p_j.position.x - p_i.position.x,
                                          p_j.position.y - p_i.position.y,
                                          p_j.position.z - p_i.position.z);
                    dr = minimum_image(dr, params.box_size);

                    float r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

                    if (r2 < CUTOFF_RADIUS_SQ) {
                        neighbor_list[neighbor_offset + neighbor_count] = j;
                        neighbor_count++;
                    }
                }
            }
        }
    }

    neighbor_counts[tid] = neighbor_count;
}

// Compute system properties kernel
extern "C" __global__ void compute_properties_kernel(
    const Particle* __restrict__ particles,
    float* __restrict__ temperature,
    float* __restrict__ pressure,
    float* __restrict__ diffusion_coefficient,
    float* __restrict__ initial_positions, // For MSD calculation
    int num_particles,
    float box_size,
    float current_time
) {
    // Cooperative calculation using shared memory
    extern __shared__ float shared_props[];

    int tid = threadIdx.x;
    int particle_idx = blockIdx.x * blockDim.x + tid;

    float local_ke = 0.0f;
    float local_virial = 0.0f;
    float local_msd = 0.0f;

    if (particle_idx < num_particles) {
        Particle p = particles[particle_idx];

        // Kinetic energy
        local_ke = 0.5f * p.mass * (p.velocity.x * p.velocity.x +
                                   p.velocity.y * p.velocity.y +
                                   p.velocity.z * p.velocity.z);

        // Mean squared displacement for diffusion
        if (initial_positions != nullptr) {
            int pos_offset = particle_idx * 3;
            float dx = p.position.x - initial_positions[pos_offset];
            float dy = p.position.y - initial_positions[pos_offset + 1];
            float dz = p.position.z - initial_positions[pos_offset + 2];

            // Account for PBC in MSD calculation
            dx = minimum_image(make_float3(dx, 0, 0), box_size).x;
            dy = minimum_image(make_float3(0, dy, 0), box_size).y;
            dz = minimum_image(make_float3(0, 0, dz), box_size).z;

            local_msd = dx * dx + dy * dy + dz * dz;
        }
    }

    // Store in shared memory
    shared_props[tid] = local_ke;
    shared_props[tid + blockDim.x] = local_virial;
    shared_props[tid + 2 * blockDim.x] = local_msd;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_props[tid] += shared_props[tid + stride];
            shared_props[tid + blockDim.x] += shared_props[tid + stride + blockDim.x];
            shared_props[tid + 2 * blockDim.x] += shared_props[tid + stride + 2 * blockDim.x];
        }
        __syncthreads();
    }

    // Write results
    if (tid == 0) {
        atomicAdd(temperature, shared_props[0]);
        atomicAdd(pressure, shared_props[blockDim.x]);
        atomicAdd(diffusion_coefficient, shared_props[2 * blockDim.x]);
    }
}

// Initialize particle positions on a lattice
extern "C" __global__ void initialize_lattice_kernel(
    Particle* __restrict__ particles,
    int particles_per_dim,
    float box_size,
    float initial_temperature,
    unsigned long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_particles = particles_per_dim * particles_per_dim * particles_per_dim;
    if (tid >= num_particles) return;

    // Initialize random state
    curandState rand_state;
    curand_init(seed, tid, 0, &rand_state);

    // Calculate lattice position
    int ix = tid % particles_per_dim;
    int iy = (tid / particles_per_dim) % particles_per_dim;
    int iz = tid / (particles_per_dim * particles_per_dim);

    float spacing = box_size / particles_per_dim;

    Particle p;
    p.position.x = (ix + 0.5f) * spacing;
    p.position.y = (iy + 0.5f) * spacing;
    p.position.z = (iz + 0.5f) * spacing;

    // Random velocities with Maxwell-Boltzmann distribution
    float kT = initial_temperature * 1.38e-23f;
    float v_scale = sqrtf(kT / 1.66e-27f); // Assume unit mass in amu

    p.velocity.x = v_scale * curand_normal(&rand_state);
    p.velocity.y = v_scale * curand_normal(&rand_state);
    p.velocity.z = v_scale * curand_normal(&rand_state);

    p.force = make_float3(0.0f, 0.0f, 0.0f);
    p.mass = 1.0f;
    p.type = tid % 3; // Three particle types for MEC diversity
    p.charge = (tid % 5 == 0) ? 0.1f : 0.0f; // Some charged particles
    p.radius = 1.0f;

    particles[tid] = p;
}

// MEC phase-specific kernel: Quantum coherence calculation
extern "C" __global__ void mec_coherence_kernel(
    const Particle* __restrict__ particles,
    float* __restrict__ coherence_matrix, // [num_particles][num_particles]
    float* __restrict__ entanglement_measure,
    int num_particles,
    float interaction_radius
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_particles * num_particles;
    if (pair_idx >= total_pairs) return;

    int i = pair_idx / num_particles;
    int j = pair_idx % num_particles;

    if (i >= j) { // Only upper triangle
        coherence_matrix[pair_idx] = 0.0f;
        return;
    }

    Particle p_i = particles[i];
    Particle p_j = particles[j];

    // Distance-based quantum coherence
    float3 dr = make_float3(p_j.position.x - p_i.position.x,
                           p_j.position.y - p_i.position.y,
                           p_j.position.z - p_i.position.z);
    float r = sqrtf(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + EPSILON);

    // Velocity correlation for coherence
    float v_dot = p_i.velocity.x * p_j.velocity.x +
                 p_i.velocity.y * p_j.velocity.y +
                 p_i.velocity.z * p_j.velocity.z;
    float v_mag_i = sqrtf(p_i.velocity.x * p_i.velocity.x +
                         p_i.velocity.y * p_i.velocity.y +
                         p_i.velocity.z * p_i.velocity.z + EPSILON);
    float v_mag_j = sqrtf(p_j.velocity.x * p_j.velocity.x +
                         p_j.velocity.y * p_j.velocity.y +
                         p_j.velocity.z * p_j.velocity.z + EPSILON);

    float velocity_coherence = v_dot / (v_mag_i * v_mag_j);

    // Spatial coherence with decay
    float spatial_coherence = expf(-r / interaction_radius);

    // Type-based coherence (particles of same type have higher coherence)
    float type_coherence = (p_i.type == p_j.type) ? 1.0f : 0.5f;

    // Combined coherence measure
    float coherence = spatial_coherence * velocity_coherence * type_coherence;

    coherence_matrix[pair_idx] = coherence;
    coherence_matrix[j * num_particles + i] = coherence; // Symmetric

    // Contribute to entanglement measure
    if (coherence > 0.5f) {
        atomicAdd(entanglement_measure, coherence);
    }
}

// Performance metrics kernel for MD simulation
extern "C" __global__ void md_performance_metrics(
    const float* __restrict__ kinetic_energy,
    const float* __restrict__ potential_energy,
    const float* __restrict__ temperature,
    float* __restrict__ metrics, // [total_energy, conservation, equilibration]
    int num_samples
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float total_e = *kinetic_energy + *potential_energy;

        // Energy conservation metric (should be ~constant)
        float energy_variance = 0.0f; // Would need history for this

        // Temperature equilibration
        float target_temp = 300.0f; // Room temperature K
        float temp_deviation = fabsf(*temperature - target_temp) / target_temp;

        metrics[0] = total_e;
        metrics[1] = 1.0f - energy_variance; // Conservation quality
        metrics[2] = 1.0f - temp_deviation;  // Equilibration quality
    }
}