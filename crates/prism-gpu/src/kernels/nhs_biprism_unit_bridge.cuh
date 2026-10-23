#pragma once
#include "nhs_biprism_variational_so3g_v10.cuh"
#include "nhs_biprism_coordinate_precision.cuh"

// Native engine: Angstrom, ps, kcal/mol/Angstrom. v10 working planes:
// bohr, atomic time, Hartree/bohr. Coefficients are explicit numerical-model
// inputs; this bridge does not establish a GFN parameterization or an energy gate.
// CODATA2022 factors are independently bound in MASS_AND_UNIT_PREFLIGHT.json.
static constexpr double BIPRISM_ANGSTROM_TO_BOHR = 1.8897261259077824;
static constexpr double BIPRISM_NATIVE_VELOCITY_TO_AU = 4.571028907499172e-5;
static constexpr double BIPRISM_NATIVE_FORCE_TO_AU = 0.0008432975634897884;
static constexpr double BIPRISM_PS_TO_ATOMIC_TIME = 41341.37333517015;
static constexpr uint32_t BIPRISM_BRIDGE_MAGIC = 0x42554231u;
static constexpr uint32_t BIPRISM_BRIDGE_DOMAIN = 1u;
static constexpr uint32_t BIPRISM_BRIDGE_EPOCH = 2u;
static constexpr uint32_t BIPRISM_BRIDGE_NONFINITE = 4u;
static constexpr uint32_t BIPRISM_BRIDGE_CONVERGENCE = 8u;
static constexpr uint32_t BIPRISM_BRIDGE_TERMINAL = 0x40000000u;

// One retained record per leaf. last_core_committed is a convergence counter,
// not a protocol clock. All faults are sticky for the allocation lifetime.
struct __align__(16) BiprismUnitBridgeStatusFfi {
    uint32_t abi_magic, leaf_id, sticky_fault, imported_step;
    uint32_t imported_fields, export_ready_step, export_ready_kind, exported_fields;
    uint32_t import_calls, export_calls, atom_count, last_core_committed;
    uint32_t reserved[4];
};
static_assert(sizeof(BiprismUnitBridgeStatusFfi) == 64, "unit bridge status ABI");

__device__ __forceinline__ bool biprism_bridge_ptrs_ok(
    const FoliatedStreamPtrTableFfi* native,
    const FoliatedStreamPtrTableFfi* au, uint32_t leaf) {
    return native && au && native->d_coords_xyz[leaf] && native->d_vels_xyz[leaf]
        && native->d_forces_xyz[leaf] && au->d_coords_xyz[leaf]
        && au->d_vels_xyz[leaf] && au->d_forces_xyz[leaf];
}

__device__ __forceinline__ bool biprism_bridge_coordinate_owners_ok(
    const FoliatedStreamPtrTableFfi* native,
    const FoliatedStreamPtrTableFfi* au, uint32_t leaf, uint32_t atom_count) {
    if (!native || !au || !native->d_coordinate_owners || !au->d_coordinate_owners
        || (reinterpret_cast<uintptr_t>(native->d_coordinate_owners) & 15u)
        || (reinterpret_cast<uintptr_t>(au->d_coordinate_owners) & 15u)) return false;
    const auto& n = native->d_coordinate_owners[leaf];
    const auto& a = au->d_coordinate_owners[leaf];
    return n.generation == a.generation
        && bp_coord_owner_valid(&n, leaf, atom_count, n.generation,
            BP_COORD_NATIVE_ANGSTROM, 7u, native->d_coords_xyz[leaf])
        && bp_coord_owner_valid(&a, leaf, atom_count, n.generation,
            BP_COORD_ATOMIC_BOHR, 7u, au->d_coords_xyz[leaf]);
}

// Exactly eight CTAs. All full-N planes are distinct owned allocations; the
// physical owner validates their extents/nonaliasing and serializes producers.
// First import/new step uses mask7. Same-step velocity refresh uses mask2 and
// retains accepted core export/counter state. Mask3 refreshes the final native
// endpoint for observation after both exports; it writes scratch only.
extern "C" __global__ __launch_bounds__(256, 2)
void nhs_biprism_import_native_state_v1(
    const FoliatedStreamPtrTableFfi* __restrict__ native,
    const FoliatedStreamPtrTableFfi* __restrict__ au,
    BiprismUnitBridgeStatusFfi* __restrict__ status,
    const uint32_t* __restrict__ protocol_step,
    uint32_t atom_count, uint32_t fields) {
    if (gridDim.x != 8u || blockDim.x != 256u || !status) return;
    const uint32_t leaf = blockIdx.x, tid = threadIdx.x;
    BiprismUnitBridgeStatusFfi& s = status[leaf];
    if (tid == 0u) {
        uint32_t fault = 0u;
        const bool fresh = s.abi_magic == 0u;
        if (!protocol_step || !biprism_bridge_ptrs_ok(native, au, leaf)
            || !biprism_bridge_coordinate_owners_ok(native, au, leaf, atom_count)
            || atom_count < 256u || atom_count > 0x01000000u
            || (fields != 7u && fields != 2u && fields != 3u)) fault |= BIPRISM_BRIDGE_DOMAIN;
        if (!fresh && (s.abi_magic != BIPRISM_BRIDGE_MAGIC || s.leaf_id != leaf
            || s.atom_count != atom_count)) fault |= BIPRISM_BRIDGE_DOMAIN;
        if (!fault) {
            const uint32_t step = *protocol_step;
            if (fields == 7u) {
                const bool cold = !fresh && s.import_calls == 1u && s.last_core_committed == 0u;
                if (!fresh && (s.imported_step == 0xffffffffu
                    || step != s.imported_step + 1u || (!cold && s.exported_fields != 3u)))
                    fault |= BIPRISM_BRIDGE_EPOCH;
                else {
                    s.imported_step = step;
                    s.imported_fields = 0u;
                    s.exported_fields = 0u;
                    s.export_ready_kind = 0u;
                }
            } else if (fields == 2u) {
                if (fresh || step != s.imported_step || s.imported_fields != 7u
                    || s.exported_fields != 1u) fault |= BIPRISM_BRIDGE_EPOCH;
            } else if (fresh || step != s.imported_step || s.imported_fields != 15u
                || s.exported_fields != 3u) fault |= BIPRISM_BRIDGE_EPOCH;
        }
        if (fresh) { s.abi_magic = BIPRISM_BRIDGE_MAGIC; s.leaf_id = leaf; s.atom_count = atom_count; }
        s.sticky_fault |= fault;
    }
    __syncthreads();
    if (s.sticky_fault) return;
    if (fields & 1u) {
        const auto& n = native->d_coordinate_owners[leaf];
        const auto& a = au->d_coordinate_owners[leaf];
        for (uint32_t atom = tid; atom < atom_count; atom += 256u) {
            const auto current = bp_coord_scale(bp_coord_load(n, BiprismCoordinatePlane::Current, atom),
                BIPRISM_ANGSTROM_TO_BOHR);
            const auto reference = bp_coord_scale(bp_coord_load(n, BiprismCoordinatePlane::MeshReference, atom),
                BIPRISM_ANGSTROM_TO_BOHR);
            if (!bp_coord_finite(current) || !bp_coord_finite(reference)) {
                atomicOr(&s.sticky_fault, BIPRISM_BRIDGE_NONFINITE);
                continue;
            }
            if (!bp_coord_store_current(a, atom, current)) {
                atomicOr(&s.sticky_fault, BIPRISM_BRIDGE_NONFINITE);
                continue;
            }
            if (fields == 7u) {
                // AU x_n is the actual pre-core import. Native DriftSaveXn is
                // later in the body and owns its independent native snapshot.
                const BiprismCoordinatePairF32 c[3] = {current.x, current.y, current.z};
                const BiprismCoordinatePairF32 r[3] = {reference.x, reference.y, reference.z};
                for (uint32_t d = 0u; d < 3u; ++d) {
                    const uint32_t i = 3u * atom + d;
                    a.step_start_hi[i] = c[d].hi; a.step_start_lo[i] = c[d].lo;
                    a.reference_hi[i] = r[d].hi; a.reference_lo[i] = r[d].lo;
                }
            }
        }
    }
    for (uint32_t i = tid; i < atom_count * 3u; i += 256u) {
        if (fields & 2u) {
            const float v = native->d_vels_xyz[leaf][i];
            const float out = (float)((double)v * BIPRISM_NATIVE_VELOCITY_TO_AU);
            if (!isfinite(v) || !isfinite(out)) atomicOr(&s.sticky_fault, BIPRISM_BRIDGE_NONFINITE);
            else au->d_vels_xyz[leaf][i] = out;
        }
        if (fields & 4u) {
            const float f = native->d_forces_xyz[leaf][i];
            const float out = (float)((double)f * BIPRISM_NATIVE_FORCE_TO_AU);
            if (!isfinite(f) || !isfinite(out)) atomicOr(&s.sticky_fault, BIPRISM_BRIDGE_NONFINITE);
            else au->d_forces_xyz[leaf][i] = out;
        }
    }
    __syncthreads();
    if (tid == 0u && !s.sticky_fault) { s.imported_fields |= (fields == 2u ? 8u : fields == 3u ? 16u : 7u); ++s.import_calls; }
}

// Separate ordered launch after every import/refresh: reject collectively.
// It does not clear a previous v10/bridge terminal latch.
extern "C" __global__ __launch_bounds__(256, 2)
void nhs_biprism_guard_unit_import_v1(
    const BiprismUnitBridgeStatusFfi* __restrict__ status,
    BiprismConvergenceRecordFfi* __restrict__ convergence,
    const uint32_t* __restrict__ protocol_step) {
    if (blockIdx.x != 0u || threadIdx.x != 0u || !status || !convergence) return;
    uint32_t fault = protocol_step ? 0u : BIPRISM_BRIDGE_DOMAIN;
    for (uint32_t r = 0u; r < 8u; ++r) {
        const auto& s = status[r];
        fault |= s.sticky_fault;
        const uint32_t expected_fields = s.exported_fields == 0u ? 7u :
            s.exported_fields == 1u ? 15u : s.exported_fields == 3u ? 31u : 0u;
        if (s.abi_magic != BIPRISM_BRIDGE_MAGIC || s.leaf_id != r || expected_fields == 0u
            || s.imported_fields != expected_fields
            || (protocol_step && s.imported_step != *protocol_step)) fault |= BIPRISM_BRIDGE_EPOCH;
    }
    if (fault) for (uint32_t r = 0u; r < 8u; ++r)
        convergence[r].terminal_fault_latch |= BIPRISM_BRIDGE_TERMINAL;
}

// kind1 exports accepted core192 positions+velocities before the endpoint
// AMBER force. kind2 exports selected256 velocities after current native-V
// refresh and successful carryover delivery. Halo coordinates are never exported.
// One CTA checks all eight leaves before a separate commit can touch native state.
extern "C" __global__ __launch_bounds__(256, 2)
void nhs_biprism_validate_unit_export_v1(
    const FoliatedStreamPtrTableFfi* __restrict__ native,
    const FoliatedStreamPtrTableFfi* __restrict__ au,
    const ContactShellSimplicialStateFfi* __restrict__ tiles,
    const BiprismConvergenceRecordFfi* __restrict__ convergence,
    BiprismUnitBridgeStatusFfi* __restrict__ status,
    const uint32_t* __restrict__ protocol_step, uint32_t atom_count, uint32_t kind) {
    if (gridDim.x != 1u || blockDim.x != 256u || !status) return;
    const uint32_t tid = threadIdx.x;
    __shared__ uint32_t fault;
    if (tid == 0u) {
        fault = (!tiles || !convergence || !protocol_step || (kind != 1u && kind != 2u))
            ? BIPRISM_BRIDGE_DOMAIN : 0u;
        if (!fault) for (uint32_t r = 0u; r < 8u; ++r) {
            const auto& s = status[r]; const auto& c = convergence[r];
            fault |= s.sticky_fault;
            if (!biprism_bridge_ptrs_ok(native, au, r)
                || !biprism_bridge_coordinate_owners_ok(native, au, r, atom_count)
                || s.abi_magic != BIPRISM_BRIDGE_MAGIC
                || s.leaf_id != r || s.atom_count != atom_count) fault |= BIPRISM_BRIDGE_DOMAIN;
            if (s.imported_step != *protocol_step || s.imported_fields != (kind == 1u ? 7u : 15u)
                || (s.exported_fields & kind) != 0u
                || (kind == 2u && (s.exported_fields & 1u) == 0u)) fault |= BIPRISM_BRIDGE_EPOCH;
            const uint32_t increment = kind == 1u ? 4u : 0u;
            if (s.last_core_committed > 0xffffffffu - increment
                || c.committed_substeps != s.last_core_committed + increment
                || c.abi_magic != BIPRISM_TAIL_ABI_MAGIC || c.leaf_id != r
                || c.terminal_fault_latch || c.epoch_cluster_fault || c.epoch_leaf_fault)
                fault |= BIPRISM_BRIDGE_CONVERGENCE;
        }
    }
    __syncthreads();
    if (!fault) {
        for (uint32_t r = 0u; r < 8u; ++r) {
            const uint32_t atom = UNPACK_GLOBAL_ATOM_ID(tiles[r].global_atom_idx[tid]);
            if (atom >= atom_count) atomicOr(&fault, BIPRISM_BRIDGE_DOMAIN);
            else if (kind == 2u || tid < 192u) for (uint32_t d = 0u; d < 3u; ++d) {
                const float v = au->d_vels_xyz[r][atom * 3u + d];
                bool valid_v = isfinite(v);
                if (kind == 1u) valid_v = valid_v
                    && isfinite((float)((double)v / BIPRISM_NATIVE_VELOCITY_TO_AU));
                else {
                    const float v0 = native->d_vels_xyz[r][atom * 3u + d];
                    const float v0_au = (float)((double)v0 * BIPRISM_NATIVE_VELOCITY_TO_AU);
                    const double dv_au = (double)v - (double)v0_au;
                    const float proposed = dv_au == 0.0 ? v0 :
                        (float)((double)v0 + dv_au / BIPRISM_NATIVE_VELOCITY_TO_AU);
                    valid_v = valid_v && isfinite(v0) && isfinite(v0_au) && isfinite(proposed);
                }
                if (!valid_v) atomicOr(&fault, BIPRISM_BRIDGE_NONFINITE);
                if (kind == 1u && d == 0u) {
                    const auto x = bp_coord_load(au->d_coordinate_owners[r], BiprismCoordinatePlane::Current, atom);
                    if (!bp_coord_finite(x) || !bp_coord_finite(bp_coord_divide(x, BIPRISM_ANGSTROM_TO_BOHR)))
                        atomicOr(&fault, BIPRISM_BRIDGE_NONFINITE);
                }
            }
        }
    }
    __syncthreads();
    if (tid < 8u) {
        status[tid].sticky_fault |= fault;
        status[tid].export_ready_kind = fault ? 0u : kind;
        if (!fault) status[tid].export_ready_step = *protocol_step;
    }
}

extern "C" __global__ __launch_bounds__(256, 2)
void nhs_biprism_commit_unit_export_v1(
    const FoliatedStreamPtrTableFfi* __restrict__ native,
    const FoliatedStreamPtrTableFfi* __restrict__ au,
    const ContactShellSimplicialStateFfi* __restrict__ tiles,
    const BiprismConvergenceRecordFfi* __restrict__ convergence,
    BiprismUnitBridgeStatusFfi* __restrict__ status,
    const uint32_t* __restrict__ protocol_step, uint32_t atom_count, uint32_t kind) {
    if (gridDim.x != 8u || blockDim.x != 256u || !status || !protocol_step) return;
    const uint32_t leaf = blockIdx.x, tid = threadIdx.x;
    // Ready flags are immutable throughout this launch. The next validation
    // clears/replaces them; do not clear them in one CTA while other CTAs read.
    for (uint32_t r = 0u; r < 8u; ++r) {
        if (status[r].sticky_fault || status[r].export_ready_step != *protocol_step
            || status[r].export_ready_kind != kind) return;
    }
    if ((status[leaf].exported_fields & kind) != 0u) return;
    const uint32_t atom = UNPACK_GLOBAL_ATOM_ID(tiles[leaf].global_atom_idx[tid]);
    if (kind == 2u || tid < 192u) for (uint32_t d = 0u; d < 3u; ++d) {
        if (kind == 1u) {
            native->d_vels_xyz[leaf][atom * 3u + d] =
                (float)((double)au->d_vels_xyz[leaf][atom * 3u + d] / BIPRISM_NATIVE_VELOCITY_TO_AU);
        } else {
            // Transfer only the delivered impulse. A zero AU change preserves
            // native velocity bits rather than forcing a lossy unit roundtrip.
            const float v0 = native->d_vels_xyz[leaf][atom * 3u + d];
            const float v0_au = (float)((double)v0 * BIPRISM_NATIVE_VELOCITY_TO_AU);
            const double dv_au = (double)au->d_vels_xyz[leaf][atom * 3u + d] - (double)v0_au;
            if (dv_au != 0.0) native->d_vels_xyz[leaf][atom * 3u + d] =
                (float)((double)v0 + dv_au / BIPRISM_NATIVE_VELOCITY_TO_AU);
        }
    }
    if (kind == 1u && tid < 192u) {
        const auto x = bp_coord_load(au->d_coordinate_owners[leaf], BiprismCoordinatePlane::Current, atom);
        const auto out = bp_coord_divide(x, BIPRISM_ANGSTROM_TO_BOHR);
        if (!bp_coord_store_current(native->d_coordinate_owners[leaf], atom, out))
            atomicOr(&status[leaf].sticky_fault, BIPRISM_BRIDGE_NONFINITE);
    }
    __syncthreads();
    if (tid == 0u) {
        status[leaf].exported_fields |= kind;
        ++status[leaf].export_calls;
        if (kind == 1u) status[leaf].last_core_committed = convergence[leaf].committed_substeps;
    }
}
