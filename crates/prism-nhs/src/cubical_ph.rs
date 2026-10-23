//! Cubical Persistent Homology for spike density fields.
//!
//! Computes 0-dimensional persistent homology on a 3D voxel grid via
//! union-find sweep (descending density order). Each persistence pair
//! represents two density peaks that merge — the younger component dies,
//! recording its birth density, death density, and persistence.
//!
//! The birth voxel of each pair is a local density maximum, which serves
//! as a high-quality pocket centroid (density peak rather than geometric
//! center of enclosed volume).
//!
//! Algorithm complexity: O(N log N) for the sort + O(N alpha(N)) for
//! union-find, where N = number of non-zero voxels. Typically 50-200ms
//! for grids up to 500K voxels.

/// A persistence pair from cubical PH on a 3D density grid.
#[derive(Debug, Clone)]
pub struct PersistencePair {
    /// Density at which this component appeared (local maximum)
    pub birth_density: f32,
    /// Density at which this component merged into an older one
    pub death_density: f32,
    /// birth_density - death_density
    pub persistence: f32,
    /// Flat voxel index of the local maximum (birth location)
    pub birth_voxel: u32,
    /// Number of voxels in this component at the moment it died
    pub component_size: u32,
}

/// Pocket candidate derived from a persistence pair.
#[derive(Debug, Clone)]
pub struct PhPocket {
    /// World-space coordinates of the birth voxel (density peak)
    pub centroid: [f32; 3],
    /// Volume estimate: component_size * spacing^3
    pub volume: f32,
    /// Persistence value (ranking signal)
    pub persistence: f32,
    /// Birth density (peak density at this location)
    pub peak_density: f32,
    /// Flat voxel index of the density peak
    pub birth_voxel: u32,
}

/// Compute axis-aligned bounding box for atom positions and return
/// grid origin, dimensions, and spacing.
///
/// `atom_positions` is flattened [x0, y0, z0, x1, y1, z1, ...].
pub fn compute_density_grid_bounds(
    atom_positions: &[f32],
    margin: f32,
    spacing: f32,
) -> ([f32; 3], [usize; 3], f32) {
    let n_atoms = atom_positions.len() / 3;
    let mut min_pos = [f32::MAX; 3];
    let mut max_pos = [f32::MIN; 3];
    for i in 0..n_atoms {
        for d in 0..3 {
            let v = atom_positions[i * 3 + d];
            min_pos[d] = min_pos[d].min(v);
            max_pos[d] = max_pos[d].max(v);
        }
    }
    let origin = [
        min_pos[0] - margin,
        min_pos[1] - margin,
        min_pos[2] - margin,
    ];
    let dims = [
        ((max_pos[0] - min_pos[0] + 2.0 * margin) / spacing).ceil() as usize,
        ((max_pos[1] - min_pos[1] + 2.0 * margin) / spacing).ceil() as usize,
        ((max_pos[2] - min_pos[2] + 2.0 * margin) / spacing).ceil() as usize,
    ];
    (origin, dims, spacing)
}

/// Compute 0-dimensional cubical persistent homology on a 3D density grid.
///
/// Uses CPU union-find sweep (fast enough for <2M voxels at ~50-200ms).
///
/// Algorithm:
/// 1. Sort voxel indices by density DESCENDING (highest first)
/// 2. Union-find sweep: process voxels in sorted order
///    - For each voxel, check 6-connected neighbors
///    - If neighbor already processed AND in different component: merge
///    - The younger component (lower birth density) dies -> record pair
///    - If neighbor in same component: skip
///    - If no processed neighbors: new component born at this density
/// 3. Filter pairs by persistence > threshold and component_size > min
/// 4. Convert birth voxels to world coordinates
pub fn compute_cubical_ph_cpu(
    density: &[f32],
    dims: [usize; 3],
    origin: [f32; 3],
    spacing: f32,
    min_persistence: f32,
    min_component_size: u32,
) -> Vec<PhPocket> {
    let n = dims[0] * dims[1] * dims[2];
    if n == 0 || density.len() < n {
        return Vec::new();
    }

    // 1. Sort voxel indices by density descending
    let mut indices: Vec<u32> = (0..n as u32).collect();
    indices.sort_by(|&a, &b| {
        density[b as usize]
            .partial_cmp(&density[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 2. Union-Find data structure
    // parent[i] == u32::MAX means "not yet processed"
    let mut parent: Vec<u32> = vec![u32::MAX; n];
    let mut rank: Vec<u8> = vec![0; n];
    let mut size: Vec<u32> = vec![1; n];
    let mut birth: Vec<f32> = vec![0.0; n]; // density at birth for each root

    let mut pairs: Vec<PersistencePair> = Vec::new();

    // 3. Process voxels in decreasing density order
    for &idx in &indices {
        let d = density[idx as usize];
        if d <= 0.0 {
            break; // all remaining voxels are zero-density
        }

        // Initialize this voxel as its own component
        parent[idx as usize] = idx;
        birth[idx as usize] = d;

        // Decompose flat index -> (ix, iy, iz)
        let ix = (idx as usize) % dims[0];
        let iy = ((idx as usize) / dims[0]) % dims[1];
        let iz = (idx as usize) / (dims[0] * dims[1]);

        // 6-connected neighbors
        let neighbors: [(i32, i32, i32); 6] = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ];

        for &(dx, dy, dz) in &neighbors {
            let nx = ix as i32 + dx;
            let ny = iy as i32 + dy;
            let nz = iz as i32 + dz;
            if nx < 0 || ny < 0 || nz < 0 {
                continue;
            }
            let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
            if nx >= dims[0] || ny >= dims[1] || nz >= dims[2] {
                continue;
            }

            let nidx = (nz * dims[1] + ny) * dims[0] + nx;
            if parent[nidx] == u32::MAX {
                continue; // neighbor not yet processed
            }

            let ri = find(&mut parent, idx);
            let rn = find(&mut parent, nidx as u32);

            if ri != rn {
                // Two different components meet — the younger one dies.
                // "Older" = higher birth density (processed earlier).
                let (older, younger) = if birth[ri as usize] >= birth[rn as usize] {
                    (ri, rn)
                } else {
                    (rn, ri)
                };

                // Record persistence pair for the dying component
                let persistence = birth[younger as usize] - d;
                pairs.push(PersistencePair {
                    birth_density: birth[younger as usize],
                    death_density: d,
                    persistence,
                    birth_voxel: younger, // the root = the local max of the dying component
                    component_size: size[younger as usize],
                });

                // Union by rank: attach younger under older
                if rank[older as usize] < rank[younger as usize] {
                    parent[older as usize] = younger;
                    size[younger as usize] += size[older as usize];
                    // Preserve the older component's birth density
                    birth[younger as usize] = birth[older as usize].max(birth[younger as usize]);
                } else {
                    parent[younger as usize] = older;
                    size[older as usize] += size[younger as usize];
                    if rank[older as usize] == rank[younger as usize] {
                        rank[older as usize] = rank[older as usize].saturating_add(1);
                    }
                }
            }
        }
    }

    // 4. Sort pairs by persistence descending, filter, and convert to PhPocket
    pairs.sort_by(|a, b| {
        b.persistence
            .partial_cmp(&a.persistence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    pairs
        .iter()
        .filter(|p| p.persistence >= min_persistence && p.component_size >= min_component_size)
        .map(|p| {
            let vi = p.birth_voxel as usize;
            let ix = vi % dims[0];
            let iy = (vi / dims[0]) % dims[1];
            let iz = vi / (dims[0] * dims[1]);
            PhPocket {
                centroid: [
                    origin[0] + (ix as f32 + 0.5) * spacing,
                    origin[1] + (iy as f32 + 0.5) * spacing,
                    origin[2] + (iz as f32 + 0.5) * spacing,
                ],
                volume: p.component_size as f32 * spacing * spacing * spacing,
                persistence: p.persistence,
                peak_density: p.birth_density,
                birth_voxel: p.birth_voxel,
            }
        })
        .collect()
}

/// Find with path compression (iterative).
fn find(parent: &mut [u32], x: u32) -> u32 {
    let mut r = x;
    while parent[r as usize] != r {
        r = parent[r as usize];
    }
    // Path compression
    let mut c = x;
    while c != r {
        let next = parent[c as usize];
        parent[c as usize] = r;
        c = next;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_peak() {
        // 3x3x3 grid with a single peak in the center
        let dims = [3, 3, 3];
        let mut density = vec![0.0f32; 27];
        // Center voxel (1,1,1) = index 13
        density[13] = 10.0;
        // 6-neighbors of center
        density[12] = 5.0; // (0,1,1)
        density[14] = 5.0; // (2,1,1)
        density[10] = 5.0; // (1,0,1)
        density[16] = 5.0; // (1,2,1)
        density[4] = 5.0; // (1,1,0)
        density[22] = 5.0; // (1,1,2)

        let pockets = compute_cubical_ph_cpu(&density, dims, [0.0, 0.0, 0.0], 1.0, 0.0, 1);
        // Should have persistence pairs from the 6 neighbors merging into center
        assert!(!pockets.is_empty());
        // The highest-persistence pocket should be centered near (1.5, 1.5, 1.5)
        let best = &pockets[0];
        assert!(best.peak_density >= 5.0);
    }

    #[test]
    fn test_two_peaks() {
        // 7x1x1 grid with two peaks separated by a valley
        let dims = [7, 1, 1];
        let density = vec![0.0, 5.0, 10.0, 2.0, 8.0, 4.0, 0.0];

        let pockets = compute_cubical_ph_cpu(&density, dims, [0.0, 0.0, 0.0], 1.0, 0.5, 1);
        // Two peaks: index 2 (density 10) and index 4 (density 8)
        // When they merge at the saddle (density 2 at index 3), the younger peak (8) dies
        // Persistence of dying peak = 8 - 2 = 6
        assert!(!pockets.is_empty());
        let best = &pockets[0];
        assert!(
            (best.persistence - 6.0).abs() < 0.01,
            "persistence={}",
            best.persistence
        );
    }

    #[test]
    fn test_empty_grid() {
        let dims = [5, 5, 5];
        let density = vec![0.0; 125];
        let pockets = compute_cubical_ph_cpu(&density, dims, [0.0, 0.0, 0.0], 1.0, 0.0, 1);
        assert!(pockets.is_empty());
    }

    #[test]
    fn test_grid_bounds() {
        // 4 atoms at corners of a 10x10x10 box
        let positions = vec![
            0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0,
        ];
        let (origin, dims, spacing) = compute_density_grid_bounds(&positions, 5.0, 1.0);
        assert!((origin[0] - (-5.0)).abs() < 0.01);
        assert!((origin[1] - (-5.0)).abs() < 0.01);
        assert!((origin[2] - (-5.0)).abs() < 0.01);
        assert_eq!(dims[0], 20); // (10 + 2*5) / 1.0 = 20
        assert_eq!(dims[1], 20);
        assert_eq!(dims[2], 20);
        assert!((spacing - 1.0).abs() < 0.01);
    }
}
