//! LBVH Phase 2 — Karras Radix Tree Builder + Bottom-Up AABB Reduce.
//!
//! Per the PRISM-4D Progressive Automation mandate §1 (operator
//! directive 2026-04-29). Linear Bounding Volume Hierarchy via
//! Karras 2012, upgraded for Blackwell sm_120 with:
//!
//! - **64-bit composite sort keys** `[30-bit Morton | 32-bit
//!   residue_id | 2 high bits unused]`. Identity tie-breaker per
//!   Mandate §M2 — eliminates the zero-length-split case under
//!   Morton collisions (two spikes at the same voxel hash to
//!   distinct keys via residue_id).
//!
//! - **64-byte cache-line-aligned [`LBVHNode`]**. Layout-pinned by
//!   the `lbvh_node_layout_is_64_bytes_64_aligned` test. The
//!   alignment guarantees the LDG.E.128 wide-load instruction can
//!   pull `aabb_min` / `aabb_max` (each `float4`) into registers in
//!   a single transaction during the bottom-up reduce.
//!
//! - **Last-arrival atomic-flag bottom-up AABB reduce**. Each
//!   internal node has a `u32` atomic counter; first child to arrive
//!   increments-and-exits; second child merges the two children's
//!   AABBs and propagates up the tree. Race-free under Blackwell's
//!   L2-coherent atomic hardware.
//!
//! # Caller contract
//!
//! 1. Compute per-spike 30-bit Morton codes via [`crate::lbvh`]
//!    (Phase 1).
//! 2. Compose 64-bit sort keys via [`compose_sort_key`] (one per
//!    spike: Morton + residue_id).
//! 3. Sort keys ascending (host-side `Vec::sort` for tests; CUB
//!    `DeviceRadixSort::SortPairs` for production paths).
//! 4. Allocate `n_leaves - 1` [`LBVHNode`]s + `n_leaves` `i32`
//!    leaf-parent slots in the F2 stream-ordered pool.
//! 5. Call [`init_internal_nodes`] on the freshly-allocated nodes.
//! 6. Call [`karras_build`] to construct the tree topology.
//! 7. Call [`aabb_reduce`] to propagate per-leaf AABBs up to the root.
//! 8. Read `internal_nodes[0]` (the root) for the global AABB; read
//!    deeper nodes for site-level AABBs.

use serde::{Deserialize, Serialize};

// ============================================================================
// LBVHNode — 64-byte cache-line-aligned tree node
// ============================================================================

/// Mirror of the C-side `LBVHNode` in `lbvh_tree.cuh`. Layout-pinned
/// by the `lbvh_node_layout_is_64_bytes_64_aligned` test below.
///
/// Field offsets:
///   0   parent_idx (i32)
///   4   left_child (i32) — Karras unified index
///   8   right_child (i32)
///   12  atomic_flag (u32)
///   16  aabb_min (4 × f32, 16-byte aligned)
///   32  aabb_max (4 × f32, 16-byte aligned)
///   48  metadata (u64)
///   56  _pad (8 bytes)
///   64  (end)
///
/// **Karras child encoding**: `left_child` / `right_child` carry a
/// unified index. For a tree with `n_leaves`, `n_internal = n_leaves - 1`.
/// A child index `< n_internal` refers to an internal node; `>= n_internal`
/// refers to leaf `(child - n_internal)`.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LBVHNode {
    pub parent_idx: i32,
    pub left_child: i32,
    pub right_child: i32,
    pub atomic_flag: u32,
    pub aabb_min: [f32; 4],
    pub aabb_max: [f32; 4],
    pub metadata: u64,
    pub _pad: [u8; 8],
}

impl LBVHNode {
    /// Sentinel for `parent_idx`: this node is the root (no parent).
    pub const PARENT_NONE: i32 = -1;
    /// Sentinel for `left_child` / `right_child` before tree build.
    pub const CHILD_MISSING: i32 = -1;

    /// All-sentinel initial state. Matches the effect of
    /// [`init_internal_nodes`] on the device.
    pub const fn uninit() -> Self {
        Self {
            parent_idx: Self::PARENT_NONE,
            left_child: Self::CHILD_MISSING,
            right_child: Self::CHILD_MISSING,
            atomic_flag: 0,
            aabb_min: [f32::MAX; 4],
            aabb_max: [f32::MIN; 4],
            metadata: 0,
            _pad: [0; 8],
        }
    }
}

// ============================================================================
// Composite sort-key helpers
// ============================================================================

/// Compose a 64-bit sort key from a 30-bit Morton code and a 32-bit
/// residue id.
///
/// Layout (MSB → LSB):
///   bits 63..62 — reserved (currently 0)
///   bits 61..32 — 30-bit Morton code
///   bits 31..0  — 32-bit residue id (signed-as-unsigned)
///
/// Bit-equivalent to `prism_lbvh_compose_sort_key` in `lbvh_tree.cuh`.
#[inline]
pub fn compose_sort_key(morton30: u32, residue_id: i32) -> u64 {
    let m = (morton30 as u64) & 0x3FFFFFFF;
    let r = residue_id as u32 as u64;
    (m << 32) | r
}

/// Extract the 30-bit Morton component from a composite key.
#[inline]
pub fn extract_morton(key: u64) -> u32 {
    ((key >> 32) & 0x3FFFFFFF) as u32
}

/// Extract the 32-bit residue id from a composite key (raw u32; cast
/// to i32 if the original was signed).
#[inline]
pub fn extract_residue_raw(key: u64) -> u32 {
    (key & 0xFFFFFFFF) as u32
}

// ============================================================================
// FFI surface
// ============================================================================

#[cfg(feature = "gpu")]
#[allow(dead_code)]
mod ffi {
    use super::LBVHNode;

    pub type CudaError = i32;
    pub const CUDA_SUCCESS: CudaError = 0;

    extern "C" {
        pub fn prism_lbvh_tree_link_probe() -> u32;

        pub fn prism_lbvh_init_internal_nodes(
            d_internal_nodes: *mut LBVHNode,
            n_internal: u32,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_lbvh_karras_build(
            d_sorted_keys: *const u64,
            n_leaves: u32,
            d_internal_nodes: *mut LBVHNode,
            d_leaf_parents: *mut i32,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_lbvh_aabb_reduce(
            d_internal_nodes: *mut LBVHNode,
            n_internal: u32,
            d_leaf_parents: *const i32,
            d_leaf_positions: *const f32,
            n_leaves: u32,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;
    }
}

#[cfg(feature = "gpu")]
pub fn link_probe() -> u32 {
    unsafe { ffi::prism_lbvh_tree_link_probe() }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lbvh_node_layout_is_64_bytes_64_aligned() {
        // The C-side static_assert pins sizeof == 64 / alignof == 64.
        // Layout drift is a Blackwell strict-alignment trap vector.
        assert_eq!(std::mem::size_of::<LBVHNode>(), 64);
        assert_eq!(std::mem::align_of::<LBVHNode>(), 64);
    }

    #[test]
    fn lbvh_node_field_offsets() {
        let n = LBVHNode::uninit();
        let base = &n as *const LBVHNode as usize;
        macro_rules! ofs {
            ($field:ident) => {
                (&n.$field as *const _ as usize) - base
            };
        }
        assert_eq!(ofs!(parent_idx), 0);
        assert_eq!(ofs!(left_child), 4);
        assert_eq!(ofs!(right_child), 8);
        assert_eq!(ofs!(atomic_flag), 12);
        assert_eq!(ofs!(aabb_min), 16);
        assert_eq!(ofs!(aabb_max), 32);
        assert_eq!(ofs!(metadata), 48);
    }

    #[test]
    fn compose_extract_round_trip() {
        for &(m, r) in &[
            (0u32, 0i32),
            (0x3FFFFFFF, 0i32), // max Morton (30 bits)
            (0u32, -1i32),      // sentinel residue
            (0x3FFFFFFF, i32::MAX),
            (0x3FFFFFFF, i32::MIN),
            (12345, 678),
        ] {
            let k = compose_sort_key(m, r);
            assert_eq!(
                extract_morton(k),
                m & 0x3FFFFFFF,
                "Morton round-trip failed for ({:#x}, {})",
                m,
                r
            );
            assert_eq!(
                extract_residue_raw(k) as i32,
                r,
                "residue round-trip failed for ({:#x}, {})",
                m,
                r
            );
        }
    }

    #[test]
    fn compose_orders_by_morton_first_then_residue() {
        // Morton-major ordering is the foundation of the LBVH's
        // spatial locality. Two spikes with different Morton must
        // sort by Morton regardless of residue.
        let k_low_morton = compose_sort_key(10, 9999);
        let k_high_morton = compose_sort_key(11, 0);
        assert!(
            k_low_morton < k_high_morton,
            "lower Morton must produce lower composite key"
        );
        // Same Morton, different residue: residue is the tiebreaker.
        let k_a = compose_sort_key(10, 5);
        let k_b = compose_sort_key(10, 8);
        assert!(
            k_a < k_b,
            "same Morton + lower residue must produce lower key"
        );
    }

    #[test]
    fn compose_morton_high_bits_masked() {
        // Inputs with bits in positions 30..31 are silently masked
        // (Morton is 30-bit; high 2 bits are reserved/unused).
        let with_high = compose_sort_key(0xFFFFFFFF, 0);
        let masked = compose_sort_key(0x3FFFFFFF, 0);
        assert_eq!(
            with_high, masked,
            "high bits 30..31 of Morton input must be masked off"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn link_probe_returns_sentinel() {
        assert_eq!(super::link_probe(), 0x0000_BABE);
    }

    #[cfg(feature = "gpu")]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn karras_8_leaves_topology_is_consistent() {
        // 8 leaves with a stride-1 sorted Morton sequence. The Karras
        // builder must produce a tree where:
        //   - Every leaf has a valid parent (in [0, n_internal)).
        //   - Every internal node's children point to one of: another
        //     internal node OR a leaf.
        //   - Every non-root internal node has exactly one parent.
        //   - The root (internal_nodes[0]) has parent_idx == -1.
        //   - Walking up from any leaf reaches the root.
        //
        // We don't pin the EXACT topology (Karras is deterministic
        // for fixed input but the structure depends on the binary-
        // search arithmetic; manual reproduction is brittle). We
        // pin the INVARIANTS.

        use cudarc::driver::{CudaContext, DevicePtr};
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[lbvh_tree topology] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream");

        // 8 sorted keys with non-trivial Morton diversity.
        let keys: Vec<u64> = (0..8u64)
            .map(|i| {
                // morton = i * 1000 to spread the keys; residue = i.
                compose_sort_key((i as u32) * 1000, i as i32)
            })
            .collect();
        let n_leaves = keys.len() as u32;
        let n_internal = n_leaves - 1;

        let mut d_keys = stream
            .alloc_zeros::<u64>(n_leaves as usize)
            .expect("alloc d_keys");
        stream.memcpy_htod(&keys, &mut d_keys).expect("htod keys");
        // 7 internal nodes × 64 B = 448 B. Allocated as u8 to avoid
        // requiring DeviceRepr on LBVHNode.
        let n_internal_bytes = (n_internal as usize) * std::mem::size_of::<LBVHNode>();
        let d_internal_bytes = stream
            .alloc_zeros::<u8>(n_internal_bytes)
            .expect("alloc d_internal");
        let d_leaf_parents = stream
            .alloc_zeros::<i32>(n_leaves as usize)
            .expect("alloc d_leaf_parents");

        let raw_stream = stream.cu_stream() as usize;
        let (keys_dev, _g1) = d_keys.device_ptr(&stream);
        let (internal_dev, _g2) = d_internal_bytes.device_ptr(&stream);
        let (leaf_parents_dev, _g3) = d_leaf_parents.device_ptr(&stream);

        // 1. Init internal nodes.
        let rc = unsafe {
            ffi::prism_lbvh_init_internal_nodes(
                internal_dev as *mut LBVHNode,
                n_internal,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, ffi::CUDA_SUCCESS, "init failed: rc={}", rc);

        // 2. Build Karras tree.
        let rc = unsafe {
            ffi::prism_lbvh_karras_build(
                keys_dev as *const u64,
                n_leaves,
                internal_dev as *mut LBVHNode,
                leaf_parents_dev as *mut i32,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, ffi::CUDA_SUCCESS, "karras_build failed: rc={}", rc);
        stream.synchronize().expect("stream sync");

        // dtoh and verify invariants.
        let mut nodes_bytes = vec![0u8; n_internal_bytes];
        stream
            .memcpy_dtoh(&d_internal_bytes, &mut nodes_bytes)
            .expect("dtoh nodes");
        let nodes: Vec<LBVHNode> = (0..n_internal as usize)
            .map(|i| unsafe {
                std::ptr::read_unaligned(
                    nodes_bytes
                        .as_ptr()
                        .add(i * std::mem::size_of::<LBVHNode>())
                        as *const LBVHNode,
                )
            })
            .collect();
        let mut leaf_parents = vec![0i32; n_leaves as usize];
        stream
            .memcpy_dtoh(&d_leaf_parents, &mut leaf_parents)
            .expect("dtoh leaf_parents");

        // Invariant 1: root is at index 0 with parent_idx == -1.
        assert_eq!(
            nodes[0].parent_idx,
            LBVHNode::PARENT_NONE,
            "root's parent_idx must be -1"
        );

        // Invariant 2: every internal node has children in valid
        // range (children either internal index in [0, n_internal)
        // or leaf index in [n_internal, 2*n_internal+1)).
        for (i, node) in nodes.iter().enumerate() {
            assert!(
                node.left_child >= 0,
                "internal[{}].left_child must be set: got {}",
                i,
                node.left_child
            );
            assert!(
                node.right_child >= 0,
                "internal[{}].right_child must be set: got {}",
                i,
                node.right_child
            );
            let total_unified = (n_internal + n_leaves) as i32;
            assert!(
                node.left_child < total_unified,
                "internal[{}].left_child = {} out of range [0, {})",
                i,
                node.left_child,
                total_unified
            );
            assert!(
                node.right_child < total_unified,
                "internal[{}].right_child = {} out of range [0, {})",
                i,
                node.right_child,
                total_unified
            );
        }

        // Invariant 3: every leaf has a valid parent.
        for (k, &p) in leaf_parents.iter().enumerate() {
            assert!(
                p >= 0 && (p as u32) < n_internal,
                "leaf[{}].parent_idx = {} out of range [0, {})",
                k,
                p,
                n_internal
            );
        }

        // Invariant 4: walking up from leaf 0 reaches the root.
        let mut steps = 0;
        let mut cur: i32 = leaf_parents[0];
        while cur != LBVHNode::PARENT_NONE {
            assert!(
                cur >= 0 && (cur as u32) < n_internal,
                "walk-up: invalid index {}",
                cur
            );
            cur = nodes[cur as usize].parent_idx;
            steps += 1;
            assert!(
                steps < 100,
                "walk-up didn't terminate within 100 hops (cycle?)"
            );
        }
        assert!(steps > 0, "leaf 0's walk-up was empty");

        // Invariant 5: every NON-root internal has a valid parent.
        for (i, node) in nodes.iter().enumerate().skip(1) {
            // i = 0 is root; skip.
            // Some internals may also be the root in degenerate
            // cases — but for sorted unique keys, internal 0 is the
            // root by Karras convention. Check that THIS node's parent
            // is a different internal.
            if node.parent_idx == LBVHNode::PARENT_NONE {
                // A non-root internal with no parent is a tree-build
                // error.
                panic!("internal[{}] has parent_idx = -1 but it isn't the root", i);
            }
            let p = node.parent_idx;
            assert!(
                p >= 0 && (p as u32) < n_internal,
                "internal[{}].parent_idx = {} out of range",
                i,
                p
            );
        }
    }

    #[cfg(feature = "gpu")]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn aabb_reduce_root_matches_global_bbox() {
        // 16 synthetic positions in a non-trivial spread. Build the
        // tree, reduce AABBs, verify the ROOT'S AABB equals the
        // tight bbox over every leaf position.

        use cudarc::driver::{CudaContext, DevicePtr};
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[lbvh aabb] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream");

        // Build positions with a deterministic LCG for reproducibility.
        let mut s: u64 = 1234;
        let mut next = || -> f32 {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 32) as u32 as f32 / 4_294_967_296.0) * 100.0 - 50.0
        };
        let n_leaves = 16u32;
        let positions: Vec<f32> = (0..(n_leaves as usize) * 3).map(|_| next()).collect();

        // Compute Morton codes via a simple host-side encoder
        // (matches `lbvh::cpu_morton_30bit_encode_position` semantics
        // but we inline a minimal version here to avoid a cross-
        // module test dependency).
        let bbox_min = [-50.0f32, -50.0, -50.0];
        let bbox_max = [50.0f32, 50.0, 50.0];
        let cpu_quantize = |coord: f32, bmin: f32, bmax: f32| -> u32 {
            let span = bmax - bmin;
            let u = ((coord - bmin) / span).clamp(0.0, 1.0);
            ((u * 1023.0 + 0.5) as u32).min(1023)
        };
        let cpu_expand = |mut v: u32| -> u32 {
            v = (v | (v << 16)) & 0x030000FF;
            v = (v | (v << 8)) & 0x0300F00F;
            v = (v | (v << 4)) & 0x030C30C3;
            v = (v | (v << 2)) & 0x09249249;
            v
        };
        let mut keys: Vec<(u64, usize)> = (0..n_leaves as usize)
            .map(|i| {
                let qx = cpu_quantize(positions[i * 3 + 0], bbox_min[0], bbox_max[0]);
                let qy = cpu_quantize(positions[i * 3 + 1], bbox_min[1], bbox_max[1]);
                let qz = cpu_quantize(positions[i * 3 + 2], bbox_min[2], bbox_max[2]);
                let m = cpu_expand(qx) | (cpu_expand(qy) << 1) | (cpu_expand(qz) << 2);
                (compose_sort_key(m, i as i32), i)
            })
            .collect();
        keys.sort_by_key(|&(k, _)| k);

        // Reorder positions according to the sort permutation so leaf
        // index in the LBVH matches the sort order.
        let sorted_keys: Vec<u64> = keys.iter().map(|&(k, _)| k).collect();
        let sorted_positions: Vec<f32> = keys
            .iter()
            .flat_map(|&(_, original_i)| {
                [
                    positions[original_i * 3],
                    positions[original_i * 3 + 1],
                    positions[original_i * 3 + 2],
                ]
            })
            .collect();

        let n_internal = n_leaves - 1;
        let n_internal_bytes = (n_internal as usize) * std::mem::size_of::<LBVHNode>();

        let mut d_keys = stream
            .alloc_zeros::<u64>(n_leaves as usize)
            .expect("alloc d_keys");
        stream
            .memcpy_htod(&sorted_keys, &mut d_keys)
            .expect("htod keys");
        let mut d_positions = stream
            .alloc_zeros::<f32>(sorted_positions.len())
            .expect("alloc d_positions");
        stream
            .memcpy_htod(&sorted_positions, &mut d_positions)
            .expect("htod positions");
        let d_internal_bytes = stream
            .alloc_zeros::<u8>(n_internal_bytes)
            .expect("alloc d_internal");
        let d_leaf_parents = stream
            .alloc_zeros::<i32>(n_leaves as usize)
            .expect("alloc d_leaf_parents");

        let raw_stream = stream.cu_stream() as usize;
        let (keys_dev, _g1) = d_keys.device_ptr(&stream);
        let (positions_dev, _g2) = d_positions.device_ptr(&stream);
        let (internal_dev, _g3) = d_internal_bytes.device_ptr(&stream);
        let (leaf_parents_dev, _g4) = d_leaf_parents.device_ptr(&stream);

        let rc = unsafe {
            ffi::prism_lbvh_init_internal_nodes(
                internal_dev as *mut LBVHNode,
                n_internal,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, ffi::CUDA_SUCCESS, "init failed");

        let rc = unsafe {
            ffi::prism_lbvh_karras_build(
                keys_dev as *const u64,
                n_leaves,
                internal_dev as *mut LBVHNode,
                leaf_parents_dev as *mut i32,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, ffi::CUDA_SUCCESS, "karras_build failed");

        let rc = unsafe {
            ffi::prism_lbvh_aabb_reduce(
                internal_dev as *mut LBVHNode,
                n_internal,
                leaf_parents_dev as *const i32,
                positions_dev as *const f32,
                n_leaves,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, ffi::CUDA_SUCCESS, "aabb_reduce failed");

        stream.synchronize().expect("stream sync");

        // Read back internal node 0 (the root).
        let mut nodes_bytes = vec![0u8; n_internal_bytes];
        stream
            .memcpy_dtoh(&d_internal_bytes, &mut nodes_bytes)
            .expect("dtoh nodes");
        let root: LBVHNode =
            unsafe { std::ptr::read_unaligned(nodes_bytes.as_ptr() as *const LBVHNode) };

        // Compute the host-side bbox over sorted_positions.
        let mut bmin = [f32::INFINITY; 3];
        let mut bmax = [f32::NEG_INFINITY; 3];
        for chunk in sorted_positions.chunks_exact(3) {
            for ax in 0..3 {
                if chunk[ax] < bmin[ax] {
                    bmin[ax] = chunk[ax];
                }
                if chunk[ax] > bmax[ax] {
                    bmax[ax] = chunk[ax];
                }
            }
        }

        // Compare. min/max are exact under the AABB-reduce semantics
        // (no FP arithmetic, just min/max). f32 BitExact match.
        for ax in 0..3 {
            assert_eq!(
                root.aabb_min[ax].to_bits(),
                bmin[ax].to_bits(),
                "root.aabb_min[{}] = {} (bits {:#x}) != host bbox min {} (bits {:#x})",
                ax,
                root.aabb_min[ax],
                root.aabb_min[ax].to_bits(),
                bmin[ax],
                bmin[ax].to_bits()
            );
            assert_eq!(
                root.aabb_max[ax].to_bits(),
                bmax[ax].to_bits(),
                "root.aabb_max[{}] = {} != host bbox max {}",
                ax,
                root.aabb_max[ax],
                bmax[ax]
            );
        }
    }
}
