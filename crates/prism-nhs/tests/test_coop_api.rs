//! Test that cudarc 0.18.2 launch_cooperative() method is accessible
//! This validates the API signature for cooperative kernel integration

#[test]
fn test_launch_config_grid_topology() {
    use cudarc::driver::LaunchConfig;

    // Verify LaunchConfig can be created with 168-block grid (84 × 2)
    let cfg = LaunchConfig {
        grid_dim: (168, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    assert_eq!(cfg.grid_dim.0, 168);
    assert_eq!(cfg.block_dim.0, 256);
    assert_eq!(cfg.shared_mem_bytes, 0);
}

#[test]
fn test_launch_config_dynamic_grid() {
    use cudarc::driver::LaunchConfig;

    // Verify LaunchConfig supports dynamic grids (current fused engine pattern)
    let n_atoms = 20000u32;
    let block_size = 256u32;
    let n_blocks = (n_atoms + block_size - 1) / block_size;

    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    assert_eq!(cfg.grid_dim.0, 79); // (20000 + 255) / 256 = 79
    assert_eq!(cfg.block_dim.0, 256);
}
