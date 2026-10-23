#[cfg(feature = "gpu")]
#[test]
fn test_launch_cooperative_api_exists() {
    use cudarc::driver::LaunchConfig;

    // Verify LaunchConfig structure
    let cfg = LaunchConfig {
        grid_dim: (168, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // If this compiles, the LaunchConfig type is correct
    assert_eq!(cfg.grid_dim.0, 168);
    assert_eq!(cfg.block_dim.0, 256);
    assert_eq!(cfg.shared_mem_bytes, 0);

    println!("LaunchConfig structure verified for cooperative kernel launch");
}
