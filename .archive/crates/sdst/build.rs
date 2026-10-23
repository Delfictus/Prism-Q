//! Build script for the sdst Rust crate.
//!
//! Links against the pre-built libsdst.so (crates/sdst/lib/) and embeds an
//! relocatable RUNPATH so the nhs_rt_full binary finds the appliance-owned
//! library beside its installation tree without requiring LD_LIBRARY_PATH.

use std::env;

fn main() {
    let manifest_dir =
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by cargo");
    // Link a staged native build without replacing a pinned/hardlinked library.
    // Runtime lookup remains relative to the executable's installation layout.
    // P0.5: the prebuilt libsdst.so is a pinned binary build input that lives
    // outside the engine tree (PrismOS/engine_build_inputs/sdst/lib). Release
    // builds require SDST_LIB_DIR and PRISM_SDST_LIB_SHA256 and fail closed on a
    // missing library or a hash mismatch. Debug builds may omit the hash pin.
    let _ = &manifest_dir;
    println!("cargo:rerun-if-env-changed=SDST_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PRISM_SDST_LIB_SHA256");
    let release = env::var("PROFILE").map(|p| p == "release").unwrap_or(false);
    let lib_dir = env::var("SDST_LIB_DIR")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| {
            panic!(
                "SDST_LIB_DIR must name the directory holding the pinned libsdst.so \
                 (PrismOS/engine_build_inputs/sdst/lib); the in-crate lib/ copy was removed"
            )
        });
    let lib_path = std::path::Path::new(&lib_dir).join("libsdst.so");
    let bytes = std::fs::read(&lib_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", lib_path.display(), e));
    let actual = sha256_hex(&bytes);
    match env::var("PRISM_SDST_LIB_SHA256").ok().filter(|v| !v.trim().is_empty()) {
        Some(expected) if expected.trim().eq_ignore_ascii_case(&actual) => {}
        Some(expected) => panic!(
            "libsdst.so sha256 mismatch: {} is {}, PRISM_SDST_LIB_SHA256={}",
            lib_path.display(),
            actual,
            expected
        ),
        None if release => panic!(
            "release build requires PRISM_SDST_LIB_SHA256 (pinned libsdst.so; {} is {})",
            lib_path.display(),
            actual
        ),
        None => println!("cargo:warning=libsdst.so not hash-pinned (debug build): {actual}"),
    }
    println!("cargo:rustc-env=PRISM_SDST_LIB_SHA256={actual}");

    // Tell the linker where to find libsdst.so at compile time
    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=dylib=sdst");

    // cudart is always available when building with the gpu feature on this machine
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Keep the installed ELF independent of the developer checkout. The appliance
    // layout is /opt/prismos/bin/<binary> plus /opt/prismos/lib/libsdst.so.
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");

    // Re-link if the library changes
    println!("cargo:rerun-if-changed={}/libsdst.so", lib_dir);
    println!("cargo:rerun-if-changed=include/sdst_api.h");
}

/// Minimal dependency-free SHA-256 (FIPS 180-4) for the build-input pin.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}
