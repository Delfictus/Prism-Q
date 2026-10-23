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
    let lib_dir = env::var("SDST_LIB_DIR").unwrap_or_else(|_| format!("{}/lib", manifest_dir));
    println!("cargo:rerun-if-env-changed=SDST_LIB_DIR");

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
