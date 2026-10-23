//! PrismOS Beta2.0 desktop application entry point.
//!
//! With the default `gui` feature this launches the GTK4 desktop shell. Built
//! with `--no-default-features` it runs a headless self-check of the pure
//! command-adapter + evidence-model core (useful for CI on machines without the
//! GTK system libraries, and for verifying backend binary resolution).

fn main() {
    #[cfg(feature = "gui")]
    {
        std::process::exit(prismos_ui::app::run());
    }

    #[cfg(not(feature = "gui"))]
    {
        headless_selfcheck();
    }
}

#[cfg(not(feature = "gui"))]
fn headless_selfcheck() {
    use prismos_ui::adapters::{BinaryResolver, PrismBin, PrismCommand};
    use std::path::Path;

    let resolver = BinaryResolver::with_defaults();
    println!("PrismOS UI — pure-core self-check (built without GTK)\n");
    println!("Backend binary resolution:");
    for bin in PrismBin::ALL {
        println!(
            "  {:<26} -> {}",
            bin.file_name(),
            resolver.resolve(bin).display()
        );
    }
    println!("\nSample typed argv (no shell):");
    let sample = PrismCommand::qnanl_header(Path::new("/runs/demo/x.prismqnanl"));
    println!("  {}", sample.display(&resolver));
}
