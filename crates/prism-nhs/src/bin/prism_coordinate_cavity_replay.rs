use prism_nhs::coordinate_cavity::saved_ghost;
use clap::Parser;
fn main() -> anyhow::Result<()> {
    if std::env::args_os().any(|arg| arg == "--saved-ghost-components") {
        let filtered = std::env::args_os().filter(|arg| arg != "--saved-ghost-components");
        return saved_ghost::run(saved_ghost::Args::parse_from(filtered));
    }
    if std::env::args_os().any(|arg| arg == "--interface-component") {
        let filtered = std::env::args_os().filter(|arg| arg != "--interface-component");
        return prism_nhs::coordinate_cavity::recorded_interface::run(
            prism_nhs::coordinate_cavity::recorded_interface::Args::parse_from(filtered),
        );
    }
    prism_nhs::coordinate_cavity::recorded_roi::run(
        prism_nhs::coordinate_cavity::recorded_roi::Args::parse(),
    )
}
