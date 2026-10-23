//! Loads the delfictus GTK theme + bundled fonts into the default display.

use gtk::gdk;

/// Install the application stylesheet (delfictus light/cyan tokens). Called once
/// at application startup.
pub fn install() {
    let provider = gtk::CssProvider::new();
    provider.load_from_string(include_str!("../assets/theme.css"));
    if let Some(display) = gdk::Display::default() {
        gtk::style_context_add_provider_for_display(
            &display,
            &provider,
            gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    }
    // Force a light color scheme so the delfictus palette renders as designed.
    adw::StyleManager::default().set_color_scheme(adw::ColorScheme::ForceLight);
}
