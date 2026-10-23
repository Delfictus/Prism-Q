#!/usr/bin/env bash
#
# Install the PrismOS Beta2.0 desktop app + its assets into an appliance image
# (or a developer system). Idempotent. Honors DESTDIR for packaging into a .deb
# staging root. Does NOT reference /tmp.
#
# Layout installed:
#   $PREFIX/bin/prismos-ui                              (the binary)
#   $PREFIX/share/applications/...desktop               (launcher)
#   $PREFIX/share/icons/hicolor/scalable/apps/...svg    (icon)
#   $PREFIX/share/prismos/ui/theme.css                  (GTK theme)
#   $PREFIX/share/prismos/ui/branding/prismos-splash.png
#   $PREFIX/share/fonts/prismos/*.ttf,*.otf             (bundled fonts)
#
# Usage:
#   ./install-prismos-ui.sh                 # installs to /usr (needs root)
#   PREFIX=/opt/prismos ./install-prismos-ui.sh
#   DESTDIR=/staging PREFIX=/usr ./install-prismos-ui.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # crate root
PREFIX="${PREFIX:-/usr}"
DESTDIR="${DESTDIR:-}"
DEST="${DESTDIR}${PREFIX}"

# The build controller must bind the exact frozen binary or use this workspace's
# release output. Host-global Cargo caches and /tmp are never payload authority.
BIN="${PRISMOS_UI_BINARY:-${HERE}/../../target/release/prismos-ui}"
if [ ! -x "$BIN" ] || [ -L "$BIN" ]; then
  echo "error: prismos-ui binary not found. Build it first:" >&2
  echo "  cargo build -p prismos-ui --release" >&2
  exit 1
fi

install -Dm0755 "$BIN" "${DEST}/bin/prismos-ui"
install -Dm0644 "${HERE}/data/applications/com.delfictus.PrismOS.Ui.desktop" \
    "${DEST}/share/applications/com.delfictus.PrismOS.Ui.desktop"
python3 - "$PREFIX" "${DEST}/share/applications/com.delfictus.PrismOS.Ui.desktop" <<'PY'
import pathlib
import sys

prefix = pathlib.PurePosixPath(sys.argv[1])
desktop = pathlib.Path(sys.argv[2])
exe = str(prefix / "bin" / "prismos-ui")
text = desktop.read_text()
text = text.replace("Exec=prismos-ui\n", f"Exec={exe}\n")
text = text.replace("TryExec=prismos-ui\n", f"TryExec={exe}\n")
desktop.write_text(text)
PY
install -Dm0644 "${HERE}/data/icons/hicolor/scalable/apps/com.delfictus.PrismOS.Ui.svg" \
    "${DEST}/share/icons/hicolor/scalable/apps/com.delfictus.PrismOS.Ui.svg"
install -Dm0644 "${HERE}/assets/theme.css" "${DEST}/share/prismos/ui/theme.css"
install -Dm0644 "${HERE}/assets/branding/prismos-splash.png" \
    "${DEST}/share/prismos/ui/branding/prismos-splash.png"

for font in "${HERE}"/assets/fonts/*.ttf "${HERE}"/assets/fonts/*.otf; do
  [ -e "$font" ] || continue
  install -Dm0644 "$font" "${DEST}/share/fonts/prismos/$(basename "$font")"
done

# Refresh caches on a live system (skipped during DESTDIR staging).
if [ -z "$DESTDIR" ]; then
  command -v fc-cache              >/dev/null 2>&1 && fc-cache -f "${DEST}/share/fonts/prismos" || true
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "${DEST}/share/applications" || true
  command -v gtk-update-icon-cache  >/dev/null 2>&1 && gtk-update-icon-cache -f "${DEST}/share/icons/hicolor" || true
fi

echo "PrismOS UI installed under ${DEST}"
