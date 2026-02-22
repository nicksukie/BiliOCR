#!/bin/bash
# Build BiliOCR .app and DMG for distribution
# Run from project root: ./build_dmg.sh
#
# Prerequisites:
#   pip install pyinstaller
#   brew install create-dmg   (optional, for nicer DMG; hdiutil fallback works without it)

set -e
cd "$(dirname "$0")"
APP_NAME="BiliOCR"
DMG_NAME="BiliOCR"

echo "==> Building $APP_NAME.app with PyInstaller..."
echo "    (This may take 10-20 min on first run due to ML dependencies)"
pyinstaller BiliOCR.spec

if [ ! -d "dist/$APP_NAME.app" ]; then
    echo "ERROR: dist/$APP_NAME.app not found. Build failed."
    exit 1
fi

echo ""
echo "==> Creating DMG..."

# Prepare DMG folder
DMG_DIR="dist/dmg"
rm -rf "$DMG_DIR"
mkdir -p "$DMG_DIR"
cp -R "dist/$APP_NAME.app" "$DMG_DIR/"

# Remove old DMG if exists
rm -f "dist/$DMG_NAME.dmg"

# Use create-dmg if available (nicer: Applications link, window layout)
if command -v create-dmg &>/dev/null; then
    create-dmg \
        --volname "$APP_NAME" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "$APP_NAME.app" 175 120 \
        --hide-extension "$APP_NAME.app" \
        --app-drop-link 425 120 \
        "dist/$DMG_NAME.dmg" \
        "$DMG_DIR/"
    echo ""
    echo "==> Done! DMG created: dist/$DMG_NAME.dmg"
else
    # Fallback: use hdiutil (built into macOS)
    echo "    (create-dmg not found; using hdiutil for basic DMG)"
    echo "    Install 'brew install create-dmg' for a nicer installer with Applications link."
    hdiutil create -volname "$APP_NAME" -srcfolder "$DMG_DIR" -ov -format UDZO "dist/$DMG_NAME.dmg"
    echo ""
    echo "==> Done! DMG created: dist/$DMG_NAME.dmg"
fi

echo ""
echo "Users can:"
echo "  1. Download the .dmg file"
echo "  2. Double-click to open it"
echo "  3. Drag BiliOCR.app to Applications"
echo ""
echo "Note: First run may need to right-click > Open (Gatekeeper on unsigned apps)"
echo "      For audio mode, users need BlackHole: brew install blackhole-2ch"
