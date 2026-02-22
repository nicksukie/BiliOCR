# Packaging BiliOCR as a DMG

This guide explains how to build a distributable `.dmg` file so users can download and install BiliOCR without setting up Python.

## Quick build

```bash
pip install pyinstaller
./build_dmg.sh
```

Output: `dist/BiliOCR.dmg`

## Prerequisites

- **Python 3** with all dependencies from `requirements.txt`
- **PyInstaller**: `pip install pyinstaller`
- **create-dmg** (optional): `brew install create-dmg` — produces a nicer DMG with Applications shortcut. Without it, the script falls back to `hdiutil` (built into macOS).

## Build time & size

- **First build**: 10–20 minutes (PyInstaller analyzes and bundles torch, faster-whisper, TTS, etc.)
- **DMG size**: ~1–3 GB (ML models included)

## What users get

1. Download `BiliOCR.dmg`
2. Double-click to open
3. Drag BiliOCR.app to Applications
4. Run the app

**Note**: Unsigned apps may trigger Gatekeeper. Users can right-click → Open on first launch, or use System Settings → Privacy & Security to allow it.

## User requirements

- **macOS** (Intel or Apple Silicon)
- **API keys**: Users must add translation keys (DeepL, Google, etc.) — copy `.env.example` to `.env` in the app bundle or configure in-app
- **Audio mode**: For audio transcription, users need [BlackHole](https://github.com/ExistentialAudio/BlackHole): `brew install blackhole-2ch`

## Code signing (optional)

To avoid Gatekeeper warnings, sign and notarize the app (requires Apple Developer account, $99/year):

```bash
codesign --deep --force --sign "Developer ID Application: Your Name" dist/BiliOCR.app
# Then notarize with xcrun notarytool
```

## Troubleshooting

- **Missing module errors**: Add the module to `hiddenimports` in `BiliOCR.spec`
- **App crashes on launch**: Run from Terminal to see errors: `dist/BiliOCR.app/Contents/MacOS/BiliOCR`
- **Large bundle**: Consider excluding optional backends (e.g. TTS, funasr) if not needed for your use case
