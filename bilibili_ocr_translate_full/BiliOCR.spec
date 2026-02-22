# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for BiliOCR - Mac .app bundle
# Build: pyinstaller BiliOCR.spec
#
# NOTE: Bundle will be large (1-3GB+) due to torch, faster-whisper, TTS.
# First build can take 10-20 minutes.

block_cipher = None

# Data files to bundle
datas = [
    ('img', 'img'),
    ('.env.example', '.'),
]

# Modules that must be explicitly collected (PyInstaller may miss them)
hiddenimports = [
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'numpy',
    'PIL',
    'PIL.Image',
    'requests',
    'mss',
    'mss.mss',
    'sounddevice',
    'dotenv',
    'jieba',
    'jieba.analyse',
    'jieba.posseg',
    'pypinyin',
    'pypinyin.style',
    'cepy_dict',
    'streaming_reconciler',
    'starred_db',
    'ocr_providers',
    'ocr_correct',
    'learn_keywords',
    'tts_engine',
    'audio_pipeline',
    'audio_capture',
    'audio_transcriber',
    'audio_config',
    'capture_mac',
    'vision_ocr',
    # PyObjC / macOS frameworks
    'Vision',
    'Quartz',
    'Foundation',
    'AppKit',
    'objc',
]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',  # Not used
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # Put binaries in COLLECT for .app structure
    name='BiliOCR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # No terminal window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='BiliOCR',
)

# Build as .app bundle (Mac only)
app = BUNDLE(
    coll,
    name='BiliOCR.app',
    icon=None,  # Add 'img/icon.icns' if you have one
    bundle_identifier='com.biliocr.app',
    info_plist={
        'CFBundleName': 'BiliOCR',
        'CFBundleDisplayName': 'BiliOCR',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'NSMicrophoneUsageDescription': 'BiliOCR needs microphone access for audio transcription.',
        'NSScreenCaptureUsageDescription': 'BiliOCR needs screen capture to read subtitles.',
    },
)
