"""Mac-native OCR translator with PyQt5 overlay."""
import hashlib
import os
import sys
import time

# Load .env for API keys (DEEPL_AUTH_KEY, etc.)
try:
    from dotenv import load_dotenv
    if getattr(sys, "frozen", False):
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(app_dir, ".env"))
    if not os.path.exists(os.path.join(app_dir, ".env")):
        load_dotenv()
except ImportError:
    pass
import json
import re
import threading
from collections import Counter
from datetime import datetime
import queue
import uuid

import requests
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QDialog, QDialogButtonBox, QLineEdit, QFormLayout, QCheckBox, QListWidget, QListWidgetItem, QMenu, QWidgetAction, QRadioButton, QButtonGroup, QToolTip, QComboBox, QPlainTextEdit, QTextEdit, QSpinBox, QFileDialog, QStackedWidget, QFrame, QTabWidget, QMainWindow, QDoubleSpinBox, QGridLayout, QGraphicsOpacityEffect
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint, QEventLoop, pyqtSignal, QMetaObject, QEvent, QSize, QObject, QSettings, pyqtSlot
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QClipboard, QCursor, QFontMetrics, QTextDocument, QIcon, QTextCursor, QPixmap

from capture_mac import create_capture, DynamicRegionCapture
from vision_ocr import VisionOCR
from audio_pipeline import AudioPipeline


class _DebugOutputEmitter(QObject):
    """Thread-safe emitter for captured stdout/stderr. Used by TeeStream."""
    text_written = pyqtSignal(str)


class _TeeStream:
    """Writes to both real stdout/stderr and emits to debug terminal."""

    def __init__(self, stream, emitter):
        self._stream = stream
        self._emitter = emitter

    def write(self, data):
        try:
            self._stream.write(data)
            self._stream.flush()
        except Exception:
            pass
        try:
            if data and self._emitter:
                self._emitter.text_written.emit(str(data))
        except Exception:
            pass

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


class DebugTerminal(QWidget):
    """Pseudo terminal showing captured stdout/stderr. Toggle with F12."""

    def __init__(self, screen_w, screen_h, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Debug Output")
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating, False)
        w, h = min(600, screen_w - 40), min(400, screen_h - 100)
        self.setGeometry(20, 60, w, h)
        self.setStyleSheet("""
            QWidget { background: #1e1e1e; }
            QPlainTextEdit { background: #0d0d0d; color: #d4d4d4; font-family: Menlo, Monaco, monospace; font-size: 11px; border: none; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self._text)
        self._max_lines = 5000
        self.hide()

    @pyqtSlot(str)
    def append(self, text):
        if not text:
            return
        cursor = self._text.textCursor()
        cursor.movePosition(cursor.End)
        self._text.setTextCursor(cursor)
        self._text.insertPlainText(text)
        doc = self._text.document()
        if doc.blockCount() > self._max_lines:
            cursor = self._text.textCursor()
            cursor.setPosition(0)
            cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, doc.blockCount() - self._max_lines)
            cursor.removeSelectedText()
        self._text.verticalScrollBar().setValue(self._text.verticalScrollBar().maximum())

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.setWindowTitle("Debug Output (F12 to hide)")
            self.show()
            self.raise_()
            self.activateWindow()


class _GlobalKeyFilter(QObject):
    """Event filter: Space toggles OCR pause, Enter resumes. Both work regardless of focus."""

    def __init__(self, translator_app, parent=None, debug_terminal=None):
        super().__init__(parent)
        self._app = translator_app
        self._debug_terminal = debug_terminal

    def _in_text_field(self):
        fw = QApplication.focusWidget()
        if not fw:
            return False
        if isinstance(fw, (QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox)):
            return True
        if isinstance(fw, QComboBox) and fw.isEditable():
            return True
        return False

    def _ocr_is_running(self):
        """True when OCR is actively reading. Box is white."""
        if not self._app:
            return False
        rs = getattr(self._app, "region_selector", None)
        if not rs:
            return not self._app._ocr_paused
        # Running = confirmed, not repositioning, not paused
        active = getattr(rs, "_active", False)
        needs_reconfirm = getattr(rs, "_needs_reconfirm", False)
        paused = getattr(self._app, "_ocr_paused", False)
        return active and not needs_reconfirm and not paused

    def _set_ocr_running(self, running):
        """Set unified OCR state: running=True means white box and OCR reading."""
        if not self._app:
            return
        rs = getattr(self._app, "region_selector", None)
        if running:
            self._app._ocr_paused = False
            self._app._ocr_obstructed = False
            self._app._reset_mixed_content_tracking()
            if rs:
                rs._needs_reconfirm = False
                rs._emit_region()
                rs._update_confirm_label()
        else:
            self._app._ocr_paused = True
        if rs and rs.isVisible():
            rs.update()

    def eventFilter(self, obj, event):
        if event.type() != QEvent.KeyPress:
            return False
        key = event.key()
        if key == Qt.Key_Escape:
            req = getattr(self._app, "_request_quit", None)
            if req:
                req()
            return True
        if key == Qt.Key_F12:
            if self._debug_terminal:
                self._debug_terminal.toggle()
            return True
        if self._in_text_field():
            return False
        if key == Qt.Key_Space:
            if self._app:
                if self._app.transcription_mode == "audio":
                    self._app._audio_paused = not self._app._audio_paused
                    if self._app._audio_paused:
                        self._app._add_status_message("Paused", duration_sec=5, is_good_news=False)
                    else:
                        # Show "Resumed" for 2 seconds
                        self._app._add_status_message("Resumed", duration_sec=2, is_good_news=True)
                    print(f"[Audio] {'paused' if self._app._audio_paused else 'resumed'} (Space)")
                    overlay = getattr(self._app, "overlay", None)
                    if overlay and hasattr(overlay, "update_play_pause_state"):
                        overlay.update_play_pause_state()
                else:
                    self._app._ocr_paused = not self._app._ocr_paused
                    if not self._app._ocr_paused:
                        self._app._ocr_obstructed = False
                        self._app._reset_mixed_content_tracking()
                    status = "paused" if self._app._ocr_paused else "resumed"
                    print(f"[OCR] {status} (Space to toggle)")
                    rs = getattr(self._app, "region_selector", None)
                    if rs and rs.isVisible():
                        rs.update()
            return True
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if self._app and self._app.transcription_mode == "audio":
                if self._app._audio_paused:
                    self._app._audio_paused = False
                    # Show "Resumed" for 2 seconds (this will replace the "Paused" message)
                    self._app._add_status_message("Resumed", duration_sec=2, is_good_news=True)
                    print("[Audio] resumed (Enter)")
                    overlay = getattr(self._app, "overlay", None)
                    if overlay and hasattr(overlay, "update_play_pause_state"):
                        overlay.update_play_pause_state()
                return True
            rs = getattr(self._app, "region_selector", None)
            if rs and not getattr(rs, "_active", False):
                return False  # Let RegionSelector handle initial confirm
            if not self._ocr_is_running():
                self._set_ocr_running(True)
                print("[OCR] resumed (Enter)")
            return True
        if key == Qt.Key_T:
            # TTS toggle shortcut
            if self._app and self._app.transcription_mode == "ocr":
                self._app.tts_enabled = not getattr(self._app, "tts_enabled", False)
                if not self._app.tts_enabled and hasattr(self._app, "tts_engine"):
                    self._app.tts_engine.stop()  # Cut off audio immediately, clear queue
                if hasattr(self._app, "_add_status_message"):
                    self._app._add_status_message("TTS on" if self._app.tts_enabled else "TTS off", duration_sec=2, is_good_news=self._app.tts_enabled)
                overlay = getattr(self._app, "overlay", None)
                if overlay and hasattr(overlay, "_update_speak_button_states"):
                    overlay._update_speak_button_states()
                print(f"[TTS] Toggled via keyboard: {'enabled' if self._app.tts_enabled else 'disabled'}")
                return True
        return False

def _mac_set_activation_policy_accessory():
    """Set app to accessory (no Dock icon) so windows can float above fullscreen apps."""
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        from ctypes import util
        libobjc = ctypes.CDLL(util.find_library("objc"))
        libobjc.objc_getClass.restype = ctypes.c_void_p
        libobjc.objc_getClass.argtypes = [ctypes.c_char_p]
        libobjc.sel_registerName.restype = ctypes.c_void_p
        libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
        libobjc.objc_msgSend.restype = ctypes.c_void_p
        libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        nsapp_class = libobjc.objc_getClass(b"NSApplication")
        sel_shared = libobjc.sel_registerName(b"sharedApplication")
        nsapp = libobjc.objc_msgSend(nsapp_class, sel_shared)
        if nsapp:
            # NSApplicationActivationPolicyAccessory = 1
            libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
            sel_policy = libobjc.sel_registerName(b"setActivationPolicy:")
            libobjc.objc_msgSend(ctypes.c_void_p(nsapp), sel_policy, 1)
    except Exception:
        pass


def _mac_set_fullscreen_overlay(widget):
    """Configure Qt window to appear above fullscreen apps on macOS."""
    if sys.platform != "darwin":
        return
    debug = "--debug" in sys.argv
    try:
        import ctypes
        from ctypes import util

        # On macOS Qt, winId() returns NSView* (the content view)
        wid = widget.winId()
        if not wid:
            if debug:
                print("[Fullscreen overlay] winId is 0")
            return

        view_ptr = ctypes.c_void_p(int(wid))

        # Use ctypes + objc_msgSend (no PyObjC required)
        libobjc = ctypes.CDLL(util.find_library("objc"))
        libobjc.sel_registerName.restype = ctypes.c_void_p
        libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
        libobjc.objc_msgSend.restype = ctypes.c_void_p
        libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        # [view window] -> NSWindow*
        sel_window = libobjc.sel_registerName(b"window")
        nswin_ptr = libobjc.objc_msgSend(view_ptr, sel_window)
        if not nswin_ptr:
            if debug:
                print("[Fullscreen overlay] view.window() returned NULL")
            return

        nswin = ctypes.c_void_p(nswin_ptr)

        # Add third arg for methods that take one argument
        libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]

        # Use CGWindowLevelForKey(.screenSaverWindow) = 1000 to float above fullscreen video
        # NSStatusWindowLevel (25) is too low for fullscreen apps
        kCGWindowLevelForKey = 1000
        NSWindowCollectionBehaviorCanJoinAllSpaces = 1 << 0   # 1
        NSWindowCollectionBehaviorFullScreenAuxiliary = 1 << 8  # 256
        NSWindowCollectionBehaviorStationary = 1 << 4           # 16
        behavior = (
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary
        )

        sel_level = libobjc.sel_registerName(b"setLevel:")
        libobjc.objc_msgSend(nswin, sel_level, kCGWindowLevelForKey)

        sel_behavior = libobjc.sel_registerName(b"setCollectionBehavior:")
        libobjc.objc_msgSend(nswin, sel_behavior, behavior)

        # Force window to front - critical for fullscreen overlay
        sel_order = libobjc.sel_registerName(b"orderFrontRegardless")
        libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        libobjc.objc_msgSend(nswin, sel_order)

        if debug:
            print("[Fullscreen overlay] OK (level=1000, orderFrontRegardless)")
    except Exception as e:
        if debug:
            import traceback
            print(f"[Fullscreen overlay] {e}")
            traceback.print_exc()


def _mac_raise_dialog_above_overlays(dialog):
    """Set dialog window level above our overlays (1000) so menus appear on top."""
    if sys.platform != "darwin":
        return
    try:
        wid = dialog.winId()
        if not wid:
            return
        import ctypes
        from ctypes import util
        view_ptr = ctypes.c_void_p(int(wid))
        libobjc = ctypes.CDLL(util.find_library("objc"))
        libobjc.sel_registerName.restype = ctypes.c_void_p
        libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
        libobjc.objc_msgSend.restype = ctypes.c_void_p
        libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        sel_window = libobjc.sel_registerName(b"window")
        nswin_ptr = libobjc.objc_msgSend(view_ptr, sel_window)
        if not nswin_ptr:
            return
        nswin = ctypes.c_void_p(nswin_ptr)
        libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        sel_level = libobjc.sel_registerName(b"setLevel:")
        libobjc.objc_msgSend(nswin, sel_level, 1001)  # Above overlays (1000)
        sel_order = libobjc.sel_registerName(b"orderFrontRegardless")
        libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        libobjc.objc_msgSend(nswin, sel_order)
    except Exception:
        pass


class _DialogRaiseFilter(QObject):
    """Event filter: when dialog is shown, raise it above overlays (macOS)."""
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Show:
            obj.removeEventFilter(self)
            # Defer so window is fully realized (winId valid)
            QTimer.singleShot(0, lambda: _mac_raise_dialog_above_overlays(obj))
        return False


# Language codes: internal -> (DeepL, Baidu, Youdao, Google). None = API uses auto or omit.
_LANG_MAP = {
    "auto": (None, "auto", "auto", None),
    "zh": ("ZH", "zh", "zh-CHS", "zh"),
    "ja": ("JA", "jp", "ja", "ja"),
    "en": ("EN", "en", "en", "en"),
    "ko": ("KO", "kor", "ko", "ko"),
    "es": ("ES", "spa", "es", "es"),
    "fr": ("FR", "fra", "fr", "fr"),
    "de": ("DE", "de", "de", "de"),
    "it": ("IT", "it", "it", "it"),
    "pt": ("PT", "pt", "pt", "pt"),
    "ru": ("RU", "ru", "ru", "ru"),
    "ar": ("AR", "ara", "ar", "ar"),
    "th": ("TH", "th", "th", "th"),
    "vi": ("VI", "vie", "vi", "vi"),
    "id": ("ID", "id", "id", "id"),
    "ms": ("MS", "ms", "ms", "ms"),
    "tr": ("TR", "tr", "tr", "tr"),
    "pl": ("PL", "pl", "pl", "pl"),
    "nl": ("NL", "nl", "nl", "nl"),
    "sv": ("SV", "swe", "sv", "sv"),
    "da": ("DA", "dan", "da", "da"),
    "fi": ("FI", "fin", "fi", "fi"),
    "no": ("NB", "nor", "no", "nb"),
    "el": ("EL", "el", "el", "el"),
    "he": ("HE", "heb", "he", "he"),
    "hi": ("HI", "hi", "hi", "hi"),
    "bn": ("BN", "ben", "bn", "bn"),
    "ta": ("TA", "tam", "ta", "ta"),
    "te": ("TE", "tel", "te", "te"),
    "uk": ("UK", "ukr", "uk", "uk"),
    "cs": ("CS", "cs", "cs", "cs"),
    "ro": ("RO", "rom", "ro", "ro"),
    "hu": ("HU", "hu", "hu", "hu"),
    "sk": ("SK", "sk", "sk", "sk"),
    "bg": ("BG", "bul", "bg", "bg"),
    "hr": ("HR", "hrv", "hr", "hr"),
    "sr": ("SR", "srp", "sr", "sr"),
    "sl": ("SL", "slo", "sl", "sl"),
    "et": ("ET", "est", "et", "et"),
    "lv": ("LV", "lav", "lv", "lv"),
    "lt": ("LT", "lit", "lt", "lt"),
    "fa": ("FA", "per", "fa", "fa"),
    "sw": ("SW", "swa", "sw", "sw"),
    "af": ("AF", "afr", "af", "af"),
    "ca": ("CA", "cat", "ca", "ca"),
    "gl": ("GL", "glg", "gl", "gl"),
    "eu": ("EU", "baq", "eu", "eu"),
}

# Display order for dropdowns: (label, internal_code)
_LANG_OPTIONS = [
    # ("Auto (detect)", "auto"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Arabic", "ar"),
    ("Thai", "th"),
    ("Vietnamese", "vi"),
    ("Indonesian", "id"),
    ("Malay", "ms"),
    ("Turkish", "tr"),
    ("Polish", "pl"),
    ("Dutch", "nl"),
    ("Swedish", "sv"),
    ("Danish", "da"),
    ("Finnish", "fi"),
    ("Norwegian", "no"),
    ("Greek", "el"),
    ("Hebrew", "he"),
    ("Hindi", "hi"),
    ("Bengali", "bn"),
    ("Tamil", "ta"),
    ("Telugu", "te"),
    ("Ukrainian", "uk"),
    ("Czech", "cs"),
    ("Romanian", "ro"),
    ("Hungarian", "hu"),
    ("Slovak", "sk"),
    ("Bulgarian", "bg"),
    ("Croatian", "hr"),
    ("Serbian", "sr"),
    ("Slovenian", "sl"),
    ("Estonian", "et"),
    ("Latvian", "lv"),
    ("Lithuanian", "lt"),
    ("Persian", "fa"),
    ("Swahili", "sw"),
    ("Afrikaans", "af"),
    ("Catalan", "ca"),
    ("Galician", "gl"),
    ("Basque", "eu"),
]

_LANG_OPTIONS_TARGET = [(lbl, code) for lbl, code in _LANG_OPTIONS if code != "auto"]


def _app_dir():
    """Directory for .env (next to executable when frozen, else script dir)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _has_any_api_key():
    return any(os.environ.get(k) for k in (
        "DEEPL_AUTH_KEY", "GOOGLE_TRANSLATE_API_KEY",
        "OPENAI_API_KEY", "ELEVENLABS_API_KEY", "SILICONFLOW_COM_API_KEY", "SILICONFLOW_CN_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY",
        "GROQ_API_KEY", "TOGETHER_API_KEY", "HF_API_KEY", "YANDEX_API_KEY", "LIBRETRANSLATE_API_KEY",
        "CAIYUN_TOKEN", "NIUTRANS_APIKEY",
    ))


def show_api_keys_dialog(parent=None):
    """Show dialog to enter API keys. Saves to .env in app dir."""
    dlg = QDialog(parent)
    dlg.setWindowTitle("API Keys")
    dlg.setMinimumWidth(500)
    layout = QVBoxLayout(dlg)

    info = QLabel("At least one service needed. Keys are saved locally.")
    info.setWordWrap(True)
    layout.addWidget(info)

    form = QFormLayout()
    pw_edits = []
    
    # Helper function to delete a key from .env file
    def delete_key_from_env(env_key_name):
        """Remove a key from .env file."""
        env_path = os.path.join(_app_dir(), ".env")
        if not os.path.exists(env_path):
            return
        try:
            with open(env_path, "r") as f:
                lines = f.readlines()
            with open(env_path, "w") as f:
                for line in lines:
                    if not line.strip().startswith(env_key_name + "="):
                        f.write(line)
        except OSError:
            pass
    
    # Helper function to create a row with delete button
    def create_key_row(label_text, env_key_name, placeholder_text=""):
        """Create a form row with label, text input, and delete button."""
        key_edit = QLineEdit()
        key_edit.setPlaceholderText(placeholder_text)
        key_edit.setText(os.environ.get(env_key_name, ""))
        key_edit.setEchoMode(QLineEdit.Password)
        pw_edits.append(key_edit)
        
        # Delete button - just a small "x" character
        delete_btn = QPushButton("×")
        delete_btn.setFixedSize(20, 20)
        delete_btn.setToolTip("Delete this API key")
        delete_btn.setCursor(Qt.PointingHandCursor)
        delete_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #999;
                border: none;
                font-size: 16px;
                font-weight: normal;
                padding: 0;
            }
            QPushButton:hover {
                color: #333;
            }
        """)
        
        def on_delete():
            key_edit.clear()
            delete_key_from_env(env_key_name)
            # Also remove from environment
            if env_key_name in os.environ:
                del os.environ[env_key_name]
        
        delete_btn.clicked.connect(on_delete)
        
        # Horizontal layout for edit + button
        row_layout = QHBoxLayout()
        key_edit.setMinimumWidth(250)  # Make the text field wider
        row_layout.addWidget(key_edit, 1)
        row_layout.addWidget(delete_btn, 0)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        
        row_widget = QWidget()
        row_widget.setLayout(row_layout)
        form.addRow(label_text, row_widget)
        
        return key_edit

    siliconflow_com_key = create_key_row("SiliconFlow.com:", "SILICONFLOW_COM_API_KEY", "api.siliconflow.com")

    siliconflow_cn_key = create_key_row("SiliconFlow.cn:", "SILICONFLOW_CN_API_KEY", "api.siliconflow.cn")

    openai_key = create_key_row("OpenAI:", "OPENAI_API_KEY", "OpenAI / GPT / Whisper / TTS")

    elevenlabs_key = create_key_row("ElevenLabs:", "ELEVENLABS_API_KEY", "ElevenLabs TTS (elevenlabs.io)")

    deepseek_key = create_key_row("DeepSeek:", "DEEPSEEK_API_KEY", "DeepSeek")

    anthropic_key = create_key_row("Anthropic:", "ANTHROPIC_API_KEY", "Anthropic Claude")

    groq_key = create_key_row("Groq:", "GROQ_API_KEY", "api.groq.com")

    together_key = create_key_row("Together:", "TOGETHER_API_KEY", "api.together.xyz")

    hf_key = create_key_row("HuggingFace API:", "HF_API_KEY", "HuggingFace Inference API")

    deepl = create_key_row("DeepL:", "DEEPL_AUTH_KEY", "DeepL API key (500K chars/month free)")

    google_key = create_key_row("Google:", "GOOGLE_TRANSLATE_API_KEY", "Google Cloud Translation API key (500K chars/month free)")

    yandex_key = create_key_row("Yandex:", "YANDEX_API_KEY", "Yandex Translate API key")

    libretranslate_key = create_key_row("LibreTranslate Key:", "LIBRETRANSLATE_API_KEY", "LibreTranslate API key")
    
    # LibreTranslate URL field - right after the API key
    libretranslate_url_edit = QLineEdit()
    libretranslate_url_edit.setPlaceholderText("https://libretranslate.com (or your self-hosted URL)")
    libretranslate_url_edit.setText(os.environ.get("LIBRETRANSLATE_URL", "https://libretranslate.com"))
    libretranslate_url_edit.setEchoMode(QLineEdit.Normal)
    libretranslate_url_edit.setMinimumWidth(250)  # Make the text field wider
    
    # Delete button for LibreTranslate URL
    delete_url_btn = QPushButton("×")
    delete_url_btn.setFixedSize(20, 20)
    delete_url_btn.setToolTip("Delete this URL")
    delete_url_btn.setCursor(Qt.PointingHandCursor)
    delete_url_btn.setStyleSheet("""
        QPushButton {
            background: transparent;
            color: #999;
            border: none;
            font-size: 16px;
            font-weight: normal;
            padding: 0;
        }
        QPushButton:hover {
            color: #333;
        }
    """)
    
    def on_delete_url():
        libretranslate_url_edit.clear()
        delete_key_from_env("LIBRETRANSLATE_URL")
        if "LIBRETRANSLATE_URL" in os.environ:
            del os.environ["LIBRETRANSLATE_URL"]
    
    delete_url_btn.clicked.connect(on_delete_url)
    
    url_row_layout = QHBoxLayout()
    url_row_layout.addWidget(libretranslate_url_edit, 1)
    url_row_layout.addWidget(delete_url_btn, 0)
    url_row_layout.setContentsMargins(0, 0, 0, 0)
    url_row_layout.setSpacing(6)
    
    url_row_widget = QWidget()
    url_row_widget.setLayout(url_row_layout)
    form.addRow("LibreTranslate URL:", url_row_widget)
    
    caiyun_key = create_key_row("Caiyun (彩云):", "CAIYUN_TOKEN", "Caiyun 彩云小译 token")

    niutrans_key = create_key_row("Niutrans (小牛):", "NIUTRANS_APIKEY", "Niutrans 小牛翻译 API key")
    


    show_cb = QCheckBox("Show keys")
    def toggle_echo():
        mode = QLineEdit.Normal if show_cb.isChecked() else QLineEdit.Password
        for e in pw_edits:
            e.setEchoMode(mode)
    show_cb.stateChanged.connect(lambda _: toggle_echo())
    layout.addLayout(form)
    layout.addWidget(show_cb)

    btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
    btns.button(QDialogButtonBox.Save).setText("Save")
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)

    layout.addWidget(btns)

    dlg.installEventFilter(_DialogRaiseFilter(dlg))
    if dlg.exec_() != QDialog.Accepted:
        return

    env_path = os.path.join(_app_dir(), ".env")
    lines = []
    env_keys = (
        "DEEPL_AUTH_KEY", "GOOGLE_TRANSLATE_API_KEY",
        "SILICONFLOW_COM_API_KEY", "SILICONFLOW_CN_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY",
        "GROQ_API_KEY", "TOGETHER_API_KEY", "HF_API_KEY", "YANDEX_API_KEY", "LIBRETRANSLATE_API_KEY", "LIBRETRANSLATE_URL",
        "CAIYUN_TOKEN", "NIUTRANS_APIKEY",
    )
    try:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if not any(line.strip().startswith(k + "=") for k in env_keys):
                        lines.append(line.rstrip())
    except OSError:
        pass

    def add(k, v):
        if v := v.strip():
            os.environ[k] = v
            lines.append(f"{k}={v}")

    add("DEEPL_AUTH_KEY", deepl.text())
    add("GOOGLE_TRANSLATE_API_KEY", google_key.text())
    add("SILICONFLOW_COM_API_KEY", siliconflow_com_key.text())
    add("SILICONFLOW_CN_API_KEY", siliconflow_cn_key.text())
    add("OPENAI_API_KEY", openai_key.text())
    add("ELEVENLABS_API_KEY", elevenlabs_key.text())
    add("DEEPSEEK_API_KEY", deepseek_key.text())
    add("ANTHROPIC_API_KEY", anthropic_key.text())
    add("GROQ_API_KEY", groq_key.text())
    add("TOGETHER_API_KEY", together_key.text())
    add("HF_API_KEY", hf_key.text())
    add("YANDEX_API_KEY", yandex_key.text())
    add("LIBRETRANSLATE_API_KEY", libretranslate_key.text())
    if libretranslate_url_edit.text().strip():
        add("LIBRETRANSLATE_URL", libretranslate_url_edit.text())
    add("CAIYUN_TOKEN", caiyun_key.text())
    add("NIUTRANS_APIKEY", niutrans_key.text())

    try:
        with open(env_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    except OSError as ex:
        if "--debug" in sys.argv:
            print(f"Could not save .env: {ex}")


# Bilibili brand colors
_BILIBILI_BLUE = "#00A1D6"
_BILIBILI_PINK = "#FB7299"
_BILIBILI_WHITE = "#FFFFFF"
_BILIBILI_PURPLE = "#946CE6"  # Bilibili purple for logo/titles

_SETTINGS_ORG = "BiliOCR"
_SETTINGS_APP = "BiliOCR"


def get_app_settings():
    """Load persisted settings. Returns dict with detect_mixed_content, max_words_enabled, allow_overlap, etc."""
    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    return {
        "detect_mixed_content": s.value("detect_mixed_content", False, type=bool),
        "max_words_enabled": s.value("max_words_enabled", False, type=bool),
        "max_words_for_translation": s.value("max_words_for_translation", 50, type=int),
        "allow_overlap": s.value("allow_overlap", False, type=bool),
        "auto_detect_text_region": s.value("auto_detect_text_region", False, type=bool),
        "llm_context_count": s.value("llm_context_count", 3, type=int),
        "session_output_enabled": s.value("session_output_enabled", False, type=bool),
        "session_output_path": s.value("session_output_path", "", type=str) or "",
        # Audio reconciler: X sec period, Y checks, min words
        "audio_reconciler_period_sec": s.value("audio_reconciler_period_sec", 2.0, type=float),
        "audio_reconciler_num_checks": s.value("audio_reconciler_num_checks", 4, type=int),
        "audio_reconciler_min_words": s.value("audio_reconciler_min_words", 7, type=int),
        "audio_silence_duration": s.value("audio_silence_duration", 1.0, type=float),
        "audio_max_phrase_duration": s.value("audio_max_phrase_duration", 5.0, type=float),
        # OCR reconciler settings
        "ocr_mt_reconciler_stability": s.value("ocr_mt_reconciler_stability", 0.2, type=float),
        "ocr_llm_reconciler_stability": s.value("ocr_llm_reconciler_stability", 0.12, type=float),
        "ocr_llm_reconciler_max_buffer": s.value("ocr_llm_reconciler_max_buffer", 0.6, type=float),
        "ocr_min_words_before_translate": s.value("ocr_min_words_before_translate", 0, type=int),
        "ocr_similarity_substring_chars": s.value("ocr_similarity_substring_chars", 15, type=int),

    }


def save_app_settings(settings):
    """Persist settings."""
    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    for k, v in settings.items():
        s.setValue(k, v)
    s.sync()


def _apply_settings_to_translator(translator, settings):
    """Apply settings to running translator (real-time)."""
    if not translator:
        return
    translator.detect_mixed_content = settings.get("detect_mixed_content", False)
    translator.max_words_enabled = settings.get("max_words_enabled", False)
    translator.max_words_for_translation = max(1, settings.get("max_words_for_translation", 50))
    translator.allow_overlap = settings.get("allow_overlap", False)
    translator.auto_detect_text_region = settings.get("auto_detect_text_region", False)
    translator.session_output_enabled = settings.get("session_output_enabled", False)
    translator.session_output_path = (settings.get("session_output_path", "") or "").strip()
    translator._session_output_path = None
    if not translator.auto_detect_text_region:
        translator._text_region = None
        translator._text_region_readings = 0
        translator._text_region_min_y = []
        translator._text_region_max_y = []
    # OCR reconciler: update settings dynamically
    if hasattr(translator, "reconciler") and translator.reconciler:
        translator.reconciler.stability_threshold = settings.get("ocr_mt_reconciler_stability", 0.2)
    if hasattr(translator, "llm_reconciler") and translator.llm_reconciler:
        translator.llm_reconciler.stability_threshold = settings.get("ocr_llm_reconciler_stability", 0.12)
        translator.llm_reconciler.max_buffer_time = settings.get("ocr_llm_reconciler_max_buffer", 0.6)
    # Store min words setting for use in OCR processing
    translator.ocr_min_words_before_translate = settings.get("ocr_min_words_before_translate", 0)
    translator.ocr_similarity_substring_chars = max(0, settings.get("ocr_similarity_substring_chars", 15))
    translator.llm_context_count = max(0, settings.get("llm_context_count", 3))
    # Audio mode: mutable settings for real-time tuning
    if hasattr(translator, "audio_buffer_settings"):
        translator.audio_buffer_settings.update({
            "reconciler_period_sec": settings.get("audio_reconciler_period_sec", 2.0),
            "reconciler_num_checks": settings.get("audio_reconciler_num_checks", 4),
            "reconciler_min_words": settings.get("audio_reconciler_min_words", 7),
            "silence_duration": settings.get("audio_silence_duration", 1.0),
            "max_phrase_duration": settings.get("audio_max_phrase_duration", 5.0),
        })


def show_settings_dialog(parent=None, translator=None, transcription_mode="ocr"):
    """Show settings dialog. Apply = real-time update; OK = save and close; Cancel = discard."""
    settings = get_app_settings()
    dlg = QDialog(parent)
    dlg.setWindowTitle("Settings")
    dlg.setMinimumWidth(420)
    dlg.setStyleSheet(f"""
        QDialog {{ background: rgba(255, 255, 255, 0.92); }}
        QLabel {{ color: #333; font-size: 13px; }}
        QCheckBox {{ color: #333; font-size: 13px; }}
        QPushButton {{ background: {_BILIBILI_BLUE}; color: white; border: none; border-radius: 8px; padding: 10px 18px; }}
        QPushButton:hover {{ background: #0090bc; }}
        QDialogButtonBox QPushButton[text="OK"] {{ background: {_BILIBILI_BLUE}; }}
        QDialogButtonBox QPushButton[text="Cancel"] {{ background: #aaa; color: #333; }}
        QTabWidget::pane {{ border: 1px solid #ddd; border-radius: 6px; }}
        QDoubleSpinBox, QSpinBox {{ min-width: 80px; }}
    """)
    layout = QVBoxLayout(dlg)
    layout.setSpacing(14)
    layout.setContentsMargins(24, 24, 24, 24)
    title = QLabel("Settings")
    title.setStyleSheet(f"color: {_BILIBILI_PURPLE}; font-size: 18px; font-weight: bold;")
    layout.addWidget(title)
    tabs = QTabWidget()
    
    # --- General tab (shared) ---
    general = QWidget()
    general_layout = QVBoxLayout(general)
    session_output_cb = QCheckBox("Save session to JSON (OCR, translations, model used)")
    session_output_cb.setChecked(settings.get("session_output_enabled", False))
    session_output_cb.setToolTip("When enabled: writes session data to a JSON file every ~10 translations.")
    general_layout.addWidget(session_output_cb)
    session_path_row = QHBoxLayout()
    session_path_edit = QLineEdit()
    session_path_edit.setPlaceholderText("Default: app directory")
    session_path_edit.setText(settings.get("session_output_path", "") or "")
    session_path_row.addWidget(QLabel("Output path:"))
    session_path_row.addWidget(session_path_edit)
    browse_btn = QPushButton("Browse...")
    def pick_session_dir():
        current = session_path_edit.text().strip()
        start = current if current and os.path.isdir(current) else os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(dlg, "Session output directory", start)
        if path:
            session_path_edit.setText(path)
    browse_btn.clicked.connect(pick_session_dir)
    session_path_row.addWidget(browse_btn)
    general_layout.addLayout(session_path_row)
    
    # LLM context count
    general_layout.addWidget(QLabel(""))  # Spacer
    llm_context_label = QLabel("LLM context count (previous translations):")
    llm_context_tooltip = "Number of previous translations to include in LLM prompt for context.\n\nHelps with topic/name consistency and terminology. Higher = more context but longer prompts.\n0 = no context (faster, less consistent)"
    llm_context_label.setToolTip(llm_context_tooltip)
    llm_context_spin = QSpinBox()
    llm_context_spin.setRange(0, 10)
    llm_context_spin.setSuffix(" translations")
    llm_context_spin.setValue(settings.get("llm_context_count", 3))
    llm_context_spin.setToolTip(llm_context_tooltip)
    llm_context_row = QHBoxLayout()
    llm_context_row.addWidget(llm_context_label)
    llm_context_row.addWidget(llm_context_spin)
    llm_context_row.addStretch()
    general_layout.addLayout(llm_context_row)
    
    general_layout.addStretch()
    tabs.addTab(general, "General")
    
    # --- OCR tab ---
    ocr_tab = QWidget()
    ocr_layout = QVBoxLayout(ocr_tab)
    detect_mixed_cb = QCheckBox("Detect mixed content (warn if OCR area captures static content besides subtitles)")
    detect_mixed_cb.setChecked(settings.get("detect_mixed_content", False))
    ocr_layout.addWidget(detect_mixed_cb)
    max_words_cb = QCheckBox("Disable translation if text exceeds")
    max_words_cb.setChecked(settings.get("max_words_enabled", False))
    max_words_spin = QSpinBox()
    max_words_spin.setRange(1, 500)
    max_words_spin.setSuffix(" words")
    max_words_spin.setValue(settings.get("max_words_for_translation", 50))
    max_words_spin.setToolTip("Typical subtitles are 5–50 words; large values suggest UI/metadata capture.")
    max_words_spin.setEnabled(max_words_cb.isChecked())
    max_words_cb.toggled.connect(max_words_spin.setEnabled)
    max_words_row = QHBoxLayout()
    max_words_row.addWidget(max_words_cb)
    max_words_row.addWidget(max_words_spin)
    max_words_row.addStretch()
    ocr_layout.addLayout(max_words_row)
    auto_detect_cb = QCheckBox("Auto-detect OCR text area (learn from first few readings)")
    auto_detect_cb.setChecked(settings.get("auto_detect_text_region", False))
    ocr_layout.addWidget(auto_detect_cb)
    allow_overlap_cb = QCheckBox("Allow overlay overlap with OCR area (causes flickering)")
    allow_overlap_cb.setChecked(settings.get("allow_overlap", False))
    allow_overlap_cb.setToolTip("When unchecked (default): overlap pauses OCR and shows a message. When checked: overlay hides briefly during capture, causing flicker.")
    ocr_layout.addWidget(allow_overlap_cb)
    
    # --- OCR Reconciler settings ---
    ocr_layout.addWidget(QLabel(""))
    reconciler_header = QLabel("<b>Reconciler Settings:</b>")
    ocr_layout.addWidget(reconciler_header)
    reconciler_desc = QLabel("Controls when OCR text is sent for translation (stability-based).")
    reconciler_desc.setWordWrap(True)
    ocr_layout.addWidget(reconciler_desc)
    
    # Create a form layout for reconciler settings
    reconciler_form = QFormLayout()
    
    # MT Reconciler (StreamingReconciler)
    mt_stability_label = QLabel("MT Stability threshold:")
    mt_stability_tooltip = "Stability threshold: How long OCR text must remain unchanged before sending for translation.\n\nLower = faster translation (may catch incomplete text)\nHigher = waits longer for complete text (more accurate)"
    mt_stability_label.setToolTip(mt_stability_tooltip)
    ocr_mt_stability = QDoubleSpinBox()
    ocr_mt_stability.setRange(0.1, 2.0)
    ocr_mt_stability.setSingleStep(0.1)
    ocr_mt_stability.setSuffix(" s")
    ocr_mt_stability.setValue(settings.get("ocr_mt_reconciler_stability", 0.2))
    ocr_mt_stability.setToolTip(mt_stability_tooltip)
    reconciler_form.addRow(mt_stability_label, ocr_mt_stability)
    
    # LLM Reconciler settings
    llm_stability_label = QLabel("LLM Stability threshold:")
    llm_stability_tooltip = "Stability threshold: How long OCR text must remain unchanged before sending for translation.\n\nLower = faster translation (may catch incomplete text)\nHigher = waits longer for complete text (more accurate)"
    llm_stability_label.setToolTip(llm_stability_tooltip)
    ocr_llm_stability = QDoubleSpinBox()
    ocr_llm_stability.setRange(0.05, 1.0)
    ocr_llm_stability.setSingleStep(0.05)
    ocr_llm_stability.setSuffix(" s")
    ocr_llm_stability.setValue(settings.get("ocr_llm_reconciler_stability", 0.12))
    ocr_llm_stability.setToolTip(llm_stability_tooltip)
    reconciler_form.addRow(llm_stability_label, ocr_llm_stability)
    
    llm_max_buffer_label = QLabel("LLM Max buffer time:")
    llm_max_buffer_tooltip = "Max buffer time: Maximum time to wait before forcing translation (even if text is still changing).\n\nPrevents long delays when text keeps updating. If text hasn't stabilized after this time, it sends anyway."
    llm_max_buffer_label.setToolTip(llm_max_buffer_tooltip)
    ocr_llm_max_buffer = QDoubleSpinBox()
    ocr_llm_max_buffer.setRange(0.2, 3.0)
    ocr_llm_max_buffer.setSingleStep(0.1)
    ocr_llm_max_buffer.setSuffix(" s")
    ocr_llm_max_buffer.setValue(settings.get("ocr_llm_reconciler_max_buffer", 0.6))
    ocr_llm_max_buffer.setToolTip(llm_max_buffer_tooltip)
    reconciler_form.addRow(llm_max_buffer_label, ocr_llm_max_buffer)
    
    # Minimum words before sending
    min_words_label = QLabel("Minimum words before send:")
    min_words_tooltip = "Don't translate if text has fewer than this many words.\n\n0 = translate any length (default)\nUseful to filter out noise from partial OCR captures or very short text."
    min_words_label.setToolTip(min_words_tooltip)
    ocr_min_words = QSpinBox()
    ocr_min_words.setRange(0, 50)
    ocr_min_words.setSuffix(" words")
    ocr_min_words.setValue(settings.get("ocr_min_words_before_translate", 0))
    ocr_min_words.setToolTip(min_words_tooltip)
    reconciler_form.addRow(min_words_label, ocr_min_words)
    
    # Similarity threshold for avoiding repeated translations
    similarity_label = QLabel("Similarity: substring extension (chars):")
    similarity_tooltip = "When new text contains recently translated text as a substring, skip translation if the extra chars are ≤ this value.\n\nHigher = more aggressive dedup (fewer repeated translations)\nLower = less dedup (may translate similar text again)"
    similarity_label.setToolTip(similarity_tooltip)
    ocr_similarity_chars = QSpinBox()
    ocr_similarity_chars.setRange(0, 50)
    ocr_similarity_chars.setSuffix(" chars")
    ocr_similarity_chars.setValue(settings.get("ocr_similarity_substring_chars", 15))
    ocr_similarity_chars.setToolTip(similarity_tooltip)
    reconciler_form.addRow(similarity_label, ocr_similarity_chars)
    
    ocr_layout.addLayout(reconciler_form)
    
    ocr_layout.addStretch()
    tabs.addTab(ocr_tab, "OCR")
    
    # --- Audio tab: reconciler (X sec period, Y checks) ---
    audio_tab = QWidget()
    audio_layout = QFormLayout(audio_tab)
    audio_layout.addRow(QLabel("Reconciler: within X seconds, check Y times for sentence completion:"))
    audio_reconciler_period = QDoubleSpinBox()
    audio_reconciler_period.setRange(0.5, 5.0)
    audio_reconciler_period.setSingleStep(0.5)
    audio_reconciler_period.setSuffix(" s")
    audio_reconciler_period.setValue(settings.get("audio_reconciler_period_sec", 2.0))
    audio_reconciler_period.setToolTip("Max seconds before forcing send (X).")
    audio_layout.addRow("Period (X sec):", audio_reconciler_period)
    audio_reconciler_checks = QSpinBox()
    audio_reconciler_checks.setRange(2, 20)
    audio_reconciler_checks.setValue(settings.get("audio_reconciler_num_checks", 4))
    audio_reconciler_checks.setToolTip("Number of completion checks in that period (Y).")
    audio_layout.addRow("Num checks (Y):", audio_reconciler_checks)
    audio_reconciler_min_words = QSpinBox()
    audio_reconciler_min_words.setRange(1, 50)
    audio_reconciler_min_words.setValue(settings.get("audio_reconciler_min_words", 7))
    audio_reconciler_min_words.setToolTip("Minimum word count before sending.")
    audio_layout.addRow("Min words:", audio_reconciler_min_words)
    audio_layout.addRow(QLabel(""))
    audio_layout.addRow(QLabel("Finalization (for phrase boundaries):"))
    audio_silence = QDoubleSpinBox()
    audio_silence.setRange(0.3, 3.0)
    audio_silence.setSingleStep(0.1)
    audio_silence.setSuffix(" s")
    audio_silence.setValue(settings.get("audio_silence_duration", 1.0))
    audio_silence.setToolTip("Silence duration to finalize a phrase.")
    audio_layout.addRow("Silence duration (finalize):", audio_silence)
    audio_max_phrase = QDoubleSpinBox()
    audio_max_phrase.setRange(2.0, 15.0)
    audio_max_phrase.setSingleStep(0.5)
    audio_max_phrase.setSuffix(" s")
    audio_max_phrase.setValue(settings.get("audio_max_phrase_duration", 5.0))
    audio_max_phrase.setToolTip("Force finalize after this many seconds of speech.")
    audio_layout.addRow("Max phrase duration:", audio_max_phrase)
    audio_layout.addRow(QLabel("Changes apply in real time when you click Apply."))
    tabs.addTab(audio_tab, "Audio")
    
    layout.addWidget(tabs)
    
    def gather_settings():
        return {
            "detect_mixed_content": detect_mixed_cb.isChecked(),
            "max_words_enabled": max_words_cb.isChecked(),
            "max_words_for_translation": max_words_spin.value(),
            "allow_overlap": allow_overlap_cb.isChecked(),
            "auto_detect_text_region": auto_detect_cb.isChecked(),
            "session_output_enabled": session_output_cb.isChecked(),
            "session_output_path": session_path_edit.text().strip(),
            "audio_reconciler_period_sec": audio_reconciler_period.value(),
            "audio_reconciler_num_checks": audio_reconciler_checks.value(),
            "audio_reconciler_min_words": audio_reconciler_min_words.value(),
            "audio_silence_duration": audio_silence.value(),
            "audio_max_phrase_duration": audio_max_phrase.value(),
            "ocr_mt_reconciler_stability": ocr_mt_stability.value(),
            "ocr_llm_reconciler_stability": ocr_llm_stability.value(),
            "ocr_llm_reconciler_max_buffer": ocr_llm_max_buffer.value(),
            "ocr_min_words_before_translate": ocr_min_words.value(),
            "ocr_similarity_substring_chars": ocr_similarity_chars.value(),
            "llm_context_count": llm_context_spin.value(),
        }
    
    def do_apply():
        s = gather_settings()
        save_app_settings(s)
        _apply_settings_to_translator(translator, s)
        if parent and hasattr(parent, "_add_status"):
            parent._add_status("Settings applied (real-time).")
    
    btns = QHBoxLayout()
    apply_btn = QPushButton("Apply")
    apply_btn.setToolTip("Save and apply now. Dialog stays open for real-time tuning.")
    apply_btn.clicked.connect(do_apply)
    btns.addWidget(apply_btn)
    btns.addStretch()
    btns2 = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    btns2.accepted.connect(dlg.accept)
    btns2.rejected.connect(dlg.reject)
    btns.addWidget(btns2)
    layout.addLayout(btns)
    dlg.installEventFilter(_DialogRaiseFilter(dlg))
    if dlg.exec_() == QDialog.Accepted:
        s = gather_settings()
        save_app_settings(s)
        _apply_settings_to_translator(translator, s)
        return True
    return False


class _LanguageSelector(QPushButton):
    """Dropdown that shows a fixed-height scrollable list (avoids broken native combos on macOS)."""
    # (label, code) tuples, selected index
    selection_changed = pyqtSignal(int)

    def __init__(self, items, default_idx=0, parent=None):
        super().__init__(parent)
        self._items = items  # list of (label, code)
        self._idx = default_idx
        self._menu = None
        self._update_text()

    def _update_text(self):
        self.setText(self._items[self._idx][0])

    def _build_menu(self):
        if self._menu:
            return
        self._menu = QMenu(self)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        lst = QListWidget()
        lst.setFixedHeight(200)
        lst.setMinimumWidth(220)
        lst.setUniformItemSizes(True)
        lst.setStyleSheet(f"""
            QListWidget {{ background: white; border: 1px solid {_BILIBILI_BLUE}; }}
            QListWidget::item {{ padding: 4px 8px; min-height: 24px; }}
            QListWidget::item:selected {{ background: {_BILIBILI_BLUE}; color: white; }}
        """)
        for lbl, _ in self._items:
            lst.addItem(lbl)
        lst.setCurrentRow(self._idx)
        lst.currentRowChanged.connect(self._on_row_changed)
        lst.itemClicked.connect(lambda item: self._menu.close())
        layout.addWidget(lst)
        act = QWidgetAction(self._menu)
        act.setDefaultWidget(container)
        self._menu.addAction(act)
        self._list = lst

    def _on_row_changed(self, row):
        self._idx = row
        self._update_text()
        self.selection_changed.emit(row)

    def get_index(self):
        return self._idx

    def showMenu(self):
        self._build_menu()
        self._list.setCurrentRow(self._idx)
        self._menu.exec_(self.mapToGlobal(self.rect().bottomLeft()))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.showMenu()
        else:
            super().mousePressEvent(e)


class DraggableTitleBar(QWidget):
    """Title bar that allows dragging the parent window when dragged."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_start = None
        self._window_start = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.globalPos()
            self._window_start = self.window().frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            delta = event.globalPos() - self._drag_start
            self.window().move(self._window_start + delta)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = None
        super().mouseReleaseEvent(event)


class LearnOverlay(QWidget):
    """Learn mode overlay: displays Chinese keywords, pinyin, and definitions. Positioned right side."""
    
    def __init__(self, left=70, top=80, width=450, height=400):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._default_height = height  # Store default height
        self.setMinimumSize(380, height)  # Set minimum to default height
        self.setMaximumHeight(800)  # Allow expansion but with reasonable max
        self.setGeometry(left, top, width, height)
        self._resize_start_pos = None
        self._resize_start_height = None
        self._tooltip_timer = QTimer()
        self._tooltip_timer.setSingleShot(True)
        # Create a custom tooltip label (separate window)
        self._tooltip_label = QLabel("Copied!")
        self._tooltip_label.setWindowFlags(Qt.ToolTip | Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self._tooltip_label.setAttribute(Qt.WA_TranslucentBackground)
        self._tooltip_label.setStyleSheet("""
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;

        """)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self._tooltip_label.setAlignment(Qt.AlignCenter)
        self._tooltip_label.adjustSize()
        self._tooltip_label.hide()
        self._tooltip_timer.timeout.connect(self._tooltip_label.hide)
        self._keywords = []  # List of keyword dicts: {word, pinyin, definition}
        self._seen_words = set()  # Track words we've already shown (no repeats)

        # Set background color for the widget - use object name to ensure it applies
        self.setObjectName("LearnOverlayWidget")
        self.setStyleSheet("""
            QWidget#LearnOverlayWidget {
                background: rgba(28, 24, 42, 0.85);
                border: none;
                border-radius: 8px;
            }
        """)

        # Layout
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # Header bar (entire row: title + buttons) - draggable
        title_row = DraggableTitleBar(self)
        title_row.setAttribute(Qt.WA_StyledBackground)  # required for stylesheet bg with translucent parent
        title_row.setStyleSheet("""
            background: rgba(28, 24, 42, 0.85);
            border: 1px solid rgba(0, 161, 214, 0.3);
            border-radius: 4px;
        """)
        title_row_layout = QHBoxLayout(title_row)
        title_row_layout.setContentsMargins(8, 4, 8, 4)
        title_row_layout.setSpacing(8)
        title = QLabel("Learn Keywords")
        title.setStyleSheet("color: white; font-size: 14px; font-weight: bold; padding: 0; margin: 0; background: transparent; border: none;")
        title.setAttribute(Qt.WA_TransparentForMouseEvents)  # so drag goes to title bar
        title_row_layout.addWidget(title, 0)
        title_row_layout.addSpacing(8)
        live_btn = QPushButton("Live")
        starred_btn = QPushButton("Starred")
        for btn in (live_btn, starred_btn):
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(0, 161, 214, 0.0);
                    color: white;
                    border: 0px solid rgba(0, 161, 214, 1.0);
                    border-radius: 3px;
                    padding: 4px 12px;
                    font-size: 12px;
                }
                QPushButton:hover { background: rgba(0, 161, 214, 0.4); }
                QPushButton:checked { background: rgba(0, 161, 214, 0.7); }
            """)
            btn.setCheckable(True)
        tab_btns = QButtonGroup(self)
        tab_btns.addButton(live_btn)
        tab_btns.addButton(starred_btn)
        live_btn.setChecked(True)
        # live_btn.setToolTip("New words from current session")
        # starred_btn.setToolTip("Starred words (saved)")
        title_row_layout.addWidget(live_btn)
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Plain)
        sep.setFixedWidth(1)
        sep.setStyleSheet("background: rgba(255,255,255,0.4);")
        sep.setAttribute(Qt.WA_TransparentForMouseEvents)  # so drag goes to title bar
        title_row_layout.addWidget(sep)
        title_row_layout.addWidget(starred_btn)
        title_row_layout.addSpacing(8)
        save_btn = QPushButton("Save")
        save_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0, 161, 214, 0.6);
                color: white;
                border: 0px solid rgba(0, 161, 214, 0.5);
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 12px;
            }
            QPushButton:hover { background: rgba(0, 161, 214, 0.8); }
            QPushButton:pressed { background: rgba(0, 161, 214, 1); }
        """)
        save_btn.clicked.connect(self._save_to_markdown)
        # save_btn.setToolTip("Save all keywords to a Markdown file")
        title_row_layout.addWidget(save_btn)
        layout.addWidget(title_row)

        # Stacked content: Live (session) and Starred (persistent)
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("""
            QStackedWidget { background: rgba(28, 24, 42, 0.85); border: 1px solid rgba(0, 161, 214, 0.3); border-radius: 6px; }
        """)

        # Live page: session keywords
        live_container = QWidget()
        live_layout = QVBoxLayout(live_container)
        live_layout.setContentsMargins(0, 0, 0, 0)
        self.list_widget = QListWidget()
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Enable selection and copying
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_widget.setStyleSheet("""
            QListWidget {
                background: rgba(28, 24, 42, 0.75);
                border: none;
                border-radius: 6px;
                color: rgba(255, 255, 255, 0.95);
                padding: 4px;
            }
            QListWidget::item {
                border: none;
                border-radius: 4px;
                padding: 4px 6px;
                margin: 2px 0px;
                background: rgba(255, 255, 255, 0.05);
                min-height: 50px;
            }
            QListWidget::item:hover {
                background: rgba(255, 255, 255, 0.1);
                border: none;
            }
            QListWidget::item:selected {
                background: rgba(0, 161, 214, 0.3);
                border: none;
            }
            QListWidget::item:selected:hover {
                background: rgba(0, 161, 214, 0.3);
                border: none;
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.1);
                width: 8px;
                border: none;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.5);
            }
        """)
        live_layout.addWidget(self.list_widget, 1)
        self.stacked_widget.addWidget(live_container)

        # Starred page: persistent from DB
        starred_container = QWidget()
        starred_layout = QVBoxLayout(starred_container)
        starred_layout.setContentsMargins(0, 0, 0, 0)
        self.starred_list_widget = QListWidget()
        self.starred_list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.starred_list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.starred_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.starred_list_widget.setStyleSheet(self.list_widget.styleSheet())
        self.starred_placeholder = QLabel("No starred words. Click ★ on words in Live tab to add.")
        self.starred_placeholder.setStyleSheet("color: rgba(255, 255, 255, 0.6); font-size: 12px; padding: 160px 4px 4px 4px;")
        self.starred_placeholder.setAlignment(Qt.AlignCenter)
        self.starred_placeholder.setWordWrap(True)
        starred_layout.addWidget(self.starred_placeholder)
        starred_layout.addWidget(self.starred_list_widget, 1)
        self.starred_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.starred_list_widget.customContextMenuRequested.connect(self._show_starred_context_menu)
        self.stacked_widget.addWidget(starred_container)

        def _on_live_clicked():
            self.stacked_widget.setCurrentIndex(0)
        def _on_starred_clicked():
            self.stacked_widget.setCurrentIndex(1)
            self._refresh_starred_tab()
        live_btn.clicked.connect(_on_live_clicked)
        starred_btn.clicked.connect(_on_starred_clicked)

        layout.addWidget(self.stacked_widget, 1)

        # Resize handle indicator
        resize_handle = QLabel("━━━")
        resize_handle.setStyleSheet("""
            color: rgba(0, 161, 214, 0.7);
            font-size: 10px; 
            padding: 6px 4px;
            background: rgba(28, 24, 42, 0.85);
            border: 1px solid rgba(0, 161, 214, 0.3);
            border-radius: 4px;
        """)
        resize_handle.setAlignment(Qt.AlignCenter)
        resize_handle.setCursor(Qt.SizeVerCursor)
        layout.addWidget(resize_handle)

        # Status/placeholder
        self.placeholder = QLabel("Waiting for Chinese subtitles...")
        self.placeholder.setStyleSheet("color: rgba(255, 255, 255, 0.6); font-size: 12px; padding: 4px;")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setWordWrap(True)
        layout.addWidget(self.placeholder)

        self.setLayout(layout)
        
        # Enable context menu for copying
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        
        # Enable keyboard shortcuts (Ctrl+C) for both lists
        def _install_copy_shortcut(list_widget):
            orig = list_widget.keyPressEvent
            def wrapper(event):
                if event.key() == Qt.Key_C and event.modifiers() == Qt.ControlModifier:
                    self._copy_selected_from(list_widget)
                    event.accept()
                else:
                    orig(event)
            list_widget.keyPressEvent = wrapper
        self.list_widget.setFocusPolicy(Qt.StrongFocus)
        self.starred_list_widget.setFocusPolicy(Qt.StrongFocus)
        _install_copy_shortcut(self.list_widget)
        _install_copy_shortcut(self.starred_list_widget)

    def update_keywords(self, keywords: list[dict]):
        """Append new keywords to the list (don't clear existing ones)."""
        if not keywords or not isinstance(keywords, list):
            if self.list_widget.count() == 0:
                self.placeholder.setText("No keywords found")
                self.placeholder.show()
            return

        self.placeholder.hide()
        
        # Check if user was at bottom BEFORE adding items (adding changes scrollbar max)
        sb = self.list_widget.verticalScrollBar()
        was_at_bottom = sb.value() >= sb.maximum() - 10
        
        # Append only new keywords that we haven't seen before (no repeated words)
        new_count = 0
        for kw in keywords:
            word = kw.get("word", "")
            if not word or word in self._seen_words:
                continue  # Skip if we've already shown this word
            
            self._seen_words.add(word)
            pinyin = kw.get("pinyin", "")
            definition = kw.get("definition", "")
            # Preserve metadata if present
            keyword_dict = {"word": word, "pinyin": pinyin, "definition": definition}
            if "_metadata" in kw:
                keyword_dict["_metadata"] = kw["_metadata"]
            self._keywords.append(keyword_dict)

            # Format item and append to bottom
            item = QListWidgetItem()
            try:
                from starred_db import is_starred
                starred = is_starred(word)
            except Exception:
                starred = False
            # Get metadata from keyword dict if present
            metadata = kw.get("_metadata")
            widget = self._create_keyword_widget(word, pinyin, definition, is_starred=starred, for_starred_tab=False, metadata=metadata)
            
            # Calculate accurate line count using QFontMetrics
            # All measurements in pixels: width=170px, font-size=12px
            font = QFont()
            font.setPixelSize(12)  # Use pixel size for consistency
            metrics = QFontMetrics(font)
            definition_width_px = 170  # Maximum width in pixels
            # Calculate how many lines the text actually takes
            text_rect = metrics.boundingRect(0, 0, definition_width_px, 0, Qt.TextWordWrap | Qt.AlignLeft, definition)
            line_height_px = metrics.lineSpacing()  # Actual line height in pixels (~14-15px for 12px font)
            num_lines = max(1, (text_rect.height() + line_height_px - 1) // line_height_px)  # Round up
            
            # Default height for 1-3 lines, then scale per line after that
            default_height_px = 60  # Default height for 1-3 lines
            item_padding_px = 8  # Item padding top/bottom (4px each)
            safety_margin_px = 6  # Safety margin to prevent cutoff (top and bottom)
            
            if num_lines <= 3:
                # Use default height for 1-3 lines + safety margin
                item_height = default_height_px + safety_margin_px
            else:
                # Default height + extra height for lines beyond 3 + safety margin
                extra_lines = num_lines - 3
                extra_height_px = extra_lines * line_height_px
                item_height = default_height_px + extra_height_px + safety_margin_px
            
            item.setSizeHint(QSize(widget.sizeHint().width(), item_height))
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)
            new_count += 1

        # Auto-scroll to bottom only if user was already at the bottom (within 10px)
        if new_count > 0 and was_at_bottom:
            self.list_widget.scrollToBottom()

        # Ensure height stays fixed - don't resize based on content
        current_height = self.height()
        if current_height < self._default_height:
            self.resize(self.width(), self._default_height)

    def _create_keyword_widget(self, word: str, pinyin: str, definition: str, is_starred: bool = False, for_starred_tab: bool = False, metadata: dict = None) -> QWidget:
        """Create a keyword list item widget with star button."""
        widget = QWidget()
        widget.setStyleSheet("background: transparent; border: none;")
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 4, 6, 4)
        layout.setSpacing(4)

        # Star button
        star_btn = QPushButton("★" if is_starred else "☆")
        star_btn.setFixedSize(28, 28)
        star_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #FFD700;
                font-size: 16px;
                border: none;
                padding: 0;
            }
            QPushButton:hover { color: #FFA500; }
        """)
        star_btn.setToolTip("Remove from starred" if for_starred_tab else "Add to starred")
        star_btn.setCursor(Qt.PointingHandCursor)

        def _on_star_clicked():
            try:
                from starred_db import add_star, remove_star, get_all_starred
                if for_starred_tab:
                    remove_star(word)
                    self._refresh_starred_tab()
                else:
                    # Get metadata from keyword if available
                    metadata = getattr(widget, "_keyword_metadata", None)
                    provider = metadata.get("provider") if metadata else None
                    provider_display = metadata.get("provider_display") if metadata else None
                    model = metadata.get("model") if metadata else None
                    add_star(word, pinyin, definition, provider=provider, provider_display=provider_display, model=model)
                    star_btn.setText("★")
                    star_btn.setToolTip("Remove from starred")
            except Exception:
                pass

        star_btn.clicked.connect(_on_star_clicked)


        # Chinese word
        word_lbl = QLabel(word)
        word_lbl.setStyleSheet("color: white; font-size: 14px; font-weight: bold; padding: 0px; margin: 0px;")
        word_lbl.setMinimumWidth(50)
        layout.addWidget(word_lbl, 0)

        # Separator (no spacing)
        sep = QLabel("│")
        sep.setStyleSheet("color: rgba(255, 255, 255, 0.3); padding: 0px; margin: 0px;")
        layout.addWidget(sep, 0)

        # Pinyin (directly next to word with no spacing)
        pin_lbl = QLabel(pinyin)
        pin_lbl.setStyleSheet("color: #B5BCC5; font-size: 12px; padding: 0px; margin: 0px;")
        pin_lbl.setMinimumWidth(90)
        layout.addWidget(pin_lbl, 0)
        
        # Add spacer to push definition to right
        layout.addSpacing(15)

        # Definition - centered vertically, limited width to prevent cutoff
        def_lbl = QLabel(definition)
        def_lbl.setStyleSheet("color: rgba(255, 255, 255, 0.9); font-size: 12px; padding: 0px; margin: 0px;")  # No padding on label itself
        def_lbl.setWordWrap(True)
        def_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Vertical center instead of top
        # Limit width to prevent cutoff
        def_lbl.setMaximumWidth(170)
        layout.addWidget(def_lbl, 1)
        layout.addWidget(star_btn, 0)
        widget.setLayout(layout)
        # Store metadata in widget for later retrieval
        if metadata:
            widget._keyword_metadata = metadata
        return widget

    def _refresh_starred_tab(self):
        """Load starred words from DB and populate the Starred tab."""
        try:
            from starred_db import get_all_starred
            starred = get_all_starred()
        except Exception:
            starred = []
        self.starred_list_widget.clear()
        if not starred:
            self.starred_placeholder.setText("No starred words. Click ★ on words in Live tab to add.")
            self.starred_placeholder.show()
        else:
            self.starred_placeholder.hide()
            for kw in starred:
                item = QListWidgetItem()
                metadata = kw.get("_metadata")
                widget = self._create_keyword_widget(
                    kw["word"], kw["pinyin"], kw["definition"],
                    is_starred=True, for_starred_tab=True, metadata=metadata
                )
                self.starred_list_widget.addItem(item)
                item.setSizeHint(QSize(widget.sizeHint().width(), 60))
                self.starred_list_widget.setItemWidget(item, widget)

    def _show_starred_context_menu(self, position):
        menu = QMenu(self)
        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(lambda: self._copy_selected_from(self.starred_list_widget))
        menu.exec_(self.starred_list_widget.mapToGlobal(position))

    def _copy_selected_from(self, list_widget):
        """Copy selected items from the given list widget."""
        selected_items = list_widget.selectedItems()
        if not selected_items:
            return
        texts = []
        for item in selected_items:
            widget = list_widget.itemWidget(item)
            if widget:
                labels = widget.findChildren(QLabel)
                parts = [lbl.text() for lbl in labels if lbl.text() and lbl.text() != "│"]
                if parts:
                    text_line = " | ".join(parts)
                    # Add metadata if available
                    metadata = getattr(widget, "_keyword_metadata", None)
                    if metadata:
                        provider_display = metadata.get("provider_display", "")
                        model = metadata.get("model", "")
                        if provider_display or model:
                            meta_parts = []
                            if provider_display:
                                meta_parts.append(f"Provider: {provider_display}")
                            if model and model != "default":
                                meta_parts.append(f"Model: {model}")
                            if meta_parts:
                                text_line += f" [{', '.join(meta_parts)}]"
                    texts.append(text_line)
        if texts:
            QApplication.clipboard().setText("\n".join(texts))
            if getattr(self, "_tooltip_label", None):
                self._tooltip_label.setText("Copied!")
                self._tooltip_label.move(QCursor.pos().x() + 10, QCursor.pos().y() - 30)
                self._tooltip_label.show()
                self._tooltip_label.raise_()
                self._tooltip_timer.stop()
                self._tooltip_timer.start(2000)

    def clear_keywords(self):
        """Clear all keywords and show placeholder."""
        self.list_widget.clear()
        self._keywords.clear()
        self._seen_words.clear()
        self.placeholder.setText("No Chinese text detected")
        self.placeholder.show()

    def _save_to_markdown(self):
        """Save keywords to a Markdown file. If Starred tab is active, saves starred words with default filename."""
        is_starred_tab = self.stacked_widget.currentIndex() == 1
        if is_starred_tab:
            try:
                from starred_db import get_all_starred
                keywords = get_all_starred()
            except Exception:
                keywords = []
            default_path = os.path.join(os.path.expanduser("~"), "starred words.md")
            caption = "Save Starred Keywords"
        else:
            keywords = self._keywords
            if not keywords:
                keywords = []
                for i in range(self.list_widget.count()):
                    item = self.list_widget.item(i)
                    widget = self.list_widget.itemWidget(item)
                    if widget:
                        labels = widget.findChildren(QLabel)
                        parts = [lbl.text() for lbl in labels if lbl.text() and lbl.text() != "│"]
                        if len(parts) >= 1:
                            word = parts[0]
                            pinyin = parts[1] if len(parts) > 1 else ""
                            definition = parts[2] if len(parts) > 2 else ""
                            keywords.append({"word": word, "pinyin": pinyin, "definition": definition})
            default_path = ""
            caption = "Save Keywords"
        if not keywords:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, caption, default_path, "Markdown (*.md);;All Files (*)"
        )
        if not path:
            return
        try:
            def esc(s):
                return (s or "").replace("|", "\\|").replace("\n", " ")
            title = "# Starred Keywords\n" if is_starred_tab else "# Learn Keywords\n"
            lines = [title, f"*{len(keywords)} words*\n"]
            lines.append("| Word | Pinyin | Definition | Provider | Model |")
            lines.append("|------|--------|------------|----------|-------|")
            for kw in keywords:
                word = esc(kw.get("word", ""))
                pinyin = esc(kw.get("pinyin", ""))
                definition = esc(kw.get("definition", ""))
                metadata = kw.get("_metadata", {})
                provider_display = esc(metadata.get("provider_display", ""))
                model = esc(metadata.get("model", ""))
                lines.append(f"| {word} | {pinyin} | {definition} | {provider_display} | {model} |")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            if getattr(self, "_tooltip_label", None):
                self._tooltip_label.setText("Saved!")
                cursor_pos = QCursor.pos()
                self._tooltip_label.move(cursor_pos.x() + 10, cursor_pos.y() - 30)
                self._tooltip_label.show()
                self._tooltip_label.raise_()
                self._tooltip_timer.stop()
                self._tooltip_timer.start(2000)
        except OSError as e:
            if getattr(self, "_tooltip_label", None):
                self._tooltip_label.setText(f"Save failed: {e}")
                self._tooltip_label.show()
                self._tooltip_timer.stop()
                self._tooltip_timer.start(3000)

    def _show_context_menu(self, position):
        """Show context menu for copying selected items."""
        menu = QMenu(self)
        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(lambda: self._copy_selected_from(self.list_widget))
        menu.exec_(self.list_widget.mapToGlobal(position))
    
    def _copy_selected(self):
        """Copy selected items to clipboard (Live tab)."""
        self._copy_selected_from(self.list_widget)
    
    def mousePressEvent(self, event):
        """Handle mouse press for resizing."""
        if event.button() == Qt.LeftButton:
            # Check if clicking near bottom edge (resize area - last 20px)
            bottom_margin = 45  # 20px from bottom for easier grabbing
            if event.pos().y() >= self.height() - bottom_margin:
                self._resize_start_pos = event.globalPos()
                self._resize_start_height = self.height()
                self.setCursor(Qt.SizeVerCursor)
                event.accept()
                return
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for resizing."""
        if self._resize_start_pos is not None:
            # Resizing
            delta_y = event.globalPos().y() - self._resize_start_pos.y()
            new_height = max(self._default_height, min(800, self._resize_start_height + delta_y))
            self.resize(self.width(), new_height)
            event.accept()
        else:
            # Check if hovering over resize area (last 20px)
            bottom_margin = 45
            if event.pos().y() >= self.height() - bottom_margin:
                self.setCursor(Qt.SizeVerCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release after resizing."""
        if event.button() == Qt.LeftButton:
            self._resize_start_pos = None
            self._resize_start_height = None
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event):
        """Forward wheel events to list widget for scrolling."""
        # Forward wheel events to the list widget so it can scroll
        if self.list_widget:
            QApplication.sendEvent(self.list_widget, event)
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        """Handle escape to close window."""
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            qa = getattr(self, "_quit_all", None)
            if qa:
                qa()
            else:
                self.window().close()


def show_language_dialog(parent=None):
    """Show language selection dialog. Returns (source_lang, target_lang) as internal codes."""
    dlg = QDialog(parent)
    dlg.setWindowTitle("BiliOCR")
    dlg.setMinimumWidth(340)
    dlg.setStyleSheet(f"""
        QDialog {{
            background: rgba(255, 255, 255, 0.92);
        }}
        QLabel {{
            color: #333;
            font-size: 13px;
        }}
        QPushButton {{
            background: {_BILIBILI_BLUE};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 18px;
        }}
        QPushButton#lang_selector {{
            background: rgba(255, 255, 255, 0.95);
            color: #333;
            border: 2px solid {_BILIBILI_BLUE};
            text-align: left;
            padding: 6px 10px;
        }}
        QPushButton#lang_selector:hover {{
            border-color: #0090bc;
            background: rgba(255, 255, 255, 0.98);
        }}
        QPushButton#lang_selector:pressed {{
            background: rgba(240, 240, 250, 1);
        }}
        QPushButton:hover {{
            background: #0090bc;
        }}
        QPushButton:pressed {{
            background: #007a9e;
        }}
        QDialogButtonBox QPushButton[text="OK"] {{
            background: {_BILIBILI_BLUE};
        }}
        QDialogButtonBox QPushButton[text="Cancel"] {{
            background: #aaa;
            color: #333;
        }}
    """)

    layout = QVBoxLayout(dlg)
    layout.setSpacing(14)
    layout.setContentsMargins(24, 24, 24, 24)

    title = QLabel("BiliOCR")
    title.setStyleSheet(f"color: {_BILIBILI_PURPLE}; font-size: 18px; font-weight: bold;")
    title.setAlignment(Qt.AlignCenter)
    layout.addWidget(title)

    row = QHBoxLayout()
    row.setSpacing(12)
    row.addWidget(QLabel("From:"))
    from_sel = _LanguageSelector(_LANG_OPTIONS, 0, dlg)  # Chinese
    from_sel.setObjectName("lang_selector")
    row.addWidget(from_sel, 1)
    layout.addLayout(row)

    row2 = QHBoxLayout()
    row2.setSpacing(12)
    row2.addWidget(QLabel("To:"))
    to_sel = _LanguageSelector(_LANG_OPTIONS_TARGET, 3, dlg)  # English
    to_sel.setObjectName("lang_selector")
    row2.addWidget(to_sel, 1)
    layout.addLayout(row2)

    model_grp = QButtonGroup()
    small_rb = QRadioButton("Small model (Machine Translation e.g. DeepL, Google Translate.)")
    large_rb = QRadioButton("Large model (LLM, e.g. GPT-4o, DeepSeek, Claude)")
    small_rb.setChecked(True)
    model_grp.addButton(small_rb)
    model_grp.addButton(large_rb)
    layout.addWidget(small_rb)
    layout.addWidget(large_rb)

    _LLM_PROVIDERS = [
        ("SiliconFlow.com", "siliconflow_com"),
        ("SiliconFlow.cn", "siliconflow_cn"),
        ("OpenAI ", "openai"),
        ("DeepSeek", "deepseek"),
        ("Anthropic", "anthropic"),
        ("Groq", "groq"),
        ("Together", "together"),
        ("HuggingFace API", "huggingface_api"),
        ("HuggingFace Local", "huggingface_local"),
    ]
    _LLM_MODELS = {
        "siliconflow_com": [
            # High Quality Tier
            ("Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"), 
            ("Qwen2.5-14B (Recommended)", "Qwen/Qwen2.5-14B-Instruct"), 
            ("Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct"),
            ("Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct"),  
            ("Qwen3-8B", "Qwen/Qwen3-8B"), # $0.06/ M Tokens  Best quality for complex dialogue
            ("DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3.2"),
            ("Tencent Hunyuan-MT-7B (Recommended)", "tencent/Hunyuan-MT-7B"),  # Excellent Chinese, very cheap
            # ("Deepseek-V3.1", "deepseek-ai/DeepSeek-V3.1"),  # Excellent Chinese, very cheap
            # ("GLM-4-9B", "THUDM/GLM-4-9B-0414"),  #$0.086/M tokens
            ("GLM-4-32B", "THUDM/GLM-4-32B-0414"),  #$0.27/M Tokens
            # ("moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.5"),
            ("MiniMax-M2.1", "MiniMaxAI/MiniMax-M2.1"),
        ],
        "siliconflow_cn": [
            # High Quality Tier
            ("Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"), 
            ("Qwen2.5-14B", "Qwen/Qwen2.5-14B-Instruct"), 
            ("Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct"),
            ("Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct"),  # Best quality for complex dialogue
            ("DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3.2"),  # Excellent Chinese, very cheap
            ("Deepseek V3 (Pro)", "Prodeepseek-ai/DeepSeek-V3"),  
            # Fast/Cheap Tier (Pro/ optimized endpoints)
            ("GLM-4-9B", "THUDM/GLM-4-9B-0414"),  
            ("GLM-4-32B", "THUDM/GLM-4-32B-0414"),  
            ("moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.5"),
            ("MiniMax-M2.1", "MiniMaxAI/MiniMax-M2.1"),
        ],
        "openai": [
          
            ("GPT-4o mini (Recommended)", "gpt-4o-mini"),              # Fastest, cheapest, good enough for most subs
            ("GPT-4o", "gpt-4o"),                        # Best quality for nuance/idioms
            # ("GPT-4o-latest", "gpt-4o-2024-11-20"),      # Auto-updates to newest 4o                     # Reasoning model - good for ambiguous context
            ("GPT-3.5 Turbo", "gpt-3.5-turbo"),  
            ("o3-mini", "o3-mini"), 
            ("GPT-5", "gpt-5"),              # Fastest, cheapest, good enough for most subs
            ("GPT-5 mini", "gpt-5-mini"),    
            ("GPT-5 nano", "gpt-5-nano"),          # Legacy fallback (cheapest)
        ],

        "deepseek": [
            ("DeepSeek-V3", "deepseek-chat"),            # General chat (the V3 model) - excellent Chinese
            ("DeepSeek-V2.5", "deepseek-chat-v2.5"),     # Older but very cheap fallback (if supported)
        ],

        # "gemini": [
        #     ("Gemini 1.5 Flash", "gemini-1.5-flash"),      # Cheap, fast, good enough
        #     ("Gemini 1.5 Pro", "gemini-1.5-pro"),          # Quality, huge context
        #     ("Gemini 1.5 Flash-8B", "gemini-1.5-flash-8b") # Experimental, fastest
        # ],
        "anthropic": [
            ("Claude 3.5 Haiku", "claude-3-5-haiku-20241022"),     # Fastest Claude, great for real-time
            ("Claude 3.5 Sonnet", "claude-3-5-sonnet-20241022"),   # Best quality/price ratio
            ("Claude 3.5 Sonnet v2", "claude-3-5-sonnet-20241022-v2"),  # Latest version with better instruction following
            ("Claude 3 Opus", "claude-3-opus-20240229"),           # Heavy quality (slowest, most expensive) - only for film/artistic content
        ],
        # "groq": [("Llama 3.1 8B", "llama-3.1-8b-instant"), ("Llama 3.1 70B", "llama-3.1-70b-versatile"), ("Mixtral 8x7B", "mixtral-8x7b-32768")],
        # "together": [("Llama 3 8B", "meta-llama/Llama-3-8b-chat-hf"), ("Llama 3 70B", "meta-llama/Llama-3-70b-chat-hf"), ("Mixtral 8x7B", "mistralai/Mixtral-8x7B-Instruct-v0.1")],
        "huggingface_api": [("opus-mt-zh-en", "Helsinki-NLP/opus-mt-zh-en"), ("nllb-200", "facebook/nllb-200-distilled-600M"), ("m2m100", "facebook/m2m100_418M"), ("Qwen2-7B", "Qwen/Qwen2-7B-Instruct")],
        "huggingface_local": [("opus-mt-zh-en", "Helsinki-NLP/opus-mt-zh-en"), ("nllb-200", "facebook/nllb-200-distilled-600M"), ("m2m100", "facebook/m2m100_418M"), ("Qwen2-7B", "Qwen/Qwen2-7B-Instruct")],
    }
    llm_label = QLabel("LLM provider:")
    llm_sel = _LanguageSelector(_LLM_PROVIDERS, 0, dlg)
    llm_sel.setObjectName("lang_selector")
    model_label = QLabel("Model:")
    model_combo = QComboBox()
    model_combo.setObjectName("model_combo")
    
    # API key validation warning label
    llm_warning_label = QLabel("")
    llm_warning_label.setStyleSheet("color: red; font-size: 11px; padding-top: 0px;")
    llm_warning_label.setWordWrap(True)
    llm_warning_label.setVisible(False)
    
    def check_llm_provider_api_key(provider_id):
        """Check if a provider has an API key available. Returns (has_key, key_name)."""
        key_map = {
            "siliconflow_com": "SILICONFLOW_COM_API_KEY",
            "siliconflow_cn": "SILICONFLOW_CN_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "together": "TOGETHER_API_KEY",
            "huggingface_api": "HF_API_KEY",
            "huggingface_local": None,  # No key needed
        }
        key_env = key_map.get(provider_id)
        if key_env is None:
            return (True, None)  # No key needed
        if isinstance(key_env, tuple):
            # Multiple keys needed
            has_all = all(os.environ.get(k) for k in key_env)
            return (has_all, key_env[0] if not has_all else None)
        has_key = bool(os.environ.get(key_env))
        return (has_key, key_env if not has_key else None)
    
    def _populate_models():
        provider = _LLM_PROVIDERS[llm_sel.get_index()][1]
        models = _LLM_MODELS.get(provider, _LLM_MODELS["siliconflow_com"])
        model_combo.blockSignals(True)
        model_combo.clear()
        for disp, _ in models:
            model_combo.addItem(disp)
        model_combo.setCurrentIndex(0)
        model_combo.blockSignals(False)
        
        # Check API key and update warning
        has_key, missing_key = check_llm_provider_api_key(provider)
        display_name = _LLM_PROVIDERS[llm_sel.get_index()][0]
        if not has_key:
            llm_warning_label.setText(f"Please add {display_name} API key")
            llm_warning_label.setVisible(True)
        else:
            llm_warning_label.setVisible(False)
    
    _populate_models()
    llm_sel.selection_changed.connect(_populate_models)
    llm_row = QHBoxLayout()
    llm_row.addWidget(llm_label)
    llm_row.addWidget(llm_sel, 1)
    model_row = QHBoxLayout()
    model_row.addWidget(model_label)
    model_row.addWidget(model_combo, 1)
    llm_container = QWidget()
    llm_layout = QVBoxLayout(llm_container)
    llm_layout.setContentsMargins(0, 0, 0, 0)
    llm_layout.addLayout(llm_row)
    llm_layout.addLayout(model_row)
    llm_layout.addWidget(llm_warning_label)
    layout.addWidget(llm_container)
    llm_container.setVisible(False)
    large_rb.toggled.connect(llm_container.setVisible)

    # Transcription mode tabs
    mode_tabs = QTabWidget()
    mode_tabs.setStyleSheet(f"""
        QTabWidget::pane {{
            border: 1px solid #ddd;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.8);
            margin-top: 4px;
        }}
        QTabBar::tab {{
            background: #f0f0f0;
            color: #333;
            padding: 8px 20px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            margin-right: 2px;
            font-size: 13px;
        }}
        QTabBar::tab:selected {{
            background: {_BILIBILI_BLUE};
            color: white;
            font-weight: bold;
        }}
    """)
    
    # OCR Tab
    ocr_tab = QWidget()
    ocr_layout = QVBoxLayout(ocr_tab)
    ocr_layout.setContentsMargins(16, 16, 16, 16)
    ocr_info = QLabel("Extract and translate subtitles from a selected screen region.\nIdeal for video content with hardcoded subtitles.")
    ocr_info.setWordWrap(True)
    ocr_info.setStyleSheet("color: #666; font-size: 12px;")
    ocr_layout.addWidget(ocr_info)
    
    # OCR Backend selection (moved here from main dialog)
    ocr_label = QLabel("OCR Backend:")
    ocr_combo = QComboBox()
    ocr_combo.setObjectName("ocr_backend_combo")  # Set object name for easy lookup
    ocr_combo.addItem("Vision (Local, Fast)", "vision")
    ocr_combo.addItem("EasyOCR (Local, Small Model)", "easyocr")
    ocr_combo.setCurrentIndex(0)  # Default to Vision
    ocr_row = QHBoxLayout()
    ocr_row.addWidget(ocr_label)
    ocr_row.addWidget(ocr_combo, 1)
    ocr_layout.addLayout(ocr_row)
    
    # Speech (TTS) panel - OCR mode only
    tts_label = QLabel("Speech (TTS):")
    tts_combo = QComboBox()
    tts_combo.setObjectName("tts_backend_combo")
    tts_combo.addItem("Piper (Local, English primarily, Free)", "piper")
    # tts_combo.addItem("XTTS v2 (Local, Free, ~2GB)", "xtts")
    tts_combo.addItem("OpenAI (tts-1) (API, $0.015 per 1K characters)", "openai")
    tts_combo.addItem("ElevenLabs Flash v2.5 (API, English only, free tier / paid)", "elevenlabs")
    tts_combo.addItem("ElevenLabs Multilingual v2 (API, Multilingual, free tier / paid)", "elevenlabs_multilingual_v2")
    tts_combo.setCurrentIndex(0)  # Default to Piper
    tts_row = QHBoxLayout()
    tts_row.addWidget(tts_label)
    tts_row.addWidget(tts_combo, 1)
    ocr_layout.addLayout(tts_row)
    
    # TTS Voice - dropdown that updates when backend changes
    from tts_engine import PIPER_VOICES, OPENAI_VOICES, ELEVENLABS_VOICES, XTTS_VOICES
    tts_voice_label = QLabel("Voice:")
    tts_voice_combo = QComboBox()
    tts_voice_combo.setObjectName("tts_voice_combo")
    tts_speed_label = QLabel("Speed:")
    tts_speed_spin = QDoubleSpinBox()
    tts_speed_spin.setObjectName("tts_speed_spin")
    tts_speed_spin.setRange(0.5, 2.0)
    tts_speed_spin.setSingleStep(0.1)
    tts_speed_spin.setValue(1.2)
    tts_speed_spin.setToolTip("Speech speed (OpenAI only). 1.0 = normal, 1.2 = faster.")
    
    def _populate_tts_voice_combo(backend_id):
        tts_voice_combo.blockSignals(True)
        tts_voice_combo.clear()
        bid = (backend_id or "piper").lower()
        if bid == "piper":
            for disp, vid in PIPER_VOICES:
                tts_voice_combo.addItem(disp, vid)
            tts_speed_label.setVisible(False)
            tts_speed_spin.setVisible(False)
        elif bid == "openai":
            for disp, vid in OPENAI_VOICES:
                tts_voice_combo.addItem(disp, vid)
            tts_speed_label.setVisible(True)
            tts_speed_spin.setVisible(True)
        elif bid in ("elevenlabs", "elevenlabs_multilingual_v2"):
            for disp, vid in ELEVENLABS_VOICES:
                tts_voice_combo.addItem(disp, vid)
            tts_speed_label.setVisible(False)
            tts_speed_spin.setVisible(False)
        elif bid == "xtts":
            for disp, vid in XTTS_VOICES:
                tts_voice_combo.addItem(disp, vid)
            tts_speed_label.setVisible(False)
            tts_speed_spin.setVisible(False)
        else:
            tts_voice_combo.addItem("Default", "default")
            tts_speed_label.setVisible(False)
            tts_speed_spin.setVisible(False)
        tts_voice_combo.setCurrentIndex(0)
        tts_voice_combo.blockSignals(False)
    
    _populate_tts_voice_combo("piper")
    tts_combo.currentIndexChanged.connect(lambda: _populate_tts_voice_combo(tts_combo.currentData()))
    
    tts_voice_row = QHBoxLayout()
    tts_voice_row.addWidget(tts_voice_label)
    tts_voice_row.addWidget(tts_voice_combo, 1)
    tts_voice_row.addWidget(tts_speed_label)
    tts_voice_row.addWidget(tts_speed_spin)
    ocr_layout.addLayout(tts_voice_row)
    
    # TTS model language support info - updates when backend changes
    TTS_LANG_INFO = {
        "openai": (
            "Cloud based, medium to high latency.\n"
            "English (Native – best), Strong American accent when speaking other languages.\n"
            "Supports Chinese/Mandarin (Fluent), Japanese (Fluent), Korean (Fluent), "
            "German (Fluent), Spanish (Fluent), French (Fluent), Italian (Fluent), Portuguese (Fluent), "
            "Dutch (Fluent), Russian (Fluent), Turkish (Good), Vietnamese (Good), Arabic (Good), "
            "Hindi (Good), Indonesian (Good)."
        ),
        "piper": (
            "Local, low latency."
            "Installs locally automatically on first use.\n"
            "Each voice profile takes a few seconds to download on first use (50-100mb per voice).\n"
            "Primarily an English speaking model. Strong American accent when speaking other languages.\n"
            "Specialized voices for certain languages (see dropdown)"
            # "English US (Excellent), English UK (Excellent), German (Good), Spanish (Good), French (Good), "
            # "Italian (Good), Portuguese (Good), Polish (Good), Czech (Fair), Russian (Fair), Ukrainian (Fair), Dutch (Fair)."
        ),
        "elevenlabs": (
            "Cloud based, medium latency.\n"
            "English only."
        ),
        "elevenlabs_multilingual_v2": (
            "Cloud based, high latency.\n"
            "28 languages: English, Chinese, German, Spanish, French, Italian, Japanese, Korean, Portuguese, "
            "Polish, Hindi, Arabic, Turkish, Dutch, Swedish, Finnish, Czech, Greek, Hebrew, Indonesian, Malay, "
            "Ukrainian, Vietnamese, Romanian, Hungarian, Danish, Norwegian, Russian.\n"
            "Voices have \"native speaker\" quality in their training language."
            # "A voice cloned from an English speaker will have an English accent when speaking other languages."
        ),
    }
    tts_lang_info_label = QLabel()
    tts_lang_info_label.setObjectName("tts_lang_info_label")
    tts_lang_info_label.setWordWrap(True)
    tts_lang_info_label.setStyleSheet("color: #555; font-size: 11px; padding: 6px 0 4px 0;")
    ##increase height of the label
    tts_lang_info_label.setFixedHeight(60)
    
    def _update_tts_lang_info(backend_id):
        bid = (backend_id or "piper").lower()
        text = TTS_LANG_INFO.get(bid, "")
        tts_lang_info_label.setText(text)
        tts_lang_info_label.setVisible(bool(text))
    
    _update_tts_lang_info("piper")
    tts_combo.currentIndexChanged.connect(lambda: _update_tts_lang_info(tts_combo.currentData()))
    ocr_layout.addWidget(tts_lang_info_label)
    
    ocr_layout.addStretch()
    mode_tabs.addTab(ocr_tab, "📝 OCR Mode")
    
    # Audio Tab
    audio_tab = QWidget()
    audio_layout = QVBoxLayout(audio_tab)
    audio_layout.setContentsMargins(16, 16, 16, 16)
    
    # Audio device selection
    audio_info = QLabel("Transcribe and translate system audio in real-time.\n\n"
                        "⚠️ For system audio: Install BlackHole (brew install blackhole-2ch),\n"
                        "then set up Multi-Output Device in Audio MIDI Setup to include BlackHole.")
    audio_info.setWordWrap(True)
    audio_info.setStyleSheet("color: #666; font-size: 12px;")
    audio_layout.addWidget(audio_info)
    
    # Audio device selection
    import sounddevice as sd
    device_label = QLabel("Audio Input Device:")
    device_combo = QComboBox()
    devices = sd.query_devices()
    print("[DEBUG] Available audio devices:")
    default_input = sd.default.device[0]
    print(f"[DEBUG] Default input device: {default_input}")
    
    # Show ALL devices (input and virtual) - filter for those with >0 input channels
    available_devices = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            label = f"{dev['name']} (ID: {i}, {dev['max_input_channels']}ch)"
            print(f"  [{i}] {label}")
            device_combo.addItem(label, i)
            available_devices.append((i, dev['name']))
    
    # Auto-select BlackHole if available
    blackhole_id = None
    for idx, name in available_devices:
        if "blackhole" in name.lower():
            blackhole_id = idx
            break
    
    if blackhole_id is not None:
        device_combo.setCurrentIndex(device_combo.findData(blackhole_id))
        print(f"[DEBUG] Auto-selected BlackHole device: {blackhole_id}")
    elif default_input >= 0:
        device_combo.setCurrentIndex(device_combo.findData(default_input))
    audio_layout.addWidget(device_label)
    audio_layout.addWidget(device_combo)
    
    # ASR Backend selection
    asr_label = QLabel("ASR Backend:")
    asr_combo = QComboBox()
    asr_combo.addItems([
        "openai (Whisper API, OpenAI key required)",
        "whisper (local, fast, good accuracy)", 
        "funasr (local, Chinese optimized; first load takes time)", 
        "mlx (local, Apple Silicon acceleration)"
    ])
    audio_layout.addWidget(asr_label)
    audio_layout.addWidget(asr_combo)
    
    # FunASR Model selection (only shown when FunASR backend is selected)
    funasr_model_label = QLabel("FunASR Model:")
    funasr_model_combo = QComboBox()
    funasr_model_combo.setEditable(True)
    funasr_model_combo.setToolTip("FunASR model (must include namespace: iic/ or FunAudioLLM/)")
    funasr_model_combo.addItems([
        "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online",
        "iic/speech_UniASR_asr_2pass-en-16k-common-vocab1080-tensorflow1-online",
        "iic/SenseVoiceSmall",
        "FunAudioLLM/SenseVoiceSmall",
        "FunAudioLLM/Fun-ASR-Nano-2512",
    ])
    try:
        from audio_config import config
        funasr_model_combo.setCurrentText(config.funasr_model)
    except Exception:
        pass
    funasr_model_label.setVisible(False)
    funasr_model_combo.setVisible(False)
    audio_layout.addWidget(funasr_model_label)
    audio_layout.addWidget(funasr_model_combo)
    
    def _update_funasr_visibility():
        is_funasr = asr_combo.currentText().strip().lower().startswith("funasr")
        funasr_model_label.setVisible(is_funasr)
        funasr_model_combo.setVisible(is_funasr)
    asr_combo.currentTextChanged.connect(_update_funasr_visibility)
    _update_funasr_visibility()
    
    audio_layout.addStretch()
    mode_tabs.addTab(audio_tab, "🎙️ Audio Mode")
    
    layout.addWidget(mode_tabs)

    # Learn mode checkbox - only show for Chinese input
    learn_cb = QCheckBox("Learn mode (Keywords, Pinyin, Definitions)")
    learn_cb.setChecked(False)
    learn_cb.setVisible(False)
    layout.addWidget(learn_cb)

    # Learn mode translation provider selection
    learn_provider_label = QLabel("Learn Mode Translation:")
    learn_provider_combo = QComboBox()
    learn_model_combo = QComboBox()
    learn_provider_label.setVisible(False)
    learn_provider_combo.setVisible(False)
    learn_model_combo.setVisible(False)
    
    def get_available_providers():
        """Get list of providers that have API keys available."""
        available = []
        key_map = {
            "siliconflow_com": "SILICONFLOW_COM_API_KEY",
            "siliconflow_cn": "SILICONFLOW_CN_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "together": "TOGETHER_API_KEY",
            "huggingface_api": "HF_API_KEY",
            "huggingface_local": None,  # No key needed
            "deepl": "DEEPL_AUTH_KEY",
            "google": "GOOGLE_TRANSLATE_API_KEY",
            "baidu": ("BAIDU_APP_ID", "BAIDU_APP_SECRET"),
            "youdao": ("YOUDAO_APP_KEY", "YOUDAO_APP_SECRET"),
            "yandex": "YANDEX_API_KEY",
            "libretranslate": "LIBRETRANSLATE_API_KEY",
            "caiyun": "CAIYUN_TOKEN",
            "niutrans": "NIUTRANS_APIKEY",
        }
        
        # Add local dictionary option (English only)
        # All entries: (provider_id, display_name) for consistency
        available.append(("local_dict", "Local Dictionary (English only)"))
        
        # _LLM_PROVIDERS is (display_name, provider_id) - we need (provider_id, display_name)
        for display_name, prov_id in _LLM_PROVIDERS:
            key_env = key_map.get(prov_id)
            if key_env is None:  # No key needed (e.g., huggingface_local)
                available.append((prov_id, display_name))
            elif isinstance(key_env, tuple):
                # Multiple keys needed
                if all(os.environ.get(k) for k in key_env):
                    available.append((prov_id, display_name))
            elif os.environ.get(key_env):
                available.append((prov_id, display_name))
        
        # Add MT providers
        if os.environ.get("DEEPL_AUTH_KEY"):
            available.append(("deepl", "DeepL"))
        if os.environ.get("GOOGLE_TRANSLATE_API_KEY"):
            available.append(("google", "Google Translate"))
        if os.environ.get("BAIDU_APP_ID") and os.environ.get("BAIDU_APP_SECRET"):
            available.append(("baidu", "Baidu"))
        if os.environ.get("YOUDAO_APP_KEY") and os.environ.get("YOUDAO_APP_SECRET"):
            available.append(("youdao", "Youdao"))
        if os.environ.get("YANDEX_API_KEY"):
            available.append(("yandex", "Yandex"))
        if os.environ.get("CAIYUN_TOKEN"):
            available.append(("caiyun", "Caiyun 彩云小译"))
        if os.environ.get("NIUTRANS_APIKEY"):
            available.append(("niutrans", "Niutrans 小牛翻译"))
        # LibreTranslate can work without API key (public instance)
        available.append(("libretranslate", "LibreTranslate"))
        
        return available
    
    def update_learn_provider_combo():
        """Update learn provider combo with available providers."""
        learn_provider_combo.blockSignals(True)
        learn_provider_combo.clear()
        available = get_available_providers()
        if available:
            for provider_id, provider_name in available:
                # Use setItemData to ensure the data is properly stored
                index = learn_provider_combo.count()
                learn_provider_combo.addItem(provider_name)
                learn_provider_combo.setItemData(index, provider_id)
        else:
            learn_provider_combo.addItem("No API keys configured")
            learn_provider_combo.setItemData(0, None)
        learn_provider_combo.blockSignals(False)
        # Update model combo after provider combo is populated
        if learn_provider_combo.count() > 0:
            update_learn_model_combo()
    
    def update_learn_model_combo():
        """Update learn model combo based on selected provider."""
        learn_model_combo.blockSignals(True)
        learn_model_combo.clear()
        
        if learn_provider_combo.count() == 0:
            learn_model_combo.blockSignals(False)
            return
        
        # Get provider ID from current selection
        current_index = learn_provider_combo.currentIndex()
        if current_index < 0:
            learn_model_combo.blockSignals(False)
            return
        
        # Get the data (provider_id) from the combo box item
        provider_id = learn_provider_combo.itemData(current_index)
        
        # If itemData returns None, try to get it from the available providers list
        if provider_id is None:
            available = get_available_providers()
            if current_index < len(available):
                provider_id = available[current_index][0]  # Get the ID from the tuple
        
        # Define MT providers (no models)
        mt_providers = {"deepl", "google", "baidu", "youdao", "yandex", "libretranslate", "caiyun", "niutrans"}
        
        # Local dictionary doesn't need a model
        if provider_id == "local_dict":
            learn_model_combo.addItem("N/A", None)
        elif provider_id in mt_providers:
            # MT providers don't have models
            learn_model_combo.addItem("N/A", None)
        elif provider_id and provider_id in _LLM_MODELS:
            models = _LLM_MODELS[provider_id]
            if models and len(models) > 0:
                for disp_name, model_name in models:
                    learn_model_combo.addItem(disp_name, model_name)
            else:
                learn_model_combo.addItem("N/A", None)
        else:
            # Unknown provider or no models available
            learn_model_combo.addItem("N/A", None)
        learn_model_combo.blockSignals(False)
    
    def on_learn_provider_changed():
        update_learn_model_combo()
        update_learn_visibility()  # Update visibility when provider changes
    
    learn_provider_combo.currentIndexChanged.connect(on_learn_provider_changed)
    update_learn_provider_combo()  # Initial population
    
    learn_provider_row = QHBoxLayout()
    learn_provider_row.addWidget(learn_provider_label)
    learn_provider_row.addWidget(learn_provider_combo, 1)
    learn_model_label = QLabel("Model:")
    learn_model_label.setVisible(False)
    learn_provider_row.addWidget(learn_model_label)
    learn_provider_row.addWidget(learn_model_combo, 1)
    layout.addLayout(learn_provider_row)

    # Check if From language is Chinese
    def update_learn_visibility():
        from_idx = from_sel.get_index()
        from_code = _LANG_OPTIONS[from_idx][1]
        to_idx = to_sel.get_index()
        to_code = _LANG_OPTIONS_TARGET[to_idx][1]
        is_chinese = (from_code == "zh")
        is_english = (to_code == "en")
        learn_cb.setVisible(is_chinese)
        learn_provider_label.setVisible(is_chinese and learn_cb.isChecked())
        learn_provider_combo.setVisible(is_chinese and learn_cb.isChecked())
        
        # Show model label/combo for LLM providers; hide only for local_dict and MT providers
        provider_id = None
        if learn_provider_combo.count() > 0 and learn_provider_combo.currentIndex() >= 0:
            current_index = learn_provider_combo.currentIndex()
            provider_id = learn_provider_combo.itemData(current_index)
            if provider_id is None:
                available = get_available_providers()
                if current_index < len(available):
                    provider_id = available[current_index][0]
        
        mt_providers = {"deepl", "google", "baidu", "youdao", "yandex", "libretranslate"}
        # Hide model dropdown only when we KNOW it's local_dict or MT
        hide_model = provider_id == "local_dict" or provider_id in mt_providers
        show_model = is_chinese and learn_cb.isChecked() and not hide_model
        learn_model_label.setVisible(show_model)
        learn_model_combo.setVisible(show_model)
        if not is_chinese:
            learn_cb.setChecked(False)
    
    def on_learn_cb_toggled(checked):
        update_learn_visibility()
    
    learn_cb.toggled.connect(on_learn_cb_toggled)

    # Initial check and connect to changes
    update_learn_visibility()
    from_sel.selection_changed.connect(update_learn_visibility)

    btn_row = QHBoxLayout()
    api_btn = QPushButton("API Keys...")
    api_btn.clicked.connect(lambda: show_api_keys_dialog(dlg))
    btn_row.addWidget(api_btn)
    settings_btn = QPushButton("Settings...")
    settings_btn.clicked.connect(lambda: show_settings_dialog(dlg))
    btn_row.addWidget(settings_btn)
    layout.addLayout(btn_row)

    btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)
    layout.addWidget(btns)

    dlg.installEventFilter(_DialogRaiseFilter(dlg))
    if dlg.exec_() != QDialog.Accepted:
        return None, None, False, None, None, False, None, None, "ocr", None, "whisper", "vision"
    use_large = large_rb.isChecked()
    llm_provider = _LLM_PROVIDERS[llm_sel.get_index()][1] if use_large else None
    llm_model = None
    if use_large and llm_provider:
        models = _LLM_MODELS.get(llm_provider, _LLM_MODELS["siliconflow_com"])
        idx = model_combo.currentIndex()
        if 0 <= idx < len(models):
            llm_model = models[idx][1]
    from_lang = _LANG_OPTIONS[from_sel.get_index()][1]
    to_lang = _LANG_OPTIONS_TARGET[to_sel.get_index()][1]
    # Learn mode: only enable if From language is Chinese AND checkbox is checked
    is_chinese_source = (from_lang == "zh")
    learn_mode = is_chinese_source and learn_cb.isChecked()
    
    # Learn mode translation provider/model
    learn_mode_provider = None
    learn_mode_model = None
    if learn_mode:
        # Get provider ID from combo box
        provider_index = learn_provider_combo.currentIndex()
        if provider_index >= 0:
            learn_mode_provider = learn_provider_combo.itemData(provider_index)
            if learn_mode_provider is None:
                # Fallback: get from available providers list
                available = get_available_providers()
                if provider_index < len(available):
                    learn_mode_provider = available[provider_index][0]
        
        # Get model from combo box
        model_index = learn_model_combo.currentIndex()
        if model_index >= 0:
            learn_mode_model = learn_model_combo.itemData(model_index)
            if learn_mode_model is None and learn_model_combo.currentText() != "N/A":
                # Fallback: get model name from display text if data is None
                # This shouldn't happen, but handle it just in case
                pass
    
    # Transcription mode: OCR or Audio
    transcription_mode = "audio" if mode_tabs.currentIndex() == 1 else "ocr"
    
    # Audio settings (only for audio mode)
    audio_device_index = device_combo.currentData() if transcription_mode == "audio" else None
    audio_asr_backend = asr_combo.currentText().split()[0] if transcription_mode == "audio" else "whisper"
    # Get FunASR model if FunASR backend is selected (from language dialog's combo)
    audio_funasr_model = None
    if transcription_mode == "audio" and audio_asr_backend == "funasr":
        audio_funasr_model = funasr_model_combo.currentText().strip()
        if not audio_funasr_model:
            from audio_config import config
            audio_funasr_model = config.funasr_model
    
    # OCR backend and TTS backend/voice/speed selection (from OCR tab)
    ocr_backend = "vision"  # Default
    tts_backend = "piper"  # Default
    tts_voice = None
    tts_speed = 1.2
    if transcription_mode == "ocr":
        ocr_tab_widgets = ocr_tab.findChildren(QComboBox)
        for widget in ocr_tab_widgets:
            if widget.objectName() == "ocr_backend_combo":
                ocr_backend = widget.currentData() if widget.currentData() else "vision"
            if widget.objectName() == "tts_backend_combo":
                tts_backend = widget.currentData() if widget.currentData() else "piper"
            if widget.objectName() == "tts_voice_combo":
                tts_voice = widget.currentData() if widget.currentData() else None
        speed_spin = ocr_tab.findChild(QDoubleSpinBox, "tts_speed_spin")
        if speed_spin:
            tts_speed = speed_spin.value()
    
    print(f"[Language Dialog] Returning: from={from_lang}, to={to_lang}, learn_mode={learn_mode}, learn_provider={learn_mode_provider}, learn_model={learn_mode_model}, transcription_mode={transcription_mode}, audio_device={audio_device_index}, asr_backend={audio_asr_backend}, funasr_model={audio_funasr_model}, ocr_backend={ocr_backend}, tts_backend={tts_backend}, tts_voice={tts_voice}, tts_speed={tts_speed}")
    return (
        from_lang,
        to_lang,
        use_large,
        llm_provider,
        llm_model,
        learn_mode,
        learn_mode_provider,
        learn_mode_model,
        transcription_mode,
        audio_device_index,
        audio_asr_backend,
        audio_funasr_model,
        ocr_backend,
        tts_backend,
        tts_voice,
        tts_speed,
    )


# --- Region selector: draggable frame (NO fullscreen - you see your video) ---


class RegionSelector(QWidget):
    """Draggable frame. Red when selecting; white/semi-transparent when active. Stays visible for repositioning."""
    finished = pyqtSignal(object)
    region_changed = pyqtSignal(object)

    def __init__(self, screen_w, screen_h):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        w, h = 800, 120
        self._inner_w, self._inner_h = w, h
        p = self._padding()
        cx = max(0, (screen_w - w) // 2)
        cy = screen_h - h - 80
        self.setGeometry(cx - p, cy - p, w + 2 * p, h + 2 * p)
        self.region = None
        self._active = False
        self._needs_reconfirm = False
        self._drag_start = None
        self._resize_corner = None

        self._confirm_label = QLabel("Press Enter to confirm", self)
        self._confirm_label.setAlignment(Qt.AlignCenter)
        self._confirm_label.setFont(QFont("Arial", 11))
        self._confirm_label.setStyleSheet("color: white; background: transparent;")
        self._confirm_label.setAttribute(Qt.WA_TranslucentBackground)
        self._update_confirm_label()

    def _update_confirm_label(self):
        r = self._inner_rect()
        self._confirm_label.setGeometry(r)
        show = not self._active or self._needs_reconfirm
        self._confirm_label.setVisible(show)

    def _padding(self):
        """Buffer around inner rect for grabbing; scales with box size."""
        m = min(self._inner_w, self._inner_h)
        return max(24, min(60, int(m * 0.12)))

    def _resize_zone_size(self):
        """Resize cursor only within 5-10 px of the edge lines; shrink to 2 px for tiny boxes."""
        m = min(self._inner_w, self._inner_h)
        return 2 if m < 80 else 8

    def _inner_rect(self):
        p = self._padding()
        return QRect(p, p, self._inner_w, self._inner_h)

    def _screen_pos(self):
        p = self._padding()
        tl = self.mapToGlobal(self.rect().topLeft())
        return tl + QPoint(p, p)

    def _get_corner(self, pos):
        m = self._resize_zone_size()
        r = self._inner_rect()
        x, y = pos.x(), pos.y()
        if x < r.left() + m and y < r.top() + m:
            return "nw"
        if x > r.right() - m and y < r.top() + m:
            return "ne"
        if x < r.left() + m and y > r.bottom() - m:
            return "sw"
        if x > r.right() - m and y > r.bottom() - m:
            return "se"
        if y < r.top() + m:
            return "n"
        if y > r.bottom() - m:
            return "s"
        if x < r.left() + m:
            return "w"
        if x > r.right() - m:
            return "e"
        return None

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            c = self._get_corner(e.pos())
            if c:
                self._resize_corner = c
            else:
                self._drag_start = (e.globalPos(), self.frameGeometry().topLeft())

    def mouseMoveEvent(self, e):
        cursors = {"nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor, "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor, "n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor, "e": Qt.SizeHorCursor, "w": Qt.SizeHorCursor}
        p = self._padding()
        if self._resize_corner:
            g = e.globalPos()
            geom = self.geometry()
            c = self._resize_corner
            min_inner = 80
            if "e" in c:
                geom.setRight(max(geom.left() + min_inner + 2 * p, g.x()))
            if "w" in c:
                geom.setLeft(min(geom.right() - min_inner - 2 * p, g.x()))
            if "s" in c:
                geom.setBottom(max(geom.top() + 40 + 2 * p, g.y()))
            if "n" in c:
                geom.setTop(min(geom.bottom() - 40 - 2 * p, g.y()))
            self.setGeometry(geom)
            self._inner_w = max(min_inner, self.width() - 2 * p)
            self._inner_h = max(40, self.height() - 2 * p)
            self._update_confirm_label()
        elif self._drag_start:
            self.setCursor(Qt.ClosedHandCursor)
            delta = e.globalPos() - self._drag_start[0]
            self.move(self._drag_start[1] + delta)
        else:
            c = self._get_corner(e.pos())
            self.setCursor(cursors.get(c, Qt.OpenHandCursor) if c else Qt.OpenHandCursor)

    def _emit_region(self):
        p = self._screen_pos()
        self.region = {"left": p.x(), "top": p.y(), "width": self._inner_w, "height": self._inner_h}
        self.region_changed.emit(self.region)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            if self._resize_corner:
                # Reapply padding for new inner size so buffer scales
                p = self._padding()
                tl = self.mapToGlobal(self.rect().topLeft())
                self.setGeometry(tl.x(), tl.y(), self._inner_w + 2 * p, self._inner_h + 2 * p)
            if self._active and (self._drag_start or self._resize_corner):
                self._emit_region()
                self._needs_reconfirm = True
                self._update_confirm_label()
                self.update()
            self._drag_start = None
            self._resize_corner = None
            cursors = {"nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor, "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor, "n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor, "e": Qt.SizeHorCursor, "w": Qt.SizeHorCursor}
            c = self._get_corner(e.pos())
            self.setCursor(cursors.get(c, Qt.OpenHandCursor) if c else Qt.OpenHandCursor)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            if not self._active:
                self._emit_region()
                self._active = True
                self._update_confirm_label()
                self.update()  # repaint red → white before event loop continues
                self.finished.emit(self.region)
            elif self._needs_reconfirm:
                self._needs_reconfirm = False
                self._emit_region()
                self._update_confirm_label()
                self.update()  # repaint immediately so red → white transition happens at once
        elif e.key() == Qt.Key_Escape:
            qa = getattr(self, "_quit_all", None)
            if qa:
                qa()
            else:
                self.close()

    def get_region(self):
        p = self._screen_pos()
        r = {"left": p.x(), "top": p.y(), "width": self._inner_w, "height": self._inner_h}
        if hasattr(self, "_region_ref"):
            self._region_ref.update(r)
        return r

    def closeEvent(self, e):
        if self.region is None and not self._active:
            self.finished.emit(None)
        e.accept()

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        r = self._inner_rect()
        ocr_paused = bool(getattr(self, "_translator_app", None) and getattr(self._translator_app, "_ocr_paused", False))
        if self._active and not self._needs_reconfirm and not ocr_paused:
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(QColor(255, 255, 255, 25))
            painter.drawRect(r)
        else:
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(r)


def select_region(app):
    """Show draggable frame selector. Returns (region, selector). Selector stays visible for repositioning."""
    result = [None]
    selector_ref = [None]
    loop = QEventLoop()
    screen = app.primaryScreen().geometry()
    sw, sh = screen.width(), screen.height()
    selector = RegionSelector(sw, sh)

    def on_finished(region):
        result[0] = region
        selector_ref[0] = selector if region else None
        loop.quit()

    selector.finished.connect(on_finished)
    selector.show()
    selector.activateWindow()
    selector.setFocus()
    QTimer.singleShot(100, lambda: _mac_set_fullscreen_overlay(selector))
    loop.exec_()
    return result[0], selector_ref[0]


# --- Side buttons (main menu, settings) ---

def _icon_path(name):
    """Resolve path to img/ SVG icons."""
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "img", f"{name}.svg")


def _icon_path_png(name):
    """Resolve path to img/ PNG icons (play, pause)."""
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "img", f"{name}.png")


class _SideButton(QPushButton):
    """Transparent icon button for the side panel."""
    _BTN_STYLE = """
        QPushButton {
            background: rgba(0, 0, 0, 25);
            border: 0px solid rgba(255, 255, 255, 50);
            border-radius: 8px;
        }
        QPushButton:hover {
            background: rgba(0, 0, 0, 70);
            border-color: rgba(255, 255, 255, 0.85);
        }
    """
    def __init__(self, icon_path, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(44, 44)
        if os.path.exists(icon_path):
            self.setIcon(QIcon(icon_path))
        self.setIconSize(QSize(24, 24))
        self.setStyleSheet(self._BTN_STYLE)


class _SideButtons(QWidget):
    """Two stacked buttons on the right side of the screen: main menu and settings."""
    def __init__(self, screen_w, screen_h, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFixedSize(44, 96)  # 44+8+44
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.menu_btn = _SideButton(_icon_path("menu"), self)
        self.menu_btn.clicked.connect(self._on_menu)
        self.settings_btn = _SideButton(_icon_path("settings"), self)
        self.settings_btn.clicked.connect(self._on_settings)
        layout.addWidget(self.menu_btn)
        layout.addWidget(self.settings_btn)
        x = screen_w - 46
        y = int(screen_h / 2 - 48)
        self.move(x, y)
        self._quit_all = None
        self._translator = None

    def set_callbacks(self, quit_all, translator=None):
        self._quit_all = quit_all
        self._translator = translator

    def _on_menu(self):
        result = show_language_dialog(self)
        if result[0] is not None:
            self._add_status("Restart the app to apply language/model changes.")

    def _on_settings(self):
        mode = getattr(self._translator, "transcription_mode", "ocr") if self._translator else "ocr"
        if show_settings_dialog(self, translator=self._translator, transcription_mode=mode):
            self._add_status("Settings saved.")

    def _add_status(self, msg):
        if self._translator and hasattr(self._translator, "_add_status_message"):
            self._translator._add_status_message(msg, duration_sec=3, is_good_news=True)


# --- Overlay ---


class SubtitleOverlay(QWidget):
    """Overlay that can be vertically resized (expandable at top and bottom edges) and moved by dragging."""

    EDGE_MARGIN = 10  # How close (px) to the edge for resize to activate
    EDGE_RESIZE_MIN_HEIGHT = 60  # Minimum height to allow
    
    def __init__(self, left=400, top=780, width=800, height=160, screen_w=None, below_ocr=True, transcription_mode="ocr"):
        super().__init__()
        self._transcription_mode = transcription_mode

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
        )

        self.setGeometry(left, top, width, height)
        self.setStyleSheet("background: transparent;")
        self.setFocusPolicy(Qt.StrongFocus)

        self._drag_start = None
        self._screen_w = screen_w
        self._region_width = width
        self._below_ocr = below_ocr  # True = overlay below OCR (grow down); False = above (grow up)

        self._resize_dir = None  # track "top", "bottom", or None
        self._resize_origin_geom = None  # (x, y, w, h) at start of resize
        self._resize_start_pos = None  # mouse pos at start of resize

        self.setCursor(Qt.OpenHandCursor)

        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(0, 25, 0, 5)  # 15px top for info pill; 5px bottom for border-radius

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setStyleSheet("color: #ff6b6b; padding: 2px 0;")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.status_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.status_label.hide()
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(0)
        status_row.addSpacing(40)  # Status bar 25px further right
        status_row.addWidget(self.status_label, 1)
        layout.addLayout(status_row)

        self.label = QLabel("Waiting for subtitles... (Esc to quit)")
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.RichText)
        self._update_subtitle_font()
        self.label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 180);
                padding: 10px;
                border-radius: 5px;
            }
        """)
        # Make overlay background transparent to show through
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(0, 0, 0, 0))  # Fully transparent
        self.setPalette(palette)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # Top align so stacked lines show correctly
        # Make label ignore mouse events so they pass through to parent for dragging
        self.label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        # Content row: play/pause (audio only) + label
        content_row = QHBoxLayout()
        content_row.setSpacing(8)
        content_row.setContentsMargins(0, 0, 0, 0)
        if transcription_mode == "audio":
            self._play_pause_container = self._create_play_pause_buttons()
            content_row.addWidget(self._play_pause_container)
        else:
            self._play_pause_container = None
            self._play_btn = None
            self._pause_btn = None
        # Speak button (OCR mode only) - in layout so it gets its own space, not overlapped
        if transcription_mode == "ocr":
            self._speak_container = self._create_speak_button()
            content_row.addWidget(self._speak_container)
        else:
            self._speak_container = None
        content_row.addWidget(self.label, 1)
        layout.addLayout(content_row)
        self.setLayout(layout)
        self._last_display_text = None
        self._status_messages = []
        self._status_bar_height = 0
        # Info pill: tab sticking off top right of subtitle area (does not move with status stack)
        self.info_pill = QLabel()
        self.info_pill.setFont(QFont("Arial", 10))
        self.info_pill.setStyleSheet("""
            color: rgba(255,255,255,0.95);
            background-color: rgba(0, 0, 0, 180);
            padding: 4px 12px 6px;
            border-radius: 6px 6px 6px 6px;
        """)
        self.info_pill.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        # Ensure the pill can extend slightly beyond widget bounds if needed
        self.info_pill.setAttribute(Qt.WA_NoSystemBackground, True)
        self.info_pill.setParent(self)
        self.info_pill.setText("—")
        self.info_pill.adjustSize()
        self.info_pill.show()
    
    def _create_play_pause_buttons(self):
        """Create play and pause buttons stacked vertically for audio mode."""
        container = QWidget(self)
        container_layout = QVBoxLayout()
        container_layout.setSpacing(8)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        play_path = _icon_path_png("play")
        pause_path = _icon_path_png("pause")
        self._play_icon = QIcon(play_path) if os.path.exists(play_path) else QIcon()
        self._pause_icon = QIcon(pause_path) if os.path.exists(pause_path) else QIcon()
        
        # Play button (shown when paused)
        self._play_btn = QPushButton(container)
        self._play_btn.setCursor(Qt.PointingHandCursor)
        self._play_btn.setFixedSize(40, 40)
        self._play_btn.setFocusPolicy(Qt.NoFocus)
        self._play_btn.setIcon(self._play_icon)
        self._play_btn.setIconSize(QSize(20, 20))
        
        # Pause button (shown when running)
        self._pause_btn = QPushButton(container)
        self._pause_btn.setCursor(Qt.PointingHandCursor)
        self._pause_btn.setFixedSize(40, 40)
        self._pause_btn.setFocusPolicy(Qt.NoFocus)
        self._pause_btn.setIcon(self._pause_icon)
        self._pause_btn.setIconSize(QSize(20, 20))
        
        # Shared stylesheet: no border, different states for enabled/disabled
        button_style = """
            QPushButton {
                background-color: rgba(0, 0, 0, 180);
                border: none;
                border-radius: 6px;
                padding: 4px;
            }
            QPushButton[enabled="true"] {
                background-color: rgba(0, 0, 0, 120);
            }
            QPushButton[enabled="true"]:hover {
                background-color: rgba(0, 0, 0, 160);
            }
            QPushButton[enabled="false"] {
                background-color: rgba(0, 0, 0, 180);
            }
        """
        self._play_btn.setStyleSheet(button_style)
        self._pause_btn.setStyleSheet(button_style)
        
        def on_play_click():
            app = getattr(self, "_translator_app", None)
            if not app or getattr(app, "transcription_mode", "") != "audio":
                return
            app._audio_paused = False
            self._update_button_states()
            app._add_status_message("Resumed", duration_sec=2, is_good_news=True)
            print("[Audio] resumed")
        
        def on_pause_click():
            app = getattr(self, "_translator_app", None)
            if not app or getattr(app, "transcription_mode", "") != "audio":
                return
            app._audio_paused = True
            self._update_button_states()
            app._add_status_message("Paused", duration_sec=2, is_good_news=False)
            print("[Audio] paused")
        
        self._play_btn.clicked.connect(on_play_click)
        self._pause_btn.clicked.connect(on_pause_click)
        container_layout.addWidget(self._play_btn)   # Play below
        container_layout.addWidget(self._pause_btn)  # Pause on top (default: running)

        container_layout.addStretch()  # Add stretch to prevent cutoff
        container.setLayout(container_layout)
        container.setFixedHeight(92)  # 40 + 8 spacing + 40 + 4 margin = 92
        
        # Set initial state (running = pause button enabled)
        self._update_button_states()
        
        return container
    
    def _create_speak_button(self):
        """Create Speak button for TTS (OCR mode only). Style matches play/pause."""
        container = QWidget(self)
        container.setFixedSize(30,30)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        speak_path = _icon_path_png("speak")
        self._speak_btn = QPushButton(container)
        self._speak_btn.setCursor(Qt.PointingHandCursor)
        self._speak_btn.setFixedSize(30, 30)
        self._speak_btn.setFocusPolicy(Qt.NoFocus)
        
        if os.path.exists(speak_path):
            self._speak_btn.setIcon(QIcon(speak_path))
            self._speak_btn.setIconSize(QSize(20, 20))
        else:
            self._speak_btn.setText("S")
            self._speak_btn.setFont(QFont("Arial", 12))
        
        def on_speak_click():
            app = getattr(self, "_translator_app", None)
            if not app:
                return
            if getattr(app, "transcription_mode", "") != "ocr":
                return
            app.tts_enabled = not getattr(app, "tts_enabled", False)
            if not app.tts_enabled and hasattr(app, "tts_engine"):
                app.tts_engine.stop()  # Cut off audio immediately, clear queue
            if hasattr(app, "_add_status_message"):
                app._add_status_message("TTS on" if app.tts_enabled else "TTS off", duration_sec=2, is_good_news=app.tts_enabled)
            self._update_speak_button_states()
        
        self._speak_btn.clicked.connect(on_speak_click)
        layout.addWidget(self._speak_btn)
        
        self._speak_container = container
        self._update_speak_button_states()
        
        return container

    def _update_speak_button_states(self):
        """Update Speak button state using background color and border."""
        if not hasattr(self, "_speak_btn") or not self._speak_btn:
            return
        app = getattr(self, "_translator_app", None)
        tts_enabled = getattr(app, "tts_enabled", False) if app else False

        # Clear any graphics effect
        self._speak_btn.setGraphicsEffect(None)
        
        if tts_enabled:
            # TTS ON: darker background with green border
            self._speak_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 0, 0, 255);
                    border: 2px solid rgba(12, 133, 88, 180);
                    border-radius: 6px;
                    padding: 4px;
                }
                QPushButton:hover {
                    background-color: rgba(40, 40, 40, 255);
                }
            """)
        else:
            # TTS OFF: lighter background, no border
            self._speak_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(60, 60, 60, 255);
                    border: none;
                    border-radius: 6px;
                    padding: 4px;
                }
                QPushButton:hover {
                    background-color: rgba(80, 80, 80, 255);
                }
            """)
        
        self._speak_btn.update()

    
    def _update_button_states(self):
        """Update play/pause button enabled states and icon opacity."""
        if not self._play_btn or not self._pause_btn:
            return
        app = getattr(self, "_translator_app", None)
        if not app:
            return
        paused = getattr(app, "_audio_paused", False)
        
        # Disabled button: icon LESS transparent (more opaque/bolder), background lighter but transparent
        # Enabled button: icon MORE transparent, background same as overlay
        if paused:
            # Play button enabled, pause button disabled
            self._play_btn.setProperty("enabled", "true")
            self._pause_btn.setProperty("enabled", "false")
            # Enabled button: overlay background, very transparent icon
            self._play_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 0, 0, 180);
                    border: none;
                    border-radius: 6px;
                    padding: 4px;
                }
                QPushButton:hover {
                    background-color: rgba(0, 0, 0, 160);
                }
            """)
            # Disabled button: lighter background, fully opaque icon
            self._pause_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 0, 0, 120);
                    border: none;
                    border-radius: 6px;
                    padding: 4px;
                }
            """)
            # Use QGraphicsOpacityEffect for icon transparency
            play_opacity = QGraphicsOpacityEffect()
            play_opacity.setOpacity(0.3)  # Enabled: very transparent
            self._play_btn.setGraphicsEffect(play_opacity)
            pause_opacity = QGraphicsOpacityEffect()
            pause_opacity.setOpacity(1.0)  # Disabled: fully opaque (bolder)
            self._pause_btn.setGraphicsEffect(pause_opacity)
        else:
            # Pause button enabled, play button disabled
            self._pause_btn.setProperty("enabled", "true")
            self._play_btn.setProperty("enabled", "false")
            # Enabled button: overlay background, very transparent icon
            self._pause_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 0, 0, 180);
                    border: none;
                    border-radius: 6px;
                    padding: 4px;
                }
                QPushButton:hover {
                    background-color: rgba(0, 0, 0, 160);
                }
            """)
            # Disabled button: lighter background, fully opaque icon
            self._play_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 0, 0, 120);
                    border: none;
                    border-radius: 6px;
                    padding: 4px;
                }
            """)
            pause_opacity = QGraphicsOpacityEffect()
            pause_opacity.setOpacity(0.3)  # Enabled: very transparent
            self._pause_btn.setGraphicsEffect(pause_opacity)
            play_opacity = QGraphicsOpacityEffect()
            play_opacity.setOpacity(1.0)  # Disabled: fully opaque (bolder)
            self._play_btn.setGraphicsEffect(play_opacity)
        
        # Refresh styles
        self._play_btn.style().unpolish(self._play_btn)
        self._play_btn.style().polish(self._play_btn)
        self._pause_btn.style().unpolish(self._pause_btn)
        self._pause_btn.style().polish(self._pause_btn)
    
    def update_play_pause_state(self):
        """Sync play/pause buttons with translator state (for Space/Enter)."""
        self._update_button_states()
        
    def set_status_messages(self, messages):
        """Show error/status messages at top. messages: list of (text, is_good_news) or plain strings (treated as error)."""
        self._status_messages = list(messages) if messages else []
        if self._status_messages:
            parts = []
            for m in self._status_messages:
                if isinstance(m, tuple):
                    text, is_good = m
                    color = "white" if is_good else "#ff6b6b"
                else:
                    text, is_good = m, False
                    color = "#ff6b6b"
                escaped = text.replace("&", "&amp;").replace("<", "&lt;")
                parts.append(f'<span style="color: {color};">{escaped}</span>')
            html = "<br>".join(parts)
            self.status_label.setText(html)
            self.status_label.show()
        else:
            self.status_label.hide()
            self.status_label.clear()
        self.adjust_height_to_content()

    def adjust_height_to_content(self):
        """Recalculate overlay height. Grows downward when below OCR, upward when above OCR."""
        text = self._last_display_text
        if not text:
            metrics = QFontMetrics(self.label.font())
            subtitle_h = max(100, int(metrics.lineSpacing() * 3))
        else:
            subtitle_h = self._content_height_for_text(text, self.width())
        subtitle_h = max(subtitle_h, 100)  # Minimum height before first translations
        status_h = 0
        if self._status_messages:
            metrics = QFontMetrics(self.status_label.font())
            status_h = metrics.lineSpacing() * len(self._status_messages) + 8
        self._status_bar_height = status_h
        # Add bottom margin (5px) to account for border-radius so corners aren't clipped
        bottom_margin = 5
        top_margin = 25  # Space for info pill above content
        new_h = subtitle_h + status_h + bottom_margin + top_margin
        old_h = self.height()
        delta = new_h - old_h
        if delta != 0 and not self._below_ocr:
            # Above OCR: grow upward (move window up so bottom stays fixed)
            self.move(self.x(), self.y() - delta)
        self.setFixedHeight(new_h)
        self.label.setMinimumHeight(subtitle_h)  # Prevent status bar from shrinking translation area
        self.updateGeometry()
        self._update_info_pill_pos()

    def _update_subtitle_font(self):
        """Larger text when OCR box > half screen width, normal otherwise."""
        if self._screen_w and self._region_width > self._screen_w // 2:
            size = 22
        else:
            size = 16
        self.label.setFont(QFont("Arial", size))

    def set_region_size(self, region_width, screen_w=None):
        """Update font size when OCR region changes."""
        self._region_width = region_width
        if screen_w is not None:
            self._screen_w = screen_w
        self._update_subtitle_font()
        self.adjust_height_to_content()

    def _update_info_pill_pos(self):
        """Position pill slightly above the main subtitle label (always in the 'above' position)."""
        if hasattr(self, "info_pill") and self.info_pill:
            margin = 4
            status_h = getattr(self, "_status_bar_height", 0)
            layout_spacing = 4  # Matches layout.setSpacing(4)
            top_margin = 25  # Matches layout top margin
            # Label starts after top margin + status bar + spacing
            label_start_y = top_margin + status_h + layout_spacing
            # Position pill 15px above the label (same position as when status bar has messages)
            pill_offset_above = 25
            pill_y = label_start_y - pill_offset_above
            self.info_pill.move(self.width() - self.info_pill.width() - margin - 10, pill_y)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_info_pill_pos()

    def set_info_pill_text(self, word_count_by_model):
        """Update info pill: per-model word count stack (vertical). word_count_by_model: {model: count}."""
        if hasattr(self, "info_pill") and self.info_pill:
            if not word_count_by_model:
                text = "0 words"
            else:
                lines = [f"{name} · {cnt:,} words" for name, cnt in word_count_by_model.items()]
                total = sum(word_count_by_model.values())
                # lines.append(f"{total:,} total")
                text = "\n".join(lines)
            self.info_pill.setText(text)
            self.info_pill.adjustSize()
            self._update_info_pill_pos()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            # Forward clicks on Speak button area - overlay may receive them when
            # label/pill pass through, so we must explicitly deliver to the button
            speak_btn = getattr(self, "_speak_btn", None)
            speak_container = getattr(self, "_speak_container", None)
            if speak_btn and speak_container:
                local_in_container = speak_container.mapFrom(self, e.pos())
                if speak_container.rect().contains(local_in_container):
                    from PyQt5.QtWidgets import QApplication
                    from PyQt5.QtCore import QEvent
                    from PyQt5.QtGui import QMouseEvent
                    local_in_btn = speak_btn.mapFrom(self, e.pos())
                    ev = QMouseEvent(
                        QEvent.MouseButtonPress,
                        local_in_btn,
                        e.globalPos(),
                        e.button(),
                        e.buttons(),
                        e.modifiers(),
                    )
                    QApplication.sendEvent(speak_btn, ev)
                    return
            self._drag_start = (e.globalPos(), self.frameGeometry().topLeft())

    def mouseMoveEvent(self, e):
        if self._drag_start:
            delta = e.globalPos() - self._drag_start[0]
            self.move(self._drag_start[1] + delta)
            self.setCursor(Qt.ClosedHandCursor)

    def snap_away_from_ocr(self, region, gap=10):
        """Animate overlay above or below OCR region (magnetic snap). Target chosen by overlay center vs region center."""
        if not region or getattr(self, "_snap_animating", False):
            return
        try:
            oy = self.y()
            oh = self.height()
            overlay_center_y = oy + oh // 2
            region_center_y = region["top"] + region["height"] // 2
            region_top = region["top"]
            region_bottom = region["top"] + region["height"]
            if overlay_center_y < region_center_y:
                target_y = region_top - oh - gap  # Snap above
            else:
                target_y = region_bottom + gap  # Snap below
            if abs(oy - target_y) < 5:
                return
            start_y = oy
            steps = 12
            step_ms = 16
            step_idx = [0]

            def step():
                step_idx[0] += 1
                t = min(1.0, step_idx[0] / steps)
                eased = 1.0 - (1.0 - t) ** 3
                new_y = int(start_y + (target_y - start_y) * eased)
                self.move(self.x(), new_y)
                if step_idx[0] < steps:
                    QTimer.singleShot(step_ms, step)
                else:
                    self._snap_animating = False
                    app = getattr(self, "_translator_app", None)
                    if app:
                        app._snap_animating = False

            self._snap_animating = True
            app = getattr(self, "_translator_app", None)
            if app:
                app._snap_animating = True
            QTimer.singleShot(step_ms, step)
        except Exception:
            self._snap_animating = False

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            # Forward release to Speak button if we forwarded the press
            speak_btn = getattr(self, "_speak_btn", None)
            speak_container = getattr(self, "_speak_container", None)
            if speak_btn and speak_container:
                local_in_container = speak_container.mapFrom(self, e.pos())
                if speak_container.rect().contains(local_in_container):
                    from PyQt5.QtWidgets import QApplication
                    from PyQt5.QtCore import QEvent
                    from PyQt5.QtGui import QMouseEvent
                    local_in_btn = speak_btn.mapFrom(self, e.pos())
                    ev = QMouseEvent(
                        QEvent.MouseButtonRelease,
                        local_in_btn,
                        e.globalPos(),
                        e.button(),
                        e.buttons(),
                        e.modifiers(),
                    )
                    QApplication.sendEvent(speak_btn, ev)
                    return
            self._drag_start = None
            self.setCursor(Qt.OpenHandCursor)

    def moveEvent(self, e):
        super().moveEvent(e)
        if getattr(self, "_hint", None):
            h = self._hint
            h.move(self.x() + self.width() - h.width() - 70, self.y() - h.height() + 10)

    def hideEvent(self, e):
        super().hideEvent(e)
        if getattr(self, "_hint", None):
            self._hint.hide()


    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            qa = getattr(self, "_quit_all", None)
            if qa:
                qa()
            else:
                self.close()

    def _content_height_for_text(self, text, width):
        """Height = sum(stack item heights) + half line height between each pair. Recomputed at each stack update."""
        if not text or not text.strip():
            return 80
        # Match actual label text width: overlay - (speak 30 or play 40) - spacing 8 - label padding 20
        fixed = 40 if self._transcription_mode == "audio" else 30
        available_width = max(100, width - fixed - 8 - 20)
        parts = [p.strip() for p in text.replace("\n\n", "\n").split("\n") if p.strip()]
        if not parts:
            return 80
        metrics = QFontMetrics(self.label.font())
        half_line = metrics.lineSpacing() / 2
        doc = QTextDocument()
        doc.setDefaultFont(self.label.font())
        doc.setTextWidth(available_width)
        content_height = 0
        for i, part in enumerate(parts):
            doc.setPlainText(part)
            content_height += doc.size().height()
            if i < len(parts) - 1:
                content_height += half_line
        return int(content_height * 1.15 + 10)  

    def update_text(self, text, allow_show=True, partial_text=None):
        """Update text. allow_show=False keeps overlay hidden during brief capture hide.
        partial_text: Last/bottom item to style with muted color.
        """
        if not text:
            self.label.setText("Waiting for subtitles...")
            if allow_show:
                self.setVisible(False)
            return
        
        # Use <br> for line breaks in RichText mode
        def to_html(s):
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        
        # Line-height 1.1 within lines; half-line gap between the two sentences via margin-bottom
        line_style = 'style="line-height: 1.1"'
        block1_style = 'style="line-height: 1.1; margin-bottom: 0em"'
        
        # When partial_text is set, it's the incomplete (bottom) item - style it muted
        if partial_text and ("\n" in text or "\n\n" in text):
            parts = text.replace("\n\n", "\n").split("\n")
            if parts and parts[-1].strip():
                last_part = parts[-1].strip()
                last_html = to_html(last_part)
                complete_parts = parts[:-1]
                if complete_parts:
                    styled = f'<p {block1_style}>' + '</p><p ' + block1_style + '>'.join(to_html(p) for p in complete_parts) + '</p>'
                    styled += f'<p {line_style}><span style="color: rgba(255,255,255,0.55); font-style: italic;">{last_html}</span></p>'
                else:
                    styled = f'<p {line_style}><span style="color: rgba(255,255,255,0.55); font-style: italic;">{last_html}</span></p>'
                self.label.setText(styled)
            else:
                self.label.setText(f'<p {line_style}>{to_html(text)}</p>')
        elif partial_text:
            self.label.setText(f'<p {line_style}><span style="color: rgba(255,255,255,0.55); font-style: italic;">{to_html(text.strip())}</span></p>')
        else:
            if "\n" in text:
                parts = text.replace("\n\n", "\n").split("\n")
                if len(parts) == 2:
                    html = f'<p {block1_style}>{to_html(parts[0])}</p><p {line_style}>{to_html(parts[1])}</p>'
                else:
                    html = f'<p {line_style}>' + '<br>'.join(to_html(p) for p in parts) + '</p>'
            else:
                html = f'<p {line_style}>{to_html(text)}</p>'
            self.label.setText(html)
        
        self._last_display_text = text
        # Recompute height when content changes (adapts lines1+lines2) or when width changed
        self.adjust_height_to_content()
        if allow_show:
            self.setVisible(bool(text))


# --- Main app ---


# Human-readable language names for LLM prompts
_LANG_NAMES = {code: lbl for lbl, code in _LANG_OPTIONS}
_LANG_NAMES["auto"] = "the detected language"


class TranslatorApp:
    def __init__(self, region, overlay, debug=False, region_selector=None, source_lang="auto", target_lang="en",
                 use_large_model=False, llm_provider=None, llm_model=None, learn_mode=False, learn_mode_provider=None, learn_mode_model=None, detect_mixed_content=False,
                 max_words_for_translation=0, max_words_enabled=False,                  allow_overlap=False, auto_detect_text_region=False, session_output_enabled=False, session_output_path="", transcription_mode="ocr", audio_device_index=None, audio_asr_backend="whisper", audio_funasr_model=None, ocr_backend="vision", tts_backend="piper", tts_voice=None, tts_speed=1.2):
        self.region = region
        self.overlay = overlay
        self.region_selector = region_selector
        self.debug = debug
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.use_large_model = bool(use_large_model)
        self.llm_provider = llm_provider or "siliconflow_com"
        self.llm_model = llm_model
        self.learn_mode = bool(learn_mode)
        self.learn_mode_provider = learn_mode_provider
        self.learn_mode_model = learn_mode_model
        if self.debug and self.learn_mode:
            print(f"[TranslatorApp] Learn mode enabled: provider={self.learn_mode_provider}, model={self.learn_mode_model}")
        self.detect_mixed_content = bool(detect_mixed_content)
        self.max_words_enabled = bool(max_words_enabled)
        self.max_words_for_translation = max(1, int(max_words_for_translation))
        self.allow_overlap = bool(allow_overlap)
        self.auto_detect_text_region = bool(auto_detect_text_region)
        self.session_output_enabled = bool(session_output_enabled)
        self.session_output_path = (session_output_path or "").strip()
        # Ensure transcription_mode is valid
        if transcription_mode not in ("ocr", "audio"):
            print(f"[TranslatorApp] WARNING: Invalid transcription_mode '{transcription_mode}', defaulting to 'ocr'")
            transcription_mode = "ocr"
        self.transcription_mode = transcription_mode
        # OCR backend selection
        self.ocr_backend = ocr_backend or "vision"
        # Initialize status messages before TTS engine (which uses status_callback)
        self._status_messages = []  # [(text, expiry_time, is_good_news)] - shown at top of overlay
        # TTS (OCR mode only) - use selected backend, voice, speed from dialog
        from tts_engine import create_tts_engine
        self.tts_engine = create_tts_engine(
            tts_backend or "piper", voice_id=tts_voice, speed=tts_speed or 1.2,
            status_callback=lambda msg, duration_sec=8, is_good_news=False: self._add_status_message(msg, duration_sec, is_good_news)
        )
        if transcription_mode == "ocr":
            print(f"[TTS] Using backend: {tts_backend or 'piper'}, voice: {tts_voice or 'default'}, speed: {tts_speed or 1.2}")
        self.tts_enabled = False
        # Audio mode: mutable dict for real-time tuning from Settings > Audio tab
        if transcription_mode == "audio":
            settings = get_app_settings()
            self.audio_buffer_settings = {
                "reconciler_period_sec": settings.get("audio_reconciler_period_sec", 2.0),
                "reconciler_num_checks": settings.get("audio_reconciler_num_checks", 4),
                "reconciler_min_words": settings.get("audio_reconciler_min_words", 7),
                "silence_duration": settings.get("audio_silence_duration", 1.0),
                "max_phrase_duration": settings.get("audio_max_phrase_duration", 5.0),
            }
        else:
            self.audio_buffer_settings = {}
        self._session_output_buffer = []
        self._session_output_path = None  # Set on first flush
        if self.debug:
            print(f"[TranslatorApp] Initialized with learn_mode={self.learn_mode}, source_lang={source_lang}, target_lang={target_lang}, transcription_mode={self.transcription_mode}")

        self.capture_queue = queue.Queue(maxsize=1)
        qsize = 20 if transcription_mode == "audio" else 5  # Audio needs larger buffer for translation latency
        self.text_queue = queue.Queue(maxsize=qsize)
        self.translated_queue = queue.Queue(maxsize=qsize)  # Stores (translated_text, is_final, original_length) tuples
        self.keyword_queue = queue.Queue(maxsize=3)  # for learn mode

        def _put_text_queue(item):
            """Put item to text queue."""
            self.text_queue.put_nowait(item)
        self._put_text_queue = _put_text_queue

        self.last_hash = None
        self.last_text = None
        self.running = True
        self._display_stack = []
        self._last_translation_time = 0
        self._stack_window_sec = 3.0
        self._last_ocr_time = 0
        self._translation_cache = {}  # source_text -> translated
        # Reconcilers: MT uses segment-by-segment stability; LLM uses accumulate-and-split-on-sentences
        # Get settings from environment or use defaults
        settings = get_app_settings()
        mt_stability = settings.get("ocr_mt_reconciler_stability", 0.2)
        llm_stability = settings.get("ocr_llm_reconciler_stability", 0.12)
        llm_max_buffer = settings.get("ocr_llm_reconciler_max_buffer", 0.6)
        self.ocr_min_words_before_translate = settings.get("ocr_min_words_before_translate", 0)
        self.ocr_similarity_substring_chars = max(0, settings.get("ocr_similarity_substring_chars", 15))
        self.llm_context_count = max(0, settings.get("llm_context_count", 3))
        try:
            from streaming_reconciler import StreamingReconciler, LLMReconciler, AudioReconciler
            self.reconciler = StreamingReconciler(stability_threshold=mt_stability, debug=debug)
            self.llm_reconciler = LLMReconciler(stability_threshold=llm_stability, max_buffer_time=llm_max_buffer, debug=debug)
            if debug:
                print("[TranslatorApp] Streaming reconciler (MT) and LLM reconciler initialized")
        except ImportError as e:
            # Fallback if module not found
            if debug:
                print(f"[Warning] streaming_reconciler not found ({e}), using immediate translation")
            self.reconciler = None
            self.llm_reconciler = None
        except Exception as e:
            if debug:
                print(f"[Warning] Failed to initialize reconciler ({e}), using immediate translation")
            import traceback
            traceback.print_exc()
            self.reconciler = None
            self.llm_reconciler = None
        
        self._translation_fail_warned = False
        self._recent_translations = []  # [(text, timestamp)] for dedup beyond stack
        self._recent_sources = []  # [(source_text, timestamp)] - skip translating if new source similar to any
        self._last_llm_text_sent = None
        self._last_llm_send_time = 0.0
        self._llm_context_sources = []  # List of (source_text, translated_text) tuples for context
        try:
            g = overlay.frameGeometry()
            self._overlay_rect = (g.x(), g.y(), g.width(), g.height())
        except Exception:
            self._overlay_rect = None
        self._learn_overlay_rect = None
        learn_o = getattr(overlay, "_learn_overlay", None)
        if learn_o:
            try:
                g = learn_o.frameGeometry()
                self._learn_overlay_rect = (g.x(), g.y(), g.width(), g.height())
            except Exception:
                pass
        self._hiding_for_capture = False
        self._ocr_paused = False
        self._audio_paused = False
        self._audio_model_loading = False
        self._audio_awaiting = False  # True when model loaded but no audio detected yet
        self._ocr_obstructed = False  # True when language mismatch or overlap detected
        self._overlap_paused = False  # True when paused due to overlay overlap (so we resume when overlap ends)
        self._snap_overlay_requested = False  # Magnetic snap when overlap detected
        self._snap_region = None  # OCR region for computing snap target
        self._snap_animating = False
        # Dynamic text region: (y_min, y_max) in frame pixels, learned from first few OCR readings
        self._text_region = None
        self._text_region_readings = 0
        self._text_region_min_y = []
        self._text_region_max_y = []
        # Mixed content detection (temporal coherence): per-band change tracking
        self._mc_bands = 5
        self._mc_band_change_counts = [0] * 5
        self._mc_total_frames = 0
        self._mc_prev_frame = None
        self._mc_min_frames = 25
        self._mc_threshold = 0.6  # max_change_ratio - min_change_ratio > this => mixed content (stricter to avoid video background)
        # Note: _status_messages initialized earlier (before TTS engine creation)
        
        # Translation state
        self._using_mt_fallback = False  # True when LLM failed and we switched to MT; reconnect thread checks LLM
        self._mt_fallback_message_shown = False  # Show "switching to MT" only once until LLM restored
        self._session_word_count_by_model = {}  # {model_name: count} per-model word count
        self._current_display_name = ""  # Display name for info pill
        
        # Audio pipeline (only for audio mode)
        self.audio_pipeline = None
        self.audio_settings = {
            "device_index": audio_device_index,
            "asr_backend": audio_asr_backend
        }
        # Store FunASR model if provided
        if audio_funasr_model:
            self.funasr_model = audio_funasr_model
        if self.transcription_mode == "audio":
            print(f"[TranslatorApp] Audio mode enabled: device={audio_device_index}, backend={audio_asr_backend}")
        self._llm_request_start_time = None
        self._llm_5sec_message_shown = False
        
    def set_audio_settings(self, device_index=None, asr_backend="whisper", silence_threshold=0.005, silence_duration=1.0):
        """Set audio configuration before starting"""
        self.audio_settings = {
            "device_index": device_index,
            "asr_backend": asr_backend,
            "silence_threshold": silence_threshold,
            "silence_duration": silence_duration
        }
        
    def start_audio_pipeline(self):
        """Initialize and start the audio transcription pipeline"""
        if self.transcription_mode != "audio" or not self.audio_settings:
            return
            
        def translate_wrapper(text):
            """Wrapper to use our existing translation method"""
            return self.translate(text)
            
        settings = self.audio_settings
        asr_backend = settings.get("asr_backend", "whisper")
        if asr_backend == "whisper":
            model_size = "base"
        elif asr_backend == "openai":
            model_size = "whisper-1"  # API uses fixed model
        else:
            model_size = settings.get("funasr_model", "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
        self.audio_pipeline = AudioPipeline(
            translator_callback=translate_wrapper,
            device_index=settings.get("device_index"),
            asr_backend=asr_backend,
            model_size=model_size,
            language=self.source_lang if self.source_lang != "auto" else None,
            sample_rate=16000,
            silence_threshold=settings.get("silence_threshold", 0.005),
            silence_duration=settings.get("silence_duration", 1.0),
            status_callback=self._add_status_message
        )
        
        # Connect signals
        self.audio_pipeline.signals.transcription_received.connect(self._on_audio_transcription)
        self.audio_pipeline.signals.translation_received.connect(self._on_audio_translation)
        self.audio_pipeline.signals.error_occurred.connect(self._on_audio_error)
        
        # Start pipeline
        self.audio_pipeline.start()
        print("[TranslatorApp] Audio pipeline started")
        
    def stop_audio_pipeline(self):
        """Stop the audio pipeline if running"""
        if self.audio_pipeline:
            self.audio_pipeline.stop()
            self.audio_pipeline = None
            print("[TranslatorApp] Audio pipeline stopped")
    
    def _on_audio_transcription(self, text):
        """Handle raw audio transcription"""
        if self.debug:
            print(f"[Audio] Transcribed: {text[:100]}...")
        # Add to text queue for processing
        if self.text_queue.full():
            self.text_queue.get_nowait()
        self._put_text_queue((text, True))
        
    def _on_audio_translation(self, translated_text):
        """Handle translated audio text"""
        if self.debug:
            print(f"[Audio] Translated: {translated_text[:100]}...")
        # Add to translation queue for display
        if self.translated_queue.full():
            self.translated_queue.get()
        self.translated_queue.put((translated_text, True, len(translated_text)))
        
    def _on_audio_error(self, error_msg):
        """Handle audio pipeline errors"""
        print(f"[Audio] Error: {error_msg}")
        self._add_status_message(f"Audio error: {error_msg}", duration_sec=10, is_good_news=False)
        # Display name for info pill: model name for LLM, provider for MT
        if use_large_model:
            self._current_display_name = self._model_display_name(llm_model, llm_provider)
        else:
            self._current_display_name = "DeepL"  # Default MT; will update on first translation

    def _add_status_message(self, text, duration_sec=8, is_good_news=False):
        """Add status message (top of overlay). Red for errors, white for good news. Stays for duration_sec."""
        self._status_messages.append((text, time.time() + duration_sec, is_good_news))
        while len(self._status_messages) > 6:
            self._status_messages.pop(0)

    def _get_active_status_messages(self):
        """Return non-expired status messages as [(text, is_good_news), ...]."""
        now = time.time()
        self._status_messages = [(t, exp, good) for t, exp, good in self._status_messages if exp > now]
        return [(t, good) for t, _, good in self._status_messages]

    def _frame_hash(self, img):
        """Simple perceptual hash for change detection."""
        if img is None or img.size == 0:
            return None
        from PIL import Image
        import numpy as np
        pil = Image.fromarray(img).resize((64, 16)).convert("L")
        arr = np.array(pil)
        avg = arr.mean()
        return "".join("1" if p > avg else "0" for p in arr.flatten())

    def has_changed(self, frame):
        if frame is None:
            return False
        h = self._frame_hash(frame)
        if h != self.last_hash:
            self.last_hash = h
            return True
        return False

    def _translate_deepl(self, text):
        """DeepL API - 500K chars/month free. Set DEEPL_AUTH_KEY."""
        key = os.environ.get("DEEPL_AUTH_KEY")
        if not key:
            return None
        row = _LANG_MAP.get(self.source_lang, (None, "auto", "auto", None))
        dl = row[0]
        row = _LANG_MAP.get(self.target_lang, (None, "en", "en", "en"))
        tl = row[0]
        payload = {"text": [text], "target_lang": tl}
        if dl:
            payload["source_lang"] = dl
        r = requests.post(
            "https://api-free.deepl.com/v2/translate",
            headers={
                "Authorization": f"DeepL-Auth-Key {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["translations"][0]["text"]

    def _translate_baidu(self, text):
        """Baidu 百度翻译 - ~2M chars/month free. Set BAIDU_APP_ID, BAIDU_APP_SECRET."""
        app_id = os.environ.get("BAIDU_APP_ID")
        secret = os.environ.get("BAIDU_APP_SECRET")
        if not app_id or not secret:
            return None
        salt = str(uuid.uuid4().hex)[:16]
        sign_str = f"{app_id}{text}{salt}{secret}"
        sign = hashlib.md5(sign_str.encode("utf-8")).hexdigest()
        row = _LANG_MAP.get(self.source_lang, (None, "auto", "auto", None))
        bd_from = row[1]
        row = _LANG_MAP.get(self.target_lang, (None, "en", "en", "en"))
        bd_to = row[1]
        r = requests.get(
            "https://api.fanyi.baidu.com/api/trans/vip/translate",
            params={"q": text, "from": bd_from, "to": bd_to, "appid": app_id, "salt": salt, "sign": sign},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if "error_code" in data:
            raise RuntimeError(data.get("error_msg", str(data)))
        return data["trans_result"][0]["dst"]

    def _translate_youdao(self, text):
        """Youdao 有道 - ~1M chars/month free. Set YOUDAO_APP_KEY, YOUDAO_APP_SECRET."""
        app_key = os.environ.get("YOUDAO_APP_KEY")
        app_secret = os.environ.get("YOUDAO_APP_SECRET")
        if not app_key or not app_secret:
            return None
        salt = str(uuid.uuid4())
        curtime = str(int(time.time()))
        raw = text if len(text) <= 20 else text[:10] + str(len(text)) + text[-10:]
        sign_str = app_key + raw + salt + curtime + app_secret
        sign = hashlib.sha256(sign_str.encode("utf-8")).hexdigest()
        row = _LANG_MAP.get(self.source_lang, (None, "auto", "auto", None))
        yd_from = row[2]
        row = _LANG_MAP.get(self.target_lang, (None, "en", "en", "en"))
        yd_to = row[2]
        r = requests.post(
            "https://openapi.youdao.com/api",
            data={
                "q": text,
                "from": yd_from,
                "to": yd_to,
                "appKey": app_key,
                "salt": salt,
                "sign": sign,
                "signType": "v3",
                "curtime": curtime,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("errorCode") != "0":
            raise RuntimeError(data.get("translation", [str(data)])[0] if isinstance(data.get("translation"), list) else str(data))
        return data["translation"][0]

    def _translate_google(self, text):
        """Google Cloud Translation API v2 - 500K chars/month free. Set GOOGLE_TRANSLATE_API_KEY."""
        key = os.environ.get("GOOGLE_TRANSLATE_API_KEY")
        if not key:
            return None
        row = _LANG_MAP.get(self.source_lang, (None, "auto", "auto", None))
        gl_from = row[3]
        row = _LANG_MAP.get(self.target_lang, (None, "en", "en", "en"))
        gl_to = row[3]
        params = {"q": text, "target": gl_to, "key": key, "format": "text"}
        if gl_from:
            params["source"] = gl_from
        r = requests.post(
            "https://translation.googleapis.com/language/translate/v2",
            params=params,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["data"]["translations"][0]["translatedText"]

    def _translate_yandex(self, text):
        """Yandex Translate API. Set YANDEX_API_KEY."""
        key = os.environ.get("YANDEX_API_KEY")
        if not key:
            return None
        row = _LANG_MAP.get(self.source_lang, (None, "auto", "auto", None))
        yandex_from = row[3] if row[3] else "auto"
        row = _LANG_MAP.get(self.target_lang, (None, "en", "en", "en"))
        yandex_to = row[3] if row[3] else "en"
        r = requests.post(
            "https://translate.yandex.net/api/v1.5/tr.json/translate",
            params={
                "key": key,
                "text": text,
                "lang": f"{yandex_from}-{yandex_to}" if yandex_from != "auto" else yandex_to,
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["text"][0]
        

    def _translate_libretranslate(self, text):
        """LibreTranslate (self-hosted or public instance). Set LIBRETRANSLATE_API_KEY and optionally LIBRETRANSLATE_URL."""
        key = os.environ.get("LIBRETRANSLATE_API_KEY")
        url = os.environ.get("LIBRETRANSLATE_URL", "https://libretranslate.com")
        row = _LANG_MAP.get(self.source_lang, (None, "auto", "auto", None))
        lt_from = row[3] if row[3] else "auto"
        row = _LANG_MAP.get(self.target_lang, (None, "en", "en", "en"))
        lt_to = row[3] if row[3] else "en"
        payload = {
            "q": text,
            "source": lt_from,
            "target": lt_to,
            "format": "text",
        }
        if key:
            payload["api_key"] = key
        r = requests.post(
            f"{url.rstrip('/')}/translate",
            json=payload,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["translatedText"]

    def _translate_caiyun(self, text):
        """Caiyun (彩云小译) - Great for natural/literary Chinese translation.
        Set CAIYUN_TOKEN environment variable."""
        token = os.environ.get("CAIYUN_TOKEN")
        if not token:
            raise ValueError("CAIYUN_TOKEN not set")
        
        # Caiyun uses specific trans_type codes: zh2en, en2zh, ja2zh, etc.
        # Map standard codes to Caiyun format
        lang_map = {
            ("zh", "en"): "zh2en",
            ("en", "zh"): "en2zh", 
            ("ja", "zh"): "ja2zh",
            ("zh", "ja"): "zh2ja",
            ("en", "ja"): "en2ja",
            ("ja", "en"): "ja2en",
        }
        
        key = (self.source_lang, self.target_lang)
        trans_type = lang_map.get(key)
        
        if not trans_type:
            # Fallback to auto if supported, or raise error
            raise ValueError(f"Caiyun doesn't support {self.source_lang} -> {self.target_lang}")
        
        payload = {
            "source": text,
            "trans_type": trans_type,
            "request_id": "bilibili_translator_" + str(hash(text) % 10000)
        }
        
        headers = {
            "X-Authorization": f"token {token}",
            "Content-Type": "application/json"
        }
        
        r = requests.post(
            "https://api.interpreter.caiyunai.com/v1/translator",
            json=payload,
            headers=headers,
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        
        # Caiyun returns target text directly
        if "target" in data:
            return data["target"]
        elif "error" in data:
            raise Exception(f"Caiyun API error: {data['error']}")
        else:
            raise Exception(f"Unexpected Caiyun response: {data}")


    def _translate_niutrans(self, text):
        """Niutrans (小牛翻译) - Good for Asian languages and technical content.
        Set NIUTRANS_APIKEY environment variable."""
        apikey = os.environ.get("NIUTRANS_APIKEY")
        if not apikey:
            raise ValueError("NIUTRANS_APIKEY not set")
        
        # Niutrans uses standard 2-letter codes but calls them differently
        # Map internal codes to Niutrans format
        # Niutrans supports: zh, en, ja, ko, fr, es, ru, etc.
        code_map = {
            "zh": "zh", "en": "en", "ja": "ja", "ko": "ko",
            "fr": "fr", "es": "es", "ru": "ru", "de": "de"
        }
        
        src = code_map.get(self.source_lang, self.source_lang)
        tgt = code_map.get(self.target_lang, self.target_lang)
        
        payload = {
            "src_text": text,
            "from": src,
            "to": tgt,
            "apikey": apikey
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        r = requests.post(
            "https://api.niutrans.com/NiuTransServer/V2/Translation",
            json=payload,
            headers=headers,
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        
        # Niutrans returns error_code 0 for success
        if data.get("error_code") != 0:
            raise Exception(f"Niutrans error {data.get('error_code')}: {data.get('error_msg', 'Unknown')}")
        
        return data["tgt_text"]


    def _llm_translate_role(self):
        """System/role message: translation rules. OCR text goes in user prompt."""
        src = _LANG_NAMES.get(self.source_lang, self.source_lang)
        tgt = _LANG_NAMES.get(self.target_lang, self.target_lang)
        rules = (
            "You are a translation tool for video subtitles. Translate from " + src + " to " + tgt + ".\n"
            "CRITICAL: Output ONLY the translated text. Nothing else whatsoever.\n"
            "- NO notes, NO explanations, NO commentary, NO reasoning, NO suggestions, NO meta-text.\n"
            "- NO lines starting with 'Note:', 'Note :', 'Correction:', 'The correct translation is:', etc.\n"
            "- NO prefixes like 'Translation:' or quotes around the whole output.\n"
            "- NEVER mix source and target languages. Output 100% in " + tgt + " only. Translate EVERY word—never leave any " + src + " text untranslated.\n"
            "- If input is garbled OCR, infer intended meaning and output natural " + tgt + ".\n"
        )
        # When target uses Latin script, forbid romanization and mixed output
        latinscript = ("en", "es", "fr", "de", "it", "pt", "id", "ms", "tr", "pl", "nl", "sv", "da", "fi", "no", "ro", "hu", "sk", "hr", "sl", "et", "lv", "lt", "sw", "af", "ca", "gl", "eu", "vi")
        if self.target_lang in latinscript:
            rules += "- Output natural " + tgt + " only. Do NOT output romanization or phonetic spellings (e.g. pinyin for Chinese, romaji for Japanese). Translate every character into the target language.\n"
        if self.source_lang == "zh":
            rules += "- If the input characters are garbled, it could be because the OCR picked up a wrong but similar character, so infer intended meaning based on what makes sense.\n"
        rules += "- You may receive recent subtitles as context. Use them to infer topic, properly romanized names, and consistent terminology.\n"
        return rules

    def _is_llm_output_sane(self, result, source_text):
        """Reject repetitive/hallucinating LLM output. Fall back to MT."""
        if not result or not isinstance(result, str):
            return False
        s = result.strip()
        if not s:
            return False
        # Length sanity: translation shouldn't be absurdly longer than source
        src_len = len(source_text.strip()) if source_text else 0
        if src_len > 0 and len(s) > max(2000, src_len * 8):
            if self.debug:
                print(f"[LLM] Rejecting output: too long ({len(s)} chars vs {src_len} source)")
            return False
        # Repetition: same token repeated many times = hallucination
        tokens = re.findall(r'\S+', s)
        if len(tokens) >= 10:
            counts = Counter(tokens)
            most_common, cnt = counts.most_common(1)[0]
            if cnt >= 15 or (cnt >= 8 and cnt >= len(tokens) * 0.3):
                if self.debug:
                    print(f"[LLM] Rejecting output: repetitive '{most_common[:30]}...' x{cnt}")
                return False
        # CJK repetition: same short run repeated (e.g. 奇迹奇迹奇迹...)
        cjk_run = re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{2,8}', s)
        if len(cjk_run) >= 10:
            counts = Counter(cjk_run)
            mc, cnt = counts.most_common(1)[0]
            if cnt >= 15:
                if self.debug:
                    print(f"[LLM] Rejecting output: repetitive CJK '{mc}' x{cnt}")
                return False
        return True

    def _fix_mixed_llm_output(self, result):
        """If LLM left source-language text in output, translate fragments via MT.
        Ensures proper spacing and capitalization when splicing translations into existing text."""
        if not result or not isinstance(result, str):
            return result
        latinscript = ("en", "es", "fr", "de", "it", "pt", "id", "ms", "tr", "pl", "nl", "sv", "da", "fi", "no", "ro", "hu", "sk", "hr", "sl", "et", "lv", "lt", "sw", "af", "ca", "gl", "eu", "vi")
        if self.target_lang not in latinscript:
            return result
        # CJK + common source scripts (Chinese, Japanese, Korean)
        cjk_re = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
        spans = [(m.group(), m.start(), m.end()) for m in cjk_re.finditer(result)]
        if not spans:
            return result

        def _is_word_char(c):
            return c.isalnum() or c in "'-" if c else False

        fns = (self._translate_deepl, self._translate_google, self._translate_baidu, self._translate_youdao, self._translate_yandex, self._translate_libretranslate, self._translate_caiyun, self._translate_niutrans)
        out = result
        for frag, start, end in reversed(spans):  # reverse so indices stay valid
            translated = None
            for fn in fns:
                try:
                    translated = fn(frag)
                    if translated:
                        break
                except Exception:
                    continue
            if not translated:
                continue
            translated = str(translated).strip()
            # Ensure proper spacing: add space before if prev char and translation start are both word chars
            need_space_before = start > 0 and translated and _is_word_char(out[start - 1]) and _is_word_char(translated[0])
            # Lowercase first letter if inserting mid-sentence (prev char is lowercase, translation starts uppercase)
            if need_space_before and out[start - 1].islower() and translated[0].isupper():
                translated = translated[0].lower() + translated[1:]
            # Avoid "such a a vast" - strip leading article when prev ends with " a"
            if need_space_before and start >= 2 and out[start - 2:start].lower() == " a" and translated:
                for article in ("a ", "an ", "the "):
                    if translated.lower().startswith(article):
                        translated = translated[len(article):].lstrip()
                        break
            if not translated:  # Guard against stripping everything
                continue
            # Add space after if translation end and next char are both word chars
            need_space_after = end < len(out) and translated and _is_word_char(translated[-1]) and _is_word_char(out[end])
            insert = (" " if need_space_before else "") + translated + (" " if need_space_after else "")
            out = out[:start] + insert + out[end:]
            if self.debug:
                print(f"[LLM] Fixed mixed output: '{frag}' -> '{translated}'")
        return out

    def _build_llm_user_message(self, text, context):
        """Build user message: optional context + current text to translate."""
        if context:
            ctx_lines = []
            for item in context:
                if isinstance(item, tuple) and len(item) == 2:
                    # (source, translation) pair
                    src, trans = item
                    ctx_lines.append(f"- Source: \"{src}\" → Translation: \"{trans}\"")
                else:
                    # Legacy: just a string (backward compatibility)
                    ctx_lines.append(f"- {item}")
            ctx = "Recent subtitles (for topic/name context and translation consistency):\n" + "\n".join(ctx_lines)
            return ctx + "\n\nTranslate:\n" + text
        return text

    def _translate_llm_openai_compat(self, text, base_url, api_key_env, model, extra_headers=None, context=None, timeout=15):
        """OpenAI-compatible chat completion (SiliconFlow, OpenAI, DeepSeek)."""
        key = os.environ.get(api_key_env)
        if not key:
            if self.debug:
                print(f"[LLM] No API key found for {api_key_env}")
            return None
        
        # Validate model name
        if not model or not isinstance(model, str) or not model.strip():
            if self.debug:
                print(f"[LLM] Invalid model name: {model}")
            return None
        
        # Validate text
        if not text or not isinstance(text, str) or not text.strip():
            if self.debug:
                print(f"[LLM] Invalid text input: {text}")
            return None
        
        role = self._llm_translate_role()
        user_content = self._build_llm_user_message(text, context)
        
        # Validate user content
        if not user_content or not user_content.strip():
            if self.debug:
                print(f"[LLM] Empty user content after building message")
            return None
        
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        
        if model == "o3-mini":
            payload = {
                "model": model.strip(),
                "messages": [
                    {"role": "system", "content": role},
                    {"role": "user", "content": user_content},
                ],
            }

        elif model in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            payload = {
                "model": model.strip(),
                "messages": [
                    {"role": "system", "content": role},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 1.0,
                "max_completion_tokens": 500,
            }
        else:
            payload = {
                "model": model.strip(),
                "messages": [
                    {"role": "system", "content": role},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.1,
                "max_completion_tokens": 500,
            }
            # [LLM] 400 Bad Request - Model: o3-mini, Base URL: https://api.openai.com/v1
            # [LLM] Error details: Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.

        
        if self.debug:
            print(f"[LLM] Request to {base_url}: model={model}, text_len={len(text)}, user_content_len={len(user_content)}")
        
        try:
            r = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()
            response_json = r.json()
            
            # Handle different response formats
            if "choices" in response_json and len(response_json["choices"]) > 0:
                out = response_json["choices"][0]["message"]["content"]
            elif "text" in response_json:
                out = response_json["text"]
            else:
                if self.debug:
                    print(f"[LLM] Unexpected response format: {response_json}")
                return None
            
            return out.strip() if isinstance(out, str) else str(out).strip()
        except requests.exceptions.HTTPError as e:
            # Better error reporting for 400 errors
            error_detail = str(e)
            r = getattr(e, 'response', None)
            if r is not None:
                if r.status_code == 400:
                    try:
                        error_json = r.json()
                        error_msg = error_json.get("error", {}).get("message", str(error_json))
                        if self.debug:
                            print(f"[LLM] 400 Bad Request - Model: {model}, Base URL: {base_url}")
                            print(f"[LLM] Error details: {error_msg}")
                            print(f"[LLM] Request payload (sanitized): model={payload['model']}, messages_count={len(payload['messages'])}, user_content_len={len(user_content)}")
                        else:
                            print(f"[LLM] 400 Bad Request: {error_msg[:200]}")
                    except Exception as parse_err:
                        if self.debug:
                            print(f"[LLM] 400 Bad Request - Model: {model}, Base URL: {base_url}")
                            print(f"[LLM] Response text: {r.text[:500]}")
                            print(f"[LLM] Failed to parse error JSON: {parse_err}")
                        else:
                            print(f"[LLM] 400 Bad Request: {r.text[:200]}")
                else:
                    if self.debug:
                        print(f"[LLM] HTTP {r.status_code}: {error_detail}")
                    else:
                        print(f"[LLM] HTTP {r.status_code}: {error_detail[:200]}")
            else:
                if self.debug:
                    print(f"[LLM] HTTP Error (no response): {error_detail}")
                else:
                    print(f"[LLM] HTTP Error: {error_detail[:200]}")
            raise
        except Exception as e:
            if self.debug:
                print(f"[LLM] Request failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            raise

    def _translate_siliconflow_com(self, text, context=None, timeout=15):
        """SiliconFlow.com - OpenAI compatible. Set SILICONFLOW_com_API_KEY."""
        # Ensure model is not None or empty
        model = self.llm_model if (self.llm_model and isinstance(self.llm_model, str) and self.llm_model.strip()) else "Qwen/Qwen2.5-7B-Instruct"
        if self.debug and model != (self.llm_model or ""):
            print(f"[LLM] Using default SiliconFlow.com model: {model} (llm_model was: {self.llm_model})")
        return self._translate_llm_openai_compat(
            text,
            "https://api.siliconflow.com/v1",
            "SILICONFLOW_COM_API_KEY",
            model,
            context=context,
            timeout=timeout,
        )

    def _translate_siliconflow_cn(self, text, context=None, timeout=15):
        """SiliconFlow.cn - OpenAI compatible. Set SILICONFLOW_CN_API_KEY."""
        # Ensure model is not None or empty
        model = self.llm_model if (self.llm_model and isinstance(self.llm_model, str) and self.llm_model.strip()) else "Qwen/Qwen2.5-7B-Instruct"
        if self.debug and model != (self.llm_model or ""):
            print(f"[LLM] Using default SiliconFlow model: {model} (llm_model was: {self.llm_model})")
        return self._translate_llm_openai_compat(
            text,
            "https://api.siliconflow.cn/v1",
            "SILICONFLOW_CN_API_KEY",
            model,
            context=context,
            timeout=timeout,
        )

    def _translate_openai(self, text, context=None, timeout=15):
        """OpenAI GPT. Set OPENAI_API_KEY."""
        model = self.llm_model or "gpt-4o-mini"
        return self._translate_llm_openai_compat(
            text,
            "https://api.openai.com/v1",
            "OPENAI_API_KEY",
            model,
            context=context,
            timeout=timeout,
        )

    def _translate_deepseek(self, text, context=None, timeout=15):
        """DeepSeek (CN). Set DEEPSEEK_API_KEY."""
        model = self.llm_model or "deepseek-chat"
        return self._translate_llm_openai_compat(
            text,
            "https://api.deepseek.com/v1",
            "DEEPSEEK_API_KEY",
            model,
            context=context,
            timeout=timeout,
        )

    def _translate_anthropic(self, text, context=None, timeout=15):
        """Anthropic Claude. Set ANTHROPIC_API_KEY. Uses Messages API."""
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        model = self.llm_model or "claude-3-5-haiku-20241022"
        role = self._llm_translate_role()
        user_content = self._build_llm_user_message(text, context)
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 500,
                "system": role,
                "messages": [{"role": "user", "content": user_content}],
            },
            timeout=timeout,
        )
        r.raise_for_status()
        blocks = r.json().get("content", [])
        for b in blocks:
            if b.get("type") == "text":
                return b.get("text", "").strip()
        return None

    def _translate_groq(self, text, context=None, timeout=15):
        """Groq API - OpenAI compatible. Set GROQ_API_KEY."""
        model = self.llm_model or "llama-3.1-8b-instant"
        return self._translate_llm_openai_compat(
            text,
            "https://api.groq.com/openai/v1",
            "GROQ_API_KEY",
            model,
            context=context,
            timeout=timeout,
        )

    def _translate_together(self, text, context=None, timeout=15):
        """Together AI - OpenAI compatible. Set TOGETHER_API_KEY."""
        model = self.llm_model or "meta-llama/Llama-3-8b-chat-hf"
        return self._translate_llm_openai_compat(
            text,
            "https://api.together.xyz/v1",
            "TOGETHER_API_KEY",
            model,
            context=context,
            timeout=timeout,
        )

    def _translate_huggingface_api(self, text, context=None, timeout=15):
        """HuggingFace Inference API. Set HF_API_KEY."""
        key = os.environ.get("HF_API_KEY")
        if not key:
            return None
        model = self.llm_model or "Helsinki-NLP/opus-mt-zh-en"
        headers = {"Authorization": f"Bearer {key}"}
        payload = {"inputs": text}
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        result = r.json()
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'translation_text' in result[0]:
                return result[0]['translation_text']
            elif isinstance(result[0], str):
                return result[0]
        return None

    def _translate_huggingface_local(self, text, context=None, timeout=15):
        """Local HuggingFace transformers pipeline. Model downloaded on first use."""
        if not hasattr(self, '_hf_local_translator'):
            try:
                from transformers import pipeline
                import torch
                model = self.llm_model or "Helsinki-NLP/opus-mt-zh-en"
                # Device selection: MPS (Mac) -> "mps", CUDA -> 0, else CPU -> -1
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = 0
                else:
                    device = -1
                self._hf_local_translator = pipeline("translation", model=model, device=device)
                print(f"[HF Local] Loaded model: {model} on device {device}")
            except ImportError:
                print("[HF Local] transformers not installed. Install with: pip install transformers torch")
                return None
            except Exception as e:
                print(f"[HF Local] Failed to load model: {e}")
                return None
        try:
            result = self._hf_local_translator(text, max_length=512)
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'translation_text' in result[0]:
                    return result[0]['translation_text']
        except Exception as e:
            print(f"[HF Local] Translation error: {e}")
            return None
        return None

    def _translate_llm(self, text, context=None, timeout=15):
        """LLM translation by selected provider."""
        providers = {
            "siliconflow_com": self._translate_siliconflow_com,
            "siliconflow_cn": self._translate_siliconflow_cn,
            "openai": self._translate_openai,
            "deepseek": self._translate_deepseek,
            "anthropic": self._translate_anthropic,
            "groq": self._translate_groq,
            "together": self._translate_together,
            "huggingface_api": self._translate_huggingface_api,
            "huggingface_local": self._translate_huggingface_local,
        }
        fn = providers.get(self.llm_provider, self._translate_siliconflow_com)
        return fn(text, context=context, timeout=timeout)

    def _translate_for_learn_mode(self, text, timeout=10):
        """Translate text using the learn mode provider/model. Used for word definitions."""
        if not self.learn_mode_provider:
            # Fallback to main translation method
            return self.translate(text)
        
        provider = self.learn_mode_provider
        
        # Local dictionary (English only) - use CEDICT lookup
        if provider == "local_dict":
            if self.target_lang != "en":
                if self.debug:
                    print("[Learn Mode] Local dictionary only works for English target language")
                return None
            try:
                from learn_keywords import KeywordExtractor
                extractor = KeywordExtractor()
                entry = extractor._lookup_cedict(text)
                if entry:
                    _, _, definition = entry
                    return definition if definition else None
            except Exception as e:
                if self.debug:
                    print(f"[Learn Mode] Local dictionary lookup failed: {e}")
                return None
        
        # LLM providers
        if provider in ("siliconflow_com", "siliconflow_cn", "openai", "deepseek", "groq", "together"):
            try:
                base_urls = {
                    "siliconflow_com": "https://api.siliconflow.com/v1",
                    "siliconflow_cn": "https://api.siliconflow.cn/v1",
                    "openai": "https://api.openai.com/v1",
                    "deepseek": "https://api.deepseek.com/v1",
                    "groq": "https://api.groq.com/openai/v1",
                    "together": "https://api.together.xyz/v1",
                }
                api_keys = {
                    "siliconflow_com": "SILICONFLOW_COM_API_KEY",
                    "siliconflow_cn": "SILICONFLOW_CN_API_KEY",
                    "openai": "OPENAI_API_KEY",
                    "deepseek": "DEEPSEEK_API_KEY",
                    "groq": "GROQ_API_KEY",
                    "together": "TOGETHER_API_KEY",
                }
                default_models = {
                    "siliconflow_com": "Qwen/Qwen2.5-7B-Instruct",
                    "siliconflow_cn": "Qwen/Qwen2.5-7B-Instruct",
                    "openai": "gpt-4o-mini",
                    "deepseek": "deepseek-chat",
                    "groq": "llama-3.1-8b-instant",
                    "together": "meta-llama/Llama-3-8b-chat-hf",
                }
                model = self.learn_mode_model or default_models.get(provider, "gpt-4o-mini")
                return self._translate_llm_openai_compat(
                    text, base_urls[provider], api_keys[provider], model, timeout=timeout
                )
            except Exception as e:
                if self.debug:
                    print(f"[Learn Mode] LLM translation failed ({provider}): {e}")
                return None
        
        if provider == "anthropic":
            try:
                return self._translate_anthropic(text, timeout=timeout)
            except Exception as e:
                if self.debug:
                    print(f"[Learn Mode] Anthropic translation failed: {e}")
                return None
        
        if provider == "huggingface_api":
            try:
                # Temporarily set model if specified
                old_model = self.llm_model
                if self.learn_mode_model:
                    self.llm_model = self.learn_mode_model
                result = self._translate_huggingface_api(text, timeout=timeout)
                self.llm_model = old_model
                return result
            except Exception as e:
                if self.debug:
                    print(f"[Learn Mode] HuggingFace API translation failed: {e}")
                return None
        
        if provider == "huggingface_local":
            try:
                # Temporarily set model if specified
                old_model = self.llm_model
                if self.learn_mode_model:
                    self.llm_model = self.learn_mode_model
                result = self._translate_huggingface_local(text, timeout=timeout)
                self.llm_model = old_model
                return result
            except Exception as e:
                if self.debug:
                    print(f"[Learn Mode] HuggingFace Local translation failed: {e}")
                return None
        
        # MT providers
        mt_providers = {
            "deepl": self._translate_deepl,
            "google": self._translate_google,
            "baidu": self._translate_baidu,
            "youdao": self._translate_youdao,
            "yandex": self._translate_yandex,
            "libretranslate": self._translate_libretranslate,
            "caiyun": self._translate_caiyun,
            "niutrans": self._translate_niutrans,
        }
        
        if provider in mt_providers:
            try:
                return mt_providers[provider](text)
            except Exception as e:
                if self.debug:
                    print(f"[Learn Mode] MT translation failed ({provider}): {e}")
                return None
        
        # Fallback to main translation
        return self.translate(text)

    _LLM_PROVIDER_DISPLAY = {
        "siliconflow_com": "SiliconFlow.com", 
        "siliconflow_cn": "SiliconFlow.cn",
        "openai": "OpenAI", 
        "deepseek": "DeepSeek", 
        "anthropic": "Anthropic",
        "groq": "Groq",
        "together": "Together",
        "huggingface_api": "HF API",
        "huggingface_local": "HF Local",
    }

    def _model_display_name(self, model=None, provider=None):
        """Short model name for info pill. E.g. Qwen/Qwen2.5-7B-Instruct -> Qwen2.5-7B."""
        m = model or (self.llm_model if hasattr(self, "llm_model") else None)
        p = provider or (self.llm_provider if hasattr(self, "llm_provider") else None)
        if not m:
            print(f"[LLM] No model found for {p} - {m}")
            self._add_status_message(f"No model found for {p} - {m}", duration_sec=5, is_good_news=False)
        # Strip org prefix (Qwen/, etc.) and -Instruct suffix
        s = str(m).split("/")[-1]
        if s.endswith("-Instruct"):
            s = s[:-9]
        return s

    def _llm_health_check(self):
        """Quick LLM test with 5s timeout. Returns True if LLM responds."""
        try:
            out = self._translate_llm("a", context=None, timeout=5)
            return bool(out and isinstance(out, str) and len(out.strip()) > 0)
        except Exception:
            return False

    def llm_reconnect_thread(self):
        """Background thread: when using MT fallback, check LLM every 30s and switch back when it's up."""
        while self.running:
            time.sleep(30)
            if not self.running:
                break
            if not self.use_large_model or not self._using_mt_fallback:
                continue
            if self._llm_health_check():
                self._using_mt_fallback = False
                # Keep _mt_fallback_message_shown True so in-flight MT translations don't show "switching to DeepL"
                name = self._LLM_PROVIDER_DISPLAY.get(self.llm_provider, self.llm_provider)
                self._add_status_message(f"LLM API reconnecting, switching back to {name}", duration_sec=10, is_good_news=True)
                if self.debug:
                    print(f"[Translate] LLM ({self.llm_provider}) is back, switching from MT fallback")

    def translate(self, text):
        """Translate: LLM if use_large_model else traditional MT (DeepL → Google → Yandex → LibreTranslate → Caiyun → Niutrans)."""
        if text in self._translation_cache:
            return self._translation_cache[text]
        if self.use_large_model:
            result = None
            try:
                # When using MT fallback, skip LLM and go straight to MT (reconnect thread will switch back when LLM is up)
                if not self._using_mt_fallback:
                    context = self._llm_context_sources[-self.llm_context_count:] if self._llm_context_sources and self.llm_context_count > 0 else None
                    self._llm_request_start_time = time.time()
                    self._llm_5sec_message_shown = False
                    if self.debug:
                        print(f"[LLM] Sending ({len(text)} chars): {text[:80]}{'...' if len(text) > 80 else ''}")
                    result = self._translate_llm(text, context=context)
                    if result:
                        if not self._is_llm_output_sane(result, text):
                            result = None  # Reject repetitive/hallucinating output, fall through to MT
                        elif self._llm_5sec_message_shown:
                            self._add_status_message("API responded", duration_sec=8, is_good_news=True)
                    if result:
                        if self.debug:
                            raw = result.strip() if isinstance(result, str) else str(result).strip()
                            print(f"[LLM] Response raw ({len(raw)} chars): {raw[:100]}{'...' if len(raw) > 100 else ''}")
                        result = self._fix_mixed_llm_output(result)
                        if not self._is_llm_output_sane(result, text):
                            result = None  # Fix may have amplified repetition; fall back to MT
                        if result:
                            self._llm_context_sources.append((text, result))
                            if len(self._llm_context_sources) > self.llm_context_count:
                                self._llm_context_sources = self._llm_context_sources[-self.llm_context_count:]
                            if self.debug:
                                stripped = result.strip() if isinstance(result, str) else str(result).strip()
                                print(f"[LLM] Response final ({len(stripped)} chars): {stripped[:100]}{'...' if len(stripped) > 100 else ''}")
                            self._current_display_name = self._model_display_name()
                            self._translation_cache[text] = result
                            return result
            except Exception as ex:
                error_detail = str(ex)
                is_timeout = "timed out" in error_detail.lower() or "timeout" in error_detail.lower()
                self._add_status_message("API Timed Out" if is_timeout else "API Error", duration_sec=10)
                self._using_mt_fallback = True
                self._mt_fallback_message_shown = False  # Allow "switching to MT" when we fall back
                if self.debug:
                    print(f"[Translate] LLM ({self.llm_provider}) failed: {error_detail}")
                else:
                    print(f"[Translate] LLM ({self.llm_provider}) failed: {type(ex).__name__}: {error_detail[:100]}")
                if self.debug:
                    print("[Translate] Falling back to MT (DeepL → Google → Yandex → LibreTranslate → Caiyun → Niutrans)")
            finally:
                self._llm_request_start_time = None
                self._llm_5sec_message_shown = False
            if result is None:
                # LLM failed or we're in fallback mode — try MT
                names = ("DeepL", "Google", "Yandex", "LibreTranslate", "Caiyun", "Niutrans")
                fns = (self._translate_deepl, self._translate_google,  self._translate_yandex, self._translate_libretranslate, self._translate_caiyun, self._translate_niutrans)
                for name, fn in zip(names, fns):
                    try:
                        result = fn(text)
                        if result:
                            if not self._mt_fallback_message_shown:
                                self._add_status_message(f"switching to {name}", duration_sec=10)
                                self._mt_fallback_message_shown = True
                            self._current_display_name = name
                            if self.debug:
                                print(f"[Translate] MT fallback ({name}) succeeded")
                            self._llm_context_sources.append((text, result))
                            if len(self._llm_context_sources) > self.llm_context_count:
                                self._llm_context_sources = self._llm_context_sources[-self.llm_context_count:]
                            self._translation_cache[text] = result
                            return result
                    except Exception as ex:
                        if self.debug:
                            print(f"[Translate] MT fallback {name} failed: {ex}")
                        continue
        if not self.use_large_model:
            names = ("DeepL", "Google", "Baidu", "Youdao", "Yandex", "LibreTranslate", "Caiyun", "Niutrans")
            fns = (self._translate_deepl, self._translate_google, self._translate_baidu, self._translate_youdao, self._translate_yandex, self._translate_libretranslate, self._translate_caiyun, self._translate_niutrans)
            for name, fn in zip(names, fns):
                try:
                    result = fn(text)
                    if result:
                        self._current_display_name = name
                        self._translation_cache[text] = result
                        return result
                except Exception as ex:
                    error_detail = str(ex)
                    if self.debug:
                        print(f"[Translate] {name} failed: {error_detail}")
                    else:
                        print(f"[Translate] {name} failed: {type(ex).__name__}: {error_detail[:100]}")
        # Collect all errors for better reporting
        errors = []
        if self.use_large_model:
            errors.append(f"LLM ({self.llm_provider}) + MT fallback")
        else:
            errors.append("DeepL, Google, Baidu, Youdao, Yandex, LibreTranslate")
        
        error_msg = f"All translation APIs failed ({', '.join(errors)})"
        if self.debug:
            print(f"[Translate] {error_msg} for: {text[:40]}...")
        if not self._translation_fail_warned:
            self._translation_fail_warned = True
            print(f"[Translation failed] {error_msg}")
            print(f"[Translation failed] Check API keys in .env file and network connection")
        fallback = f"Translation Failed: {text[:15]}"
        self._translation_cache[text] = fallback
        return fallback

    def _flush_session_output(self):
        """Write session buffer to JSON file. Called every ~10 translations when session_output_enabled."""
        if not self._session_output_buffer:
            return
        try:
            if self._session_output_path is None:
                base_dir = getattr(self, "session_output_path", "").strip()
                if not base_dir or not os.path.isdir(base_dir):
                    base_dir = os.path.dirname(os.path.abspath(__file__)) if not getattr(sys, "frozen", False) else os.path.dirname(sys.executable)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._session_output_path = os.path.join(base_dir, f"session_{stamp}.json")
            first_ts = self._session_output_buffer[0].get("timestamp", 0)
            
            # Build settings info
            settings_info = {
                "transcription_mode": self.transcription_mode,
                "ocr_backend": getattr(self, "ocr_backend", "vision"),
            }
            
            # Add audio-specific settings if in audio mode
            if self.transcription_mode == "audio":
                settings_info["audio"] = {
                    "asr_backend": getattr(self, "audio_asr_backend", "whisper"),
                    "device_index": getattr(self, "audio_device_index", None),
                    "buffer_settings": getattr(self, "audio_buffer_settings", {}),
                }
            else:
                settings_info["ocr"] = {
                    "backend": getattr(self, "ocr_backend", "vision"),
                }
            
            data = {
                "session_start": datetime.fromtimestamp(first_ts).isoformat() if first_ts else None,
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "transcription_mode": self.transcription_mode,
                "settings": settings_info,
                "entries": self._session_output_buffer,
            }
            with open(self._session_output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if self.debug:
                print(f"[Session] Wrote {len(self._session_output_buffer)} entries to {self._session_output_path}")
        except Exception as ex:
            if self.debug:
                print(f"[Session] Failed to write JSON: {ex}")

    def _rects_overlap(self, region, overlay_rect):
        """True if rects share any area."""
        if not region or not overlay_rect:
            return False
        rx, ry, rw, rh = region["left"], region["top"], region["width"], region["height"]
        ox, oy, ow, oh = overlay_rect
        return not (rx + rw <= ox or ox + ow <= rx or ry + rh <= oy or oy + oh <= ry)

    def _get_effective_region_for_overlap(self, region):
        """Return region to use for overlap/obstruction checks. When auto_detect_text_region is on and we have learned _text_region, use the text area instead of full box."""
        if not region:
            return region
        if not self.auto_detect_text_region or not getattr(self, "_text_region", None):
            return region
        y_min, y_max = self._text_region
        return {
            "left": region["left"],
            "top": region["top"] + y_min,
            "width": region["width"],
            "height": y_max - y_min,
        }

    def _overlap_is_significant(self, region, overlay_rect, inset=30, min_fraction=0.10):
        """True only when overlay is substantially INSIDE the capture box.
        inset: shrink overlay rect by this many px each side (ignores frameGeometry
               shadow/decorations that extend beyond visible content).
        min_fraction: overlap must be at least this fraction of capture region (10%)."""
        if not region or not overlay_rect:
            return False
        rx, ry, rw, rh = region["left"], region["top"], region["width"], region["height"]
        ox, oy, ow, oh = overlay_rect
        # Shrink overlay rect - require overlap of actual content, not frame/shadow
        margin = min(inset, ow // 3, oh // 3)  # at most 1/3 of overlay
        ox_in = ox + margin
        oy_in = oy + margin
        ow_in = max(10, ow - 2 * margin)
        oh_in = max(10, oh - 2 * margin)
        if rx + rw <= ox_in or ox_in + ow_in <= rx or ry + rh <= oy_in or oy_in + oh_in <= ry:
            return False
        ix = max(rx, ox_in)
        iy = max(ry, oy_in)
        iw = min(rx + rw, ox_in + ow_in) - ix
        ih = min(ry + rh, oy_in + oh_in) - iy
        overlap_area = max(0, iw) * max(0, ih)
        capture_area = rw * rh
        return overlap_area >= capture_area * min_fraction

    def _has_chinese(self, text):
        """Check if text contains Chinese characters (CJK unified ideographs)."""
        return any('\u4e00' <= c <= '\u9fff' for c in text)

    def _reset_mixed_content_tracking(self):
        """Reset temporal coherence state (e.g. when region changes or user resumes)."""
        if not hasattr(self, "_mc_bands"):
            return
        self._mc_band_change_counts = [0] * self._mc_bands
        self._mc_total_frames = 0
        self._mc_prev_frame = None
        # Reset dynamic text region so we relearn when region changes
        self._text_region = None
        self._text_region_readings = 0
        self._text_region_min_y = []
        self._text_region_max_y = []

    def _check_mixed_content_temporal(self, frame):
        """Returns (ok, err_msg). ok=False if frame shows mixed content (some bands change, others don't)."""
        if not self.detect_mixed_content or frame is None or frame.size == 0:
            return True, None
        try:
            import numpy as np
            if len(frame.shape) < 2:
                return True, None
            gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame.astype(np.float64)
            h, w = gray.shape[:2]
            if h < 20 or w < 20:
                return True, None
            bands = self._mc_bands
            band_h = h // bands
            if band_h < 2:
                return True, None
            prev = self._mc_prev_frame
            self._mc_prev_frame = gray.copy()
            if prev is None or prev.shape != gray.shape:
                self._mc_total_frames += 1
                return True, None
            # Higher thresh (15) = only substantial changes (text); ignore subtle background/particles
            thresh = 15.0
            for i in range(bands):
                start = i * band_h
                end = (i + 1) * band_h if i < bands - 1 else h
                band_curr = gray[start:end]
                band_prev = prev[start:end]
                diff = np.mean(np.abs(band_curr.astype(np.float64) - band_prev.astype(np.float64)))
                if diff > thresh:
                    self._mc_band_change_counts[i] += 1
            self._mc_total_frames += 1
            if self._mc_total_frames < self._mc_min_frames:
                return True, None
            ratios = [c / self._mc_total_frames for c in self._mc_band_change_counts]
            max_r, min_r = max(ratios), min(ratios)
            # Require: large gap AND at least one band genuinely static (min_r < 0.2)
            # Avoids triggering on video backgrounds with particles/lighting that change everywhere
            if max_r - min_r > self._mc_threshold and min_r < 0.2:
                return False, "OCR area may include static content; please place it around subtitles only"
            return True, None
        except Exception:
            return True, None

    def _detect_language_mismatch(self, text):
        """True if text language differs from source_lang (likely obstructed by another window).
        Stricter thresholds to avoid pausing on video background text (e.g. logos, ads) mixed with subtitles."""
        if not text or len(text.strip()) < 6:
            return False
        text = text.strip()
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af')
        latin_count = sum(1 for c in text if 'a' <= c <= 'z' or 'A' <= c <= 'Z')
        total = len([c for c in text if c.isalnum() or '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af'])
        if total < 8:
            return False
        cjk_ratio = cjk_count / total
        latin_ratio = latin_count / total
        if self.source_lang in ("zh", "ja", "ko"):
            # Require overwhelmingly Latin (>85%) and minimal CJK (<10%) — not just "some" background words
            if cjk_ratio < 0.10 and latin_ratio > 0.85:
                return True
        elif self.source_lang in ("en", "es", "fr", "de", "it", "pt", "id", "ms", "tr", "pl", "nl", "sv"):
            if cjk_ratio > 0.85 and latin_ratio < 0.10:
                return True
        return False

    def _count_words(self, text):
        """Count words: CJK chars + Latin/word tokens. Handles mixed text."""
        if not text or not text.strip():
            return 0
        cjk = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))
        latin = len(re.findall(r'[a-zA-Z]+', text))
        return cjk + latin
    
    def _remove_chinese(self, text):
        """Remove Chinese characters from text, keeping only non-Chinese content."""
        if not text:
            return text
        # Remove CJK unified ideographs (Chinese characters)
        filtered = ''.join(c for c in text if not ('\u4e00' <= c <= '\u9fff'))
        # Clean up extra whitespace but preserve line breaks
        lines = [line.strip() for line in filtered.split('\n')]
        filtered = '\n'.join(line for line in lines if line)
        return filtered.strip()
    
    def _wrap_text_by_length(self, text, target_length):
        """Wrap text into lines approximately matching target_length characters per line."""
        if not text or target_length <= 0:
            return [text] if text else []
        
        words = text.split()
        if not words:
            return [text]
        
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            # Add 1 for space if not first word
            space_length = 1 if current_line else 0
            
            if current_length + space_length + word_length <= target_length or not current_line:
                # Add to current line
                current_line.append(word)
                current_length += space_length + word_length
            else:
                # Start new line
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        
        # Add remaining line
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text]

    def _keywords_similar_to_recent(self, keywords):
        """
        Check if keywords are similar to recently shown ones to avoid repetition.
        Returns True if any keyword was shown recently.
        """
        if not keywords or not hasattr(self, "_recent_keywords"):
            return False
        # Keep recent keywords cache (last N batches)
        recent_words = set()
        for recent_batch in self._recent_keywords[-3:]:  # last 3 batches
            recent_words.update(kw.get("word", "") for kw in recent_batch)
        # Check overlap with current keywords
        current_words = {kw.get("word", "") for kw in keywords}
        overlap = len(current_words & recent_words) / max(len(current_words), 1)
        return overlap >= 0.6  # 60% overlap threshold
        
    def capture_thread(self):
        print("[Capture Thread] Starting capture thread...")
        if self.region_selector:
            def get_region():
                try:
                    return self.region_selector.get_region()
                except RuntimeError:
                    return self.region
            cap = DynamicRegionCapture(get_region, debug=self.debug)
        else:
            cap = DynamicRegionCapture(lambda: self.region, debug=self.debug)
        # When overlay overlaps capture region: either pause (default) or hide-capture-show (allow_overlap)
        overlap_capture_interval = 0.5  # seconds between hide-capture-show when allow_overlap
        last_overlap_capture = 0.0
        overlap_msg_shown = False
        while self.running:
            region = cap.get_region()
            if region is None:
                region = self.region
            eff_region = self._get_effective_region_for_overlap(region)
            overlap_main = bool(self._overlay_rect and self._overlap_is_significant(eff_region, self._overlay_rect))
            overlap_learn = bool(self._learn_overlay_rect and self._overlap_is_significant(eff_region, self._learn_overlay_rect))
            overlap = overlap_main or overlap_learn
            now = time.time()
            if overlap:
                if self.allow_overlap:
                    # Legacy: hide before capture (causes flicker)
                    if now - last_overlap_capture >= overlap_capture_interval:
                        last_overlap_capture = now
                        self._hiding_for_capture = True
                        try:
                            if overlap_main:
                                QMetaObject.invokeMethod(
                                    self.overlay, "hide", Qt.BlockingQueuedConnection
                                )
                            if overlap_learn:
                                learn_o = getattr(self.overlay, "_learn_overlay", None)
                                if learn_o:
                                    QMetaObject.invokeMethod(
                                        learn_o, "hide", Qt.BlockingQueuedConnection
                                    )
                        except Exception:
                            pass
                        frame = cap.capture()
                        try:
                            if overlap_main:
                                QMetaObject.invokeMethod(
                                    self.overlay, "show", Qt.BlockingQueuedConnection
                                )
                            if overlap_learn:
                                learn_o = getattr(self.overlay, "_learn_overlay", None)
                                if learn_o:
                                    QMetaObject.invokeMethod(
                                        learn_o, "show", Qt.BlockingQueuedConnection
                                    )
                        except Exception:
                            pass
                        self._hiding_for_capture = False
                    else:
                        frame = None
                else:
                    # Magnetic snap: animate main overlay above or below OCR (only when main overlaps)
                    if overlap_main and not getattr(self, "_snap_animating", False):
                        self._snap_overlay_requested = True
                        self._snap_region = eff_region  # Use effective region (text area when auto-detect)
                    frame = None
            else:
                frame = cap.capture()
            
            # Debug: show capture result
            if not hasattr(self, '_capture_debug_count'):
                self._capture_debug_count = 0
            self._capture_debug_count += 1
            if self._capture_debug_count <= 5:
                print(f"[Capture Debug {self._capture_debug_count}] frame={'captured' if frame is not None else 'None'}")
            
            if self.capture_queue.full():
                try:
                    self.capture_queue.get_nowait()
                except queue.Empty:
                    pass
            if frame is not None:
                self.capture_queue.put(frame)
                if not hasattr(self, '_capture_put_count'):
                    self._capture_put_count = 0
                self._capture_put_count += 1
                if self._capture_put_count <= 5:
                    print(f"[Capture Thread] Put frame {self._capture_put_count} into queue: shape={frame.shape if frame is not None else None}")
            time.sleep(0.1)

    def _ocr_looks_like_ui_echo(self, text):
        """Skip OCR that looks like our overlays: repetitive Latin (Classes Classes...), no CJK."""
        if not text or len(text.strip()) < 6:
            return False
        a = text.strip()
        has_cjk = any("\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff" for c in a)
        if has_cjk:
            return False  # Real subtitle content has CJK
        words = [w for w in re.split(r"\s+", a.lower()) if len(w) > 2]
        if not words:
            return False
        counts = Counter(words)
        most_common, cnt = counts.most_common(1)[0]
        if cnt >= 5:  # Same word repeated 5+ times = likely our UI
            return True
        return False

    def _ocr_matches_overlay(self, text):
        """Skip OCR text that matches our overlays – capture region can overlap overlay windows."""
        if not text:
            return False
        a = text.strip().lower()
        if not a or len(a) < 4:
            return False
        if self._ocr_looks_like_ui_echo(text):
            return True
        # Check translation overlay
        if self._display_stack:
            displayed = "\n".join(self._display_stack).lower()
            if a in displayed or displayed in a:
                return True
            words_a = set(w for w in a.split() if len(w) > 1)
            words_d = set(w for w in displayed.replace("\n", " ").split() if len(w) > 1)
            if words_a and words_d:
                overlap = len(words_a & words_d) / len(words_a)
                if overlap >= 0.7:
                    return True
        # Check Learn overlay content (keywords: word, pinyin, definition)
        if hasattr(self, "_recent_keywords") and self._recent_keywords:
            learn_text = " ".join(
                str(kw.get("word", "")) + " " + str(kw.get("definition", ""))
                for batch in self._recent_keywords[-2:]
                for kw in batch
            ).lower()
            if learn_text:
                words_a = set(w for w in a.split() if len(w) > 1)
                words_learn = set(w for w in learn_text.split() if len(w) > 1)
                if words_a and words_learn and len(words_a & words_learn) / len(words_a) >= 0.6:
                    return True
        return False

    def _texts_similar(self, a, b):
        """Check if two source texts are similar (OCR variants). Used for cache lookup and OCR dedup."""
        if not a or not b:
            return True
        a, b = a.strip(), b.strip()
        if a == b:
            return True
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if shorter in longer:
            threshold = getattr(self, "ocr_similarity_substring_chars", 15)
            return len(longer) - len(shorter) <= threshold
        # Relax length diff for OCR variants (repeated chars, corrections)
        len_ratio = len(longer) / max(1, len(shorter))
        if len_ratio > 2.0:
            return False
        has_cjk = any("\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff" for c in a + b)
        if has_cjk:
            chars_a, chars_b = set(a), set(b)
            overlap = len(chars_a & chars_b) / min(len(chars_a), len(chars_b)) if chars_a and chars_b else 0
            if overlap >= 0.6:
                return True
            # Substring overlap: does the core of one appear in the other?
            if len(shorter) >= 8:
                core = min(12, len(shorter))
                if shorter[:core] in longer or shorter[-core:] in longer:
                    return True
        diffs = sum(1 for x, y in zip(a, b) if x != y)
        max_len = max(len(a), len(b))
        return diffs <= max(6, max_len // 2)

    def _deduplicate_repeated_phrases(self, text):
        """Collapse repeated/similar phrases (OCR accumulation). Returns deduplicated text or original if no change."""
        if not text or len(text.strip()) < 40:
            return text
        # Split by whitespace and common punctuation
        segments = re.split(r'[\s。，、！？．,;]+', text.strip())
        segments = [s.strip() for s in segments if len(s.strip()) >= 4]
        if len(segments) < 4:
            return text
        seen = []
        result = []
        for seg in segments:
            is_dup = any(self._texts_similar(seg, s) for s in seen)
            if not is_dup:
                seen.append(seg)
                result.append(seg)
        if len(result) >= len(segments) * 0.7:
            return text  # Little dedup, keep original
        deduped = " ".join(result)
        if self.debug and len(deduped) < len(text) * 0.8:
            print(f"[OCR] Deduplicated {len(text)} -> {len(deduped)} chars ({len(segments)} -> {len(result)} segments)")
        return deduped

    def _strip_ocr_garbage(self, text):
        """Strip common OCR noise: video IDs trailing long digit sequences."""
        if not text or not text.strip():
            return text or ""
        # Remove ×NNNNNN (6+ digits) - B站 video ID noise
        t = re.sub(r"×\d{6,}", "", text)
        t = re.sub(r"\d{8,}\s*$", "", t)  # Trailing 8+ digit sequences
        return t.strip() if t else ""

    def _source_similar_to_any(self, text):
        """Skip translating if source is similar to any recently translated source (OCR variant dedup)."""
        if not text or not text.strip():
            return True
        now = time.time()
        self._recent_sources = [(s, ts) for s, ts in self._recent_sources if now - ts < 15]
        for prev, _ in self._recent_sources:
            if prev and self._texts_similar(text, prev):
                return True
        return False

    def _is_similar_to_last(self, text):
        """OCR returns variants (重/蛋/虫, 王不/每个). Uses _texts_similar."""
        return self.last_text and self._texts_similar(text, self.last_text)

    def ocr_thread(self):
        try:
            from ocr_providers import create_ocr_provider
            # Determine languages based on source_lang
            languages = None
            if self.source_lang == "zh":
                languages = ["zh-Hans", "en"]
            elif self.source_lang == "ja":
                languages = ["ja", "en"]
            elif self.source_lang == "ko":
                languages = ["ko", "en"]
            else:
                languages = ["en"]
            
            # Create OCR provider based on selected backend
            ocr = create_ocr_provider(
                backend=self.ocr_backend,
                languages=languages,
            )
            backend_name = self.ocr_backend.replace("_", "-").title()
            print(f"[OCR Thread] {backend_name} initialized successfully")
        except Exception as e:
            print(f"[OCR Thread] ERROR: Failed to initialize OCR backend '{self.ocr_backend}': {e}")
            import traceback
            traceback.print_exc()
            # Fallback to VisionOCR if available
            if self.ocr_backend != "vision":
                try:
                    from vision_ocr import VisionOCR
                    ocr = VisionOCR()
                    print("[OCR Thread] Fallback to VisionOCR successful")
                except Exception as fallback_error:
                    print(f"[OCR Thread] Fallback to VisionOCR also failed: {fallback_error}")
                    return  # Exit thread if all OCR initialization fails
            else:
                return  # Exit thread if OCR initialization fails
        
        ocr_debug_counter = 0
        while self.running:
            try:
                frame = self.capture_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            ocr_debug_counter += 1
            if ocr_debug_counter <= 5:
                print(f"[OCR Debug {ocr_debug_counter}] Got frame: shape={frame.shape if frame is not None else None}")
            
            if self._ocr_paused:
                if ocr_debug_counter <= 5:
                    print(f"[OCR Debug {ocr_debug_counter}] Skipping - paused")
                continue  # Drain frame but skip OCR when paused
            if self.region_selector and getattr(self.region_selector, "_needs_reconfirm", False):
                continue
            # Dynamic text region: use cropped Y range when learned (only when auto_detect_text_region is on)
            work_frame = frame
            if self.auto_detect_text_region and self._text_region is not None and frame is not None and len(frame.shape) >= 2:
                y_min, y_max = self._text_region
                h = frame.shape[0]
                y_min = max(0, min(y_min, h - 1))
                y_max = min(h, max(y_max, y_min + 1))
                if y_max > y_min:
                    work_frame = frame[y_min:y_max, :].copy()

            if self.detect_mixed_content:
                mc_ok, mc_err = self._check_mixed_content_temporal(work_frame)
            else:
                mc_ok, mc_err = True, None
            changed = self.has_changed(work_frame)
            force_ocr = (time.time() - self._last_ocr_time) > 0.5
            if changed or force_ocr:
                if not mc_ok and mc_err:
                    self._ocr_paused = True
                    self._ocr_obstructed = True
                    self._add_status_message(mc_err, duration_sec=10)
                    if self.debug:
                        print(f"[OCR] Mixed content detected, pausing")
                    continue
                self._last_ocr_time = time.time()
                try:
                    # Learn text region from first few readings (only when auto_detect_text_region is on)
                    need_boxes = self.auto_detect_text_region and self._text_region is None and self._text_region_readings < 8
                    out = ocr.process(work_frame, return_boxes=need_boxes)
                    if isinstance(out, tuple):
                        text, obs_candidates = out[0], out[1]
                        boxes = out[2] if len(out) > 2 else []
                    else:
                        text, obs_candidates, boxes = out, None, []
                except Exception as ex:
                    if self.debug:
                        print(f"[OCR error] {ex}")
                    continue
                # Learn text region from bounding boxes (first ~8 readings, only when auto_detect enabled)
                if self.auto_detect_text_region and self._text_region is None and boxes and text and text.strip():
                    for (yt, yb) in boxes:
                        self._text_region_min_y.append(yt)
                        self._text_region_max_y.append(yb)
                    self._text_region_readings += 1
                    if self._text_region_readings >= 5:
                        buf = 15
                        y_min = max(0, min(self._text_region_min_y) - buf)
                        y_max = min(work_frame.shape[0], max(self._text_region_max_y) + buf)
                        if y_max > y_min + 20:
                            self._text_region = (y_min, y_max)
                            if self.debug:
                                print(f"[OCR] Text region learned: y={y_min}-{y_max} (from {work_frame.shape[0]}px frame)")
                if text and text.strip():
                    if self._detect_language_mismatch(text):
                        self._ocr_paused = True
                        self._ocr_obstructed = True
                        self._add_status_message("Text obstructed, please move any other windows out of the way", duration_sec=12)
                        if self.debug:
                            print(f"[OCR] Language mismatch, pausing: {text[:50]}...")
                        continue
                    if self._ocr_matches_overlay(text):
                        if self.debug:
                            print(f"[OCR] skipping (overlay echo): {text[:50]}...")
                        continue
                    if self.max_words_enabled:
                        nw = self._count_words(text)
                        if nw > self.max_words_for_translation:
                            if self.debug:
                                print(f"[OCR] skipping (exceeds {self.max_words_for_translation} words): {nw} words")
                            continue

                    if self.use_large_model:
                        # LLM path: use LLM reconciler (accumulate text, split on sentence boundaries)
                        text = self._strip_ocr_garbage(text)
                        if not text or not text.strip():
                            continue
                        if self.llm_reconciler:
                            try:
                                should_translate, text_to_translate, is_final = self.llm_reconciler.ingest(text)
                                if should_translate and text_to_translate:
                                    text_to_translate = self._deduplicate_repeated_phrases(text_to_translate)
                                    if not text_to_translate or not text_to_translate.strip():
                                        continue
                                    # Check minimum words setting
                                    min_words = getattr(self, "ocr_min_words_before_translate", 0)
                                    if min_words > 0 and self._count_words(text_to_translate) < min_words:
                                        if self.debug:
                                            print(f"[OCR] Skipping translation: {self._count_words(text_to_translate)} words < {min_words} minimum")
                                        continue
                                    if not self._source_similar_to_any(text_to_translate):
                                        self.last_text = text_to_translate
                                        now = time.time()
                                        self._recent_sources.append((text_to_translate, now))
                                        if len(self._recent_sources) > 15:
                                            self._recent_sources = self._recent_sources[-15:]
                                        if self.debug:
                                            print(f"[OCR LLM] {'[FINAL]' if is_final else '[PARTIAL]'} {text_to_translate}")
                                        item = (text_to_translate, is_final, len(text_to_translate), text_to_translate)
                                        try:
                                            self._put_text_queue(item)
                                        except queue.Full:
                                            try:
                                                self.text_queue.get_nowait()
                                            except queue.Empty:
                                                pass
                                            self._put_text_queue(item)
                            except Exception as ex:
                                if self.debug:
                                    print(f"[OCR] Reconciler error: {ex}")
                                text = self._deduplicate_repeated_phrases(text)
                                if not self._source_similar_to_any(text):
                                    self.last_text = text
                                    now = time.time()
                                    self._recent_sources.append((text, now))
                                    if len(self._recent_sources) > 15:
                                        self._recent_sources = self._recent_sources[-15:]
                                    if text and text.strip():
                                        item = (text, True, len(text), text)
                                        try:
                                            self._put_text_queue(item)
                                        except queue.Full:
                                            try:
                                                self.text_queue.get_nowait()
                                            except queue.Empty:
                                                pass
                                            self._put_text_queue(item)
                        else:
                            # No reconciler: simple debounce
                            text = self._deduplicate_repeated_phrases(text)
                            if not text or not text.strip():
                                continue
                            if self._source_similar_to_any(text):
                                continue
                            now = time.time()
                            if self._last_llm_text_sent and self._texts_similar(text, self._last_llm_text_sent):
                                if now - self._last_llm_send_time < 0.25:
                                    continue
                            self.last_text = text
                            self._last_llm_text_sent = text
                            self._last_llm_send_time = now
                            self._recent_sources.append((text, now))
                            if len(self._recent_sources) > 15:
                                self._recent_sources = self._recent_sources[-15:]
                            item = (text, True, len(text), text)
                            try:
                                self._put_text_queue(item)
                            except queue.Full:
                                try:
                                    self.text_queue.get_nowait()
                                except queue.Empty:
                                    pass
                                self._put_text_queue(item)
                        continue

                    # MT path: probabilistic correction, reconciler
                    raw_ocr = text  # Original OCR before correction (for session output)
                    try:
                        from ocr_correct import correct
                        text = correct(text, obs_candidates)
                    except ImportError:
                        pass

                    # Use streaming reconciler for incremental segmentation
                    if self.reconciler:
                        # Filter out debug output that might be captured by OCR
                        text_lower = text.lower()
                        if any(marker in text_lower for marker in ['reconciler', '[ocr]', 'should_translate', 'is_final', 'ingesting', 'no translation trigger', 'text changed', 'committing', 'timeout']):
                            if self.debug:
                                print(f"[OCR] Skipping debug output: '{text[:50]}...'")
                            continue
                        
                        try:
                            should_translate, text_to_translate, is_final = self.reconciler.ingest(text)
                            if should_translate and text_to_translate:
                                text_to_translate = self._deduplicate_repeated_phrases(text_to_translate)
                                if not text_to_translate or not text_to_translate.strip():
                                    continue
                                # Check minimum words setting
                                min_words = getattr(self, "ocr_min_words_before_translate", 0)
                                if min_words > 0 and self._count_words(text_to_translate) < min_words:
                                    if self.debug:
                                        print(f"[OCR] Skipping translation: {self._count_words(text_to_translate)} words < {min_words} minimum")
                                    continue
                                if not self._source_similar_to_any(text_to_translate):
                                    self.last_text = text_to_translate
                                    now = time.time()
                                    self._recent_sources.append((text_to_translate, now))
                                    if len(self._recent_sources) > 15:
                                        self._recent_sources = self._recent_sources[-15:]
                                    if self.debug:
                                        print(f"[OCR Stable] {'[FINAL]' if is_final else '[PARTIAL]'} {text_to_translate}")
                                    original_length = len(text_to_translate)
                                    item = (text_to_translate, is_final, original_length, raw_ocr)
                                    try:
                                        self._put_text_queue(item)
                                    except queue.Full:
                                        try:
                                            self.text_queue.get_nowait()
                                        except queue.Empty:
                                            pass
                                        self._put_text_queue(item)
                        except Exception as ex:
                            if self.debug:
                                print(f"[OCR] Reconciler error: {ex}")
                            import traceback
                            traceback.print_exc()
                            # Fallback to immediate translation on error
                            text = self._deduplicate_repeated_phrases(text)
                            if not self._source_similar_to_any(text) and text and text.strip():
                                self.last_text = text
                                now = time.time()
                                self._recent_sources.append((text, now))
                                if len(self._recent_sources) > 15:
                                    self._recent_sources = self._recent_sources[-15:]
                                original_length = len(text)
                                if self.debug:
                                    print(f"[OCR] Fallback immediate: {text}")
                                item = (text, True, original_length, raw_ocr)
                                try:
                                    self._put_text_queue(item)
                                except queue.Full:
                                    try:
                                        self.text_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                    self._put_text_queue(item)
                    else:
                        # Fallback: immediate translation (old behavior)
                        text = self._deduplicate_repeated_phrases(text)
                        if self.debug and text:
                            print(f"[OCR] No reconciler, using immediate translation: '{text[:60]}...'")
                        if not self._source_similar_to_any(text) and text and text.strip():
                            self.last_text = text
                            now = time.time()
                            self._recent_sources.append((text, now))
                            if len(self._recent_sources) > 15:
                                self._recent_sources = self._recent_sources[-15:]
                            original_length = len(text)
                            if self.debug:
                                print(f"[OCR] {text}")
                            item = (text, True, original_length, raw_ocr)
                            try:
                                self._put_text_queue(item)
                            except queue.Full:
                                try:
                                    self.text_queue.get_nowait()
                                except queue.Empty:
                                    pass
                                self._put_text_queue(item)
    
    def audio_transcription_thread(self):
        """Audio transcription thread."""
        print("[Audio Thread] === STARTING AUDIO TRANSCRIPTION THREAD ===")
        self._audio_model_loading = True
        try:
            import sys
            import os
            subtitle_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtime-subtitle-master")
            if subtitle_dir not in sys.path:
                sys.path.insert(0, subtitle_dir)
            
            # Import from the realtime-subtitle-master directory
            sys.path.insert(0, subtitle_dir)
            from audio_capture import AudioCapture
            from audio_transcriber import Transcriber
            from audio_config import config as audio_config
            import numpy as np
            
            # Use settings from dialog, fallback to config file
            device_index = self.audio_settings.get("device_index") if self.audio_settings.get("device_index") is not None else audio_config.device_index
            asr_backend = self.audio_settings.get("asr_backend", audio_config.asr_backend)
            
            # Log detailed device information
            import sounddevice as sd
            if device_index is not None:
                try:
                    device_info = sd.query_devices(device_index)
                    device_name = device_info.get('name', 'Unknown')
                    print(f"[Audio Thread] Using device_index={device_index}, device_name='{device_name}', asr_backend={asr_backend}")
                    print(f"[Audio Thread] Device details: {device_info.get('max_input_channels', 0)} input channels, sample rate: {device_info.get('default_samplerate', 'unknown')} Hz")
                except Exception as e:
                    print(f"[Audio Thread] Warning: Could not query device {device_index}: {e}")
                    print(f"[Audio Thread] Using device_index={device_index}, asr_backend={asr_backend}")
            else:
                try:
                    default_device = sd.query_devices(kind='input')
                    print(f"[Audio Thread] Using DEFAULT input device: '{default_device.get('name', 'Unknown')}' (index={default_device.get('index', 'unknown')}), asr_backend={asr_backend}")
                except Exception as e:
                    print(f"[Audio Thread] Using default device (could not query: {e}), asr_backend={asr_backend}")
            
            # List all available input devices for debugging
            # print("[Audio Thread] Available input devices:")
            # try:
            #     devices = sd.query_devices()
            #     for i, dev in enumerate(devices):
            #         if dev['max_input_channels'] > 0:
            #             marker = " <-- SELECTED" if (device_index == i or (device_index is None and i == sd.default.device[0])) else ""
            #             print(f"  [{i}] {dev['name']} ({dev['max_input_channels']}ch, {dev.get('default_samplerate', '?')}Hz){marker}")
            # except Exception as e:
            #     print(f"[Audio Thread] Could not list devices: {e}")
            
            # Initialize audio capture
            print(f"[Audio Thread] Initializing AudioCapture with device_index={device_index}...")
            audio = AudioCapture(
                device_index=device_index,
                sample_rate=audio_config.sample_rate,
                silence_threshold=audio_config.silence_threshold,
                silence_duration=audio_config.silence_duration,
                chunk_duration=audio_config.chunk_duration,
                max_phrase_duration=audio_config.max_phrase_duration,
                streaming_mode=audio_config.streaming_mode,
                streaming_interval=audio_config.streaming_interval,
                streaming_step_size=audio_config.streaming_step_size,
                streaming_overlap=audio_config.streaming_overlap
            )
            print(f"[Audio Thread] AudioCapture initialized, starting generator...")
            
            # Initialize transcriber
            if asr_backend == "funasr":
                # Get FunASR model from settings if available, otherwise use config default
                model_size = getattr(self, 'funasr_model', None) or audio_config.funasr_model
            elif asr_backend == "openai":
                model_size = "whisper-1"  # API uses fixed model, not used by Transcriber
            else:
                model_size = audio_config.whisper_model
            
            transcriber = Transcriber(
                backend=asr_backend,
                model_size=model_size,
                device=audio_config.whisper_device,
                compute_type=audio_config.whisper_compute_type,
                language=self.source_lang if self.source_lang != "auto" else None,
                status_callback=self._add_status_message
            )
            
            # Warmup transcriber (overlay shows "Audio Model Loading..." until complete)
            try:
                transcriber.warmup()
            finally:
                self._audio_model_loading = False
                self._audio_awaiting = True  # Show "Awaiting audio" after loading
            self._add_status_message("Loading complete.", duration_sec=2, is_good_news=True)
            
            if self.debug:
                print(f"[Audio Transcription] Started with backend={audio_config.asr_backend}, model={model_size}")
            
            # Accumulating buffer processing loop (similar to realtime-subtitle-master/main.py)
            # Use a queue to handle overflow gracefully - process chunks sequentially even if capture is faster
            import queue as audio_queue_module
            audio_chunk_queue = audio_queue_module.Queue(maxsize=100)  # Increased buffer: 100 chunks (~20s at 0.2s/chunk)
            
            buffer = np.array([], dtype=np.float32)
            chunk_id = 1
            last_update_time = time.time()
            phrase_start_time = time.time()
            last_final_text = ""
            last_reconciler_check_time = time.time()
            last_audio_detected_time = time.time()
            no_audio_warning_shown = False
            buf_init = getattr(self, "audio_buffer_settings", {}) or {}
            from streaming_reconciler import AudioReconciler
            self._audio_reconciler = AudioReconciler(
                period_sec=buf_init.get("reconciler_period_sec", 2.0),
                num_checks=buf_init.get("reconciler_num_checks", 4),
                min_words=buf_init.get("reconciler_min_words", 7),
                debug=self.debug
            )
            # Set up simplified Chinese prompt for Whisper to guide output toward simplified characters
            # This helps ensure audio transcription outputs simplified Chinese instead of traditional
            # Whisper's initial_prompt works best with example text that demonstrates the desired style
            simplified_chinese_prompt = None
            if self.source_lang == "zh":
                # Use example simplified Chinese text to guide Whisper's output style
                simplified_chinese_prompt = "这是简体中文。你好，世界。"  # Simplified Chinese examples
                if self.debug:
                    print("[Audio Thread] Using simplified Chinese prompt to guide transcription output")
            print("[Audio Thread] Getting audio generator...")
            audio_gen = audio.generator()
            print("[Audio Thread] Generator obtained, starting capture thread and entering main loop...")
            
            # Start audio capture in a separate thread to prevent blocking
            def audio_capture_thread():
                """Capture audio chunks and put them in queue - runs independently"""
                overflow_count = 0
                try:
                    for audio_chunk in audio_gen:
                        if not self.running:
                            break
                        try:
                            audio_chunk_queue.put_nowait(audio_chunk)
                            overflow_count = 0  # Reset on success
                        except audio_queue_module.Full:
                            # Queue full - drop multiple oldest chunks to make room (more aggressive)
                            overflow_count += 1
                            dropped = 0
                            try:
                                # Drop up to 3 oldest chunks when queue is full to make more room
                                while audio_chunk_queue.qsize() > 80 and dropped < 3:
                                    audio_chunk_queue.get_nowait()
                                    dropped += 1
                                audio_chunk_queue.put_nowait(audio_chunk)
                                if overflow_count % 5 == 0:  # Only log every 5th overflow
                                    self._add_status_message("Audio queue overflow. Processing faster...", duration_sec=3, is_good_news=False)
                                    if self.debug:
                                        print(f"[Audio] Queue full, dropped {dropped} oldest chunk(s) (overflow #{overflow_count})")
                            except audio_queue_module.Empty:
                                # Queue became empty while draining, just add the new chunk
                                try:
                                    audio_chunk_queue.put_nowait(audio_chunk)
                                except:
                                    pass  # Queue full again, skip this chunk
                except Exception as e:
                    if self.debug:
                        print(f"[Audio Capture Thread] Error: {e}")
            
            import threading
            capture_thread = threading.Thread(target=audio_capture_thread, daemon=True)
            capture_thread.start()
            
            # Debug: Print first few chunks to verify audio is coming in
            debug_audio_counter = 0
            
            while self.running:
                try:
                    # Check queue size to adapt processing speed
                    queue_size = audio_chunk_queue.qsize()
                    
                    # When queue is getting full, process more aggressively
                    if queue_size > 60:
                        # Very full: drain chunks quickly, skip silent ones immediately
                        timeout = 0.01  # Very short timeout
                        max_chunks_per_iteration = 3  # Process multiple chunks at once
                    elif queue_size > 40:
                        # Getting full: process faster
                        timeout = 0.02
                        max_chunks_per_iteration = 2
                    else:
                        # Normal: standard processing
                        timeout = 0.05
                        max_chunks_per_iteration = 1
                    
                    # Process multiple chunks when queue is full
                    audio_chunk = None
                    chunks_attempted = 0
                    while chunks_attempted < max_chunks_per_iteration:
                        try:
                            chunk = audio_chunk_queue.get(timeout=timeout)
                            chunks_attempted += 1
                            
                            # When queue is very full, skip silent chunks immediately
                            if queue_size > 60:
                                chunk_rms = np.sqrt(np.mean(chunk**2))
                                if chunk_rms < audio.silence_threshold * 0.5:
                                    # Skip silent chunk, continue draining
                                    continue  # Try next chunk
                            
                            # Got a valid chunk to process
                            audio_chunk = chunk
                            break
                        except audio_queue_module.Empty:
                            # No more chunks available right now
                            break
                    
                    if audio_chunk is None:
                        # No chunks available or all were skipped
                        if chunks_attempted == 0:
                            # No audio available yet, check if we should continue
                            if not self.running:
                                break
                        continue  # Go to next iteration
                    
                    debug_audio_counter += 1
                    # Debug: Show first 10 chunks with RMS and actual data values
                    if debug_audio_counter <= 10:
                        chunk_rms = np.sqrt(np.mean(audio_chunk**2))
                        chunk_max = np.max(np.abs(audio_chunk))
                        chunk_min = np.min(np.abs(audio_chunk))
                        chunk_mean = np.mean(np.abs(audio_chunk))
                        # print(f"[DEBUG Audio {debug_audio_counter}] RMS={chunk_rms:.4f}, MAX={chunk_max:.4f}, MIN={chunk_min:.4f}, MEAN={chunk_mean:.6f}, samples={len(audio_chunk)}, shape={audio_chunk.shape}")
                        # Print first 10 values
                        # print(f"  First 10 samples: {audio_chunk[:10]}")
                    if not self.running:
                        break
                    
                    # When paused, drain queue but skip processing (clear buffer so resume starts fresh)
                    if getattr(self, "_audio_paused", False):
                        buffer = np.array([], dtype=np.float32)
                        continue
                    buffer = np.concatenate([buffer, audio_chunk])
                    now = time.time()
                    buffer_duration = len(buffer) / audio.sample_rate
                    
                    # Re-read mutable settings each iteration for real-time tuning
                    buf = getattr(self, "audio_buffer_settings", {}) or {}
                    
                    # Check silence for finalization
                    is_silence = False
                    min_silence_dur = buf.get("silence_duration", audio_config.silence_duration)
                    
                    if buffer_duration > min_silence_dur:
                        tail = buffer[-int(audio.sample_rate * min_silence_dur):]
                        rms = np.sqrt(np.mean(tail**2))
                        if rms < audio.silence_threshold:
                            is_silence = True
                    
                    # Dynamic VAD Logic
                    standard_cut = (is_silence and buffer_duration > 2.0)
                    
                    soft_limit_cut = False
                    if buffer_duration > 6.0:
                        short_tail_samps = int(audio.sample_rate * 0.4)
                        if len(buffer) > short_tail_samps:
                            t_rms = np.sqrt(np.mean(buffer[-short_tail_samps:]**2))
                            if t_rms < audio.silence_threshold:
                                soft_limit_cut = True
                    
                    max_phrase = buf.get("max_phrase_duration", audio.max_phrase_duration)
                    hard_limit_cut = (buffer_duration > max_phrase)
                    
                    should_finalize = standard_cut or soft_limit_cut or hard_limit_cut
                    
                    if should_finalize and buffer_duration > 0.5:
                        # FINALIZE
                        final_buffer = buffer.copy()
                        cid = chunk_id
                        prompt = last_final_text
                        
                        # Pre-check: Is the entire buffer actually silence?
                        overall_rms = np.sqrt(np.mean(final_buffer**2))
                        if overall_rms < audio.silence_threshold:
                            if self.debug:
                                print(f"[Audio] Skipped silent chunk {cid} (RMS={overall_rms:.4f})")
                            # Check if we haven't had audio for >10s
                            if time.time() - last_audio_detected_time > 5 and not no_audio_warning_shown:
                                self._add_status_message("No audio detected. Check input device or install BlackHole for system audio.", duration_sec=5, is_good_news=False)
                                no_audio_warning_shown = True
                        else:
                            # We have audio, reset timer
                            last_audio_detected_time = time.time()
                            no_audio_warning_shown = False
                            
                            # Check queue size before transcription - skip if queue is critically full
                            queue_size_before_transcribe = audio_chunk_queue.qsize()
                            if queue_size_before_transcribe > 80:
                                # Queue critically full - skip transcription to drain faster
                                if self.debug:
                                    print(f"[Audio] Queue critically full ({queue_size_before_transcribe}), skipping transcription to drain")
                                buffer = np.array([], dtype=np.float32)  # Reset buffer
                                continue
                            
                            # Transcribe (this may take time, but queue handles overflow)
                            try:
                                # Process final transcriptions - these are important, but skip if queue is getting too full
                                if queue_size_before_transcribe > 50 and self.debug:
                                    print(f"[Audio] Queue size: {queue_size_before_transcribe}, processing final transcription (queue will drain)")
                                
                                # Use simplified Chinese prompt if source language is Chinese
                                transcription_prompt = simplified_chinese_prompt if simplified_chinese_prompt else prompt
                                text = transcriber.transcribe(final_buffer, prompt=transcription_prompt)
                                if text and text.strip():
                                    # Clear "Awaiting audio" message once we get transcribed text
                                    if getattr(self, "_audio_awaiting", False):
                                        self._audio_awaiting = False
                                    
                                    # Filter out unwanted audio patterns (e.g., subtitle metadata, TTS announcements)
                                    unwanted_patterns = [
                                        "amara.org",
                                        "amara",
                                        "subtitle",
                                        "字幕",
                                        "provided by",
                                        "社群提供",
                                        "community",
                                        "captions",
                                        "Spiegel", 
                                        "Spiegel & Dot",
                                        "明镜与点点"
                                    ]
                                    text_lower = text.lower()
                                    if any(pattern in text_lower for pattern in unwanted_patterns):
                                        warning_msg = f"⚠️ Detected unwanted audio pattern: '{text}...'"
                                        print(f"[Audio Warning] {warning_msg}")
                                        print(f"[Audio Warning] This suggests the audio device may be capturing:")
                                        print(f"[Audio Warning]   - Text-to-speech from a browser tab")
                                        print(f"[Audio Warning]   - Accessibility features reading subtitles")
                                        print(f"[Audio Warning]   - Wrong audio source (check device selection in settings)")
                                        print(f"[Audio Warning]   - Audio from a different application")
                                        # self._add_status_message(
                                        #     f"Unwanted audio detected:... Check audio device selection.",
                                        #     duration_sec=8,
                                        #     is_good_news=False
                                        # )
                                        continue  # Skip this transcription
                                    
                                    if self.debug:
                                        print(f"[Audio Final {cid}] Transcribed: {text}")
                                    
                                    # Save for context (only if meaningful)
                                    if len(text.split()) > 2:
                                        last_final_text = text
                                    
                                    # AUDIO MODE: Finals only, same as OCR (one complete item per phrase)
                                    # Send transcriptions directly to translation queue (like realtime-subtitle-master)
                                    # Each phrase is a discrete unit; no accumulation/reconciliation
                                    text = self._deduplicate_repeated_phrases(text)
                                    if not self._source_similar_to_any(text) and text and text.strip():
                                        self.last_text = text
                                        now = time.time()
                                        self._recent_sources.append((text, now))
                                        if len(self._recent_sources) > 15:
                                            self._recent_sources = self._recent_sources[-15:]
                                        original_length = len(text)
                                        item = (text, True, original_length, None)  # is_final=True for finals
                                        try:
                                            self._put_text_queue(item)
                                        except queue.Full:
                                            try:
                                                self.text_queue.get_nowait()
                                            except queue.Empty:
                                                pass
                                            self._put_text_queue(item)
                            except Exception as ex:
                                if self.debug:
                                    print(f"[Audio Transcription Error] {ex}")
                                import traceback
                                traceback.print_exc()
                        
                        # Reset buffer
                        buffer = np.array([], dtype=np.float32)
                        chunk_id += 1
                        phrase_start_time = now
                        last_update_time = now
                        # Reset reconciler for next phrase
                        if hasattr(self, "_audio_reconciler") and self._audio_reconciler:
                            self._audio_reconciler.reset()
                        last_reconciler_check_time = now
                    
                    # AudioReconciler: within X sec, check Y times for sentence completion
                    buf = getattr(self, "audio_buffer_settings", {}) or {}
                    period_sec = buf.get("reconciler_period_sec", 2.0)
                    num_checks = buf.get("reconciler_num_checks", 4)
                    min_words = buf.get("reconciler_min_words", 7)
                    check_interval = period_sec / max(1, num_checks)
                    self._audio_reconciler.period_sec = period_sec
                    self._audio_reconciler.num_checks = num_checks
                    self._audio_reconciler.min_words = min_words
                    if buffer_duration > 0.3 and (now - last_reconciler_check_time) >= check_interval:
                        last_reconciler_check_time = now
                        partial_buffer = buffer.copy()
                        rms = np.sqrt(np.mean(partial_buffer**2))
                        if rms > audio.silence_threshold:
                            try:
                                # Use simplified Chinese prompt if source language is Chinese
                                # For partial transcriptions, combine simplified prompt with last final text
                                if simplified_chinese_prompt and last_final_text:
                                    transcription_prompt = f"{simplified_chinese_prompt} {last_final_text}"
                                elif simplified_chinese_prompt:
                                    transcription_prompt = simplified_chinese_prompt
                                else:
                                    transcription_prompt = last_final_text
                                text = transcriber.transcribe(partial_buffer, prompt=transcription_prompt)
                                if text and text.strip():
                                    # Clear "Awaiting audio" message once we get transcribed text
                                    if getattr(self, "_audio_awaiting", False):
                                        self._audio_awaiting = False
                                    
                                    # Filter out unwanted audio patterns (e.g., subtitle metadata, TTS announcements)
                                    unwanted_patterns = [
                                        "amara.org",
                                        "amara",
                                        "subtitle",
                                        "字幕",
                                        "provided by",
                                        "社群提供",
                                        "community",
                                        "captions",
                                        "closed caption",
                                        "Spiegel",
                                        "明镜与点点",
                                        "Mingjing and Diandian"
                                    ]
                                    text_lower = text.lower()
                                    should_send, text_to_send, _ = self._audio_reconciler.ingest(text)
                                    if should_send and text_to_send:
                                        # Clear "Awaiting audio" message once we get transcribed text
                                        if getattr(self, "_audio_awaiting", False):
                                            self._audio_awaiting = False
                                        text_clean = self._deduplicate_repeated_phrases(text_to_send)
                                        if not self._source_similar_to_any(text_clean) and text_clean:
                                            if self.debug:
                                                print(f"[Audio Reconciler] {text_clean[:60]}...")
                                            if len(text_clean.split()) > 2:
                                                last_final_text = text_clean
                                            item = (text_clean, True, len(text_clean), None)
                                            try:
                                                self._put_text_queue(item)
                                            except queue.Full:
                                                try:
                                                    self.text_queue.get_nowait()
                                                    self._put_text_queue(item)
                                                except queue.Empty:
                                                    pass
                                            # Discard transcribed audio - next check only has NEW audio
                                            buffer = np.array([], dtype=np.float32)
                            except Exception as ex:
                                if self.debug:
                                    print(f"[Audio Reconciler Error] {ex}")
                
                except StopIteration:
                    break
                except Exception as ex:
                    if self.debug:
                        print(f"[Audio Transcription Thread Error] {ex}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)  # Wait before retrying
            
            # Cleanup
            audio.stop()
            if self.debug:
                print("[Audio Transcription] Thread stopped")
        
        except ImportError as e:
            self._audio_model_loading = False
            error_msg = f"Failed to import audio transcription modules: {e}. Please ensure realtime-subtitle-master dependencies are installed."
            print(f"[ERROR] {error_msg}")
            self._add_status_message(error_msg, duration_sec=10)
            # Fallback: just wait
            while self.running:
                time.sleep(1)
        except Exception as e:
            self._audio_model_loading = False
            error_msg = f"Audio transcription initialization failed: {e}"
            print(f"[ERROR] {error_msg}")
            self._add_status_message(error_msg, duration_sec=10)
            import traceback
            traceback.print_exc()
            # Fallback: just wait
            while self.running:
                time.sleep(1)
                    
    def translation_thread(self):
        while self.running:
            try:
                item = self.text_queue.get(timeout=0.5)
                # Handle tuple (text, is_final, original_length[, raw_ocr]) or plain text
                raw_ocr = None
                if isinstance(item, tuple):
                    if len(item) >= 4:
                        text, is_final, original_length, raw_ocr = item[0], item[1], item[2], item[3]
                    elif len(item) == 3:
                        text, is_final, original_length = item
                    elif len(item) == 2:
                        text, is_final = item
                        original_length = len(text)
                    else:
                        text = item[0]
                        is_final, original_length = True, len(text)
                else:
                    text, is_final, original_length = item, True, len(item)
                if self.debug:
                    print(f"[Translation Thread] Got text from queue: '{text[:60]}...' (is_final={is_final}, orig_len={original_length})")
            except queue.Empty:
                continue
            try:
                if self.debug:
                    print(f"[Translation Thread] Translating: '{text[:60]}...'")
                was_cached = text in self._translation_cache
                translated = self.translate(text)
                if translated and not translated.startswith("Translation Failed") and not was_cached:
                    model = self._current_display_name
                    n = self._count_words(text)
                    self._session_word_count_by_model[model] = self._session_word_count_by_model.get(model, 0) + n
                    # Session output: record for JSON export
                    if self.session_output_enabled:
                        self._session_output_buffer.append({
                            "ocr_raw": raw_ocr if raw_ocr is not None else text,
                            "source_text": text,
                            "translation": translated,
                            "model": model,
                            "timestamp": time.time(),
                        })
                        if len(self._session_output_buffer) >= 10:
                            self._flush_session_output()
                if self.debug:
                    print(f"[Translation Thread] Translated result: '{translated[:60] if translated else 'None'}...'")
            except Exception as ex:
                translated = f"[err] {text}"
                if self.debug:
                    print(f"[Translate error] {ex}")
                import traceback
                traceback.print_exc()
            
            # Learn mode: extract keywords for Chinese text
            if self.debug and self.learn_mode:
                has_chinese = self._has_chinese(text)
                print(f"[Learn Debug] learn_mode={self.learn_mode}, has_chinese={has_chinese}, text='{text[:50]}...'")
            if self.learn_mode and self._has_chinese(text):
                try:
                    if self.debug:
                        print(f"[Learn] Processing Chinese text for keyword extraction: '{text[:50]}...'")
                    try:
                        from learn_keywords import extract_keywords
                    except ImportError as ie:
                        if self.debug:
                            print(f"[Learn] Import error: {ie}")
                        raise
                    if self.debug:
                        print(f"[Learn] Calling extract_keywords with target_lang={self.target_lang}, provider={self.learn_mode_provider}, model={self.learn_mode_model}")
                    translate_fn = lambda w: self._translate_for_learn_mode(w) if w else ""
                    keywords = extract_keywords(text, self.target_lang, translate_word_fn=translate_fn)
                    if self.debug:
                        print(f"[Learn] extract_keywords returned: {keywords}")
                    if keywords:
                        # Add metadata about which provider/model was used for definitions
                        provider_display = self.learn_mode_provider or "default"
                        model_display = self.learn_mode_model or "default"
                        # Get human-readable provider name
                        if self.learn_mode_provider == "local_dict":
                            provider_display = "Local Dictionary (CEDICT)"
                        elif self.learn_mode_provider:
                            # Find display name from available providers
                            try:
                                from learn_keywords import get_available_providers
                                available = get_available_providers()
                                for prov_id, prov_name in available:
                                    if prov_id == self.learn_mode_provider:
                                        provider_display = prov_name
                                        break
                            except:
                                pass
                        
                        # Add metadata about which provider/model was used for definitions
                        provider_display = "default"
                        model_display = "default"
                        if self.learn_mode_provider == "local_dict":
                            provider_display = "Local Dictionary (CEDICT)"
                        elif self.learn_mode_provider:
                            # Get human-readable provider name
                            # Try to find in _LLM_PROVIDERS (defined at module level)
                            try:
                                # Access module-level _LLM_PROVIDERS via globals()
                                llm_providers = globals().get('_LLM_PROVIDERS', [])
                                for disp_name, prov_id in llm_providers:
                                    if prov_id == self.learn_mode_provider:
                                        provider_display = disp_name
                                        break
                            except:
                                pass
                            # Also check MT providers
                            mt_names = {
                                "deepl": "DeepL",
                                "google": "Google Translate",
                                "baidu": "Baidu",
                                "youdao": "Youdao",
                                "yandex": "Yandex",
                                "libretranslate": "LibreTranslate",
                            }
                            if self.learn_mode_provider in mt_names:
                                provider_display = mt_names[self.learn_mode_provider]
                            elif provider_display == "default":
                                # Fallback: use provider ID as display name
                                provider_display = self.learn_mode_provider.replace("_", " ").title()
                        
                        if self.learn_mode_model:
                            model_display = self.learn_mode_model
                        
                        # Add metadata to each keyword
                        for kw in keywords:
                            kw["_metadata"] = {
                                "provider": self.learn_mode_provider or "default",
                                "provider_display": provider_display,
                                "model": model_display,
                            }
                        if self.debug:
                            words_list = ", ".join([f"{kw.get('word', '')} ({kw.get('pinyin', '')})" for kw in keywords])
                            print(f"[Learn] Extracted {len(keywords)} keywords from '{text[:30]}...': {words_list}")
                        if not self._keywords_similar_to_recent(keywords):
                            try:
                                self.keyword_queue.put_nowait(keywords)
                                if self.debug:
                                    print(f"[Learn] Added keywords to queue: {words_list}")
                            except queue.Full:
                                # Queue full, remove oldest and add new
                                try:
                                    self.keyword_queue.get_nowait()
                                except queue.Empty:
                                    pass
                                try:
                                    self.keyword_queue.put_nowait(keywords)
                                    if self.debug:
                                        print(f"[Learn] Replaced old keywords in queue: {words_list}")
                                except queue.Full:
                                    if self.debug:
                                        print(f"[Learn] Queue still full, skipping keywords: {words_list}")
                                    pass  # Still full, skip this batch
                        else:
                            if self.debug:
                                words_list = ", ".join([f"{kw.get('word', '')}" for kw in keywords])
                                print(f"[Learn] Skipped similar keywords (already shown recently): {words_list}")
                except Exception as ex:
                    if self.debug:
                        print(f"[Learn error] {ex}")
                    import traceback
                    traceback.print_exc()
            if self.debug:
                print(f"[Translation] {translated}")
            try:
                self.translated_queue.put_nowait((translated, is_final, original_length))
            except queue.Full:
                try:
                    self.translated_queue.get_nowait()
                except queue.Empty:
                    pass
                self.translated_queue.put_nowait((translated, is_final, original_length))

    def _translation_similar_to_any(self, new_text):
        """Skip if new translation is similar to stack or recently shown (reduces paraphrase repetition)."""
        a = new_text.strip()
        if not a:
            return True
        now = time.time()
        candidates = list(self._display_stack)
        candidates += [t for t, ts in self._recent_translations if now - ts < 12]
        self._recent_translations = [(t, ts) for t, ts in self._recent_translations if now - ts < 12]
        a_lower = a.lower()
        for prev in candidates:
            if not prev or not prev.strip():
                continue
            if a == prev.strip():
                return True
            b = prev.strip().lower()
            if a_lower == b:
                return True
            # Substring check: only filter if NEW is a subset of previous (repetition). Do NOT filter when
            # previous is a subset of new (b in a) — that means the new translation is longer/contains more,
            # e.g. LLM returned combined context; we should show it.
            if len(a_lower) >= 20 and len(b) >= 20:
                if a_lower in b:
                    if self.debug:
                        print(f"[Similarity] Filtered substring match (new in prev): '{a[:60]}...' vs '{prev[:60]}...'")
                    return True
            words_a = set(w for w in a_lower.split() if len(w) > 1)
            words_b = set(w for w in b.split() if len(w) > 1)
            if not words_a:
                continue
            # Word overlap: require higher threshold and more words to avoid filtering legitimate new content
            # Only filter if both have substantial word counts and high overlap
            if len(words_a) >= 8 and len(words_b) >= 8:  # Both must have substantial content
                overlap = len(words_a & words_b) / len(words_a)
                if overlap >= 0.65:  # Raised from 0.5 to 0.65 - require higher overlap
                    if self.debug:
                        print(f"[Similarity] Filtered word overlap ({overlap:.2f}): '{a[:60]}...' vs '{prev[:60]}...'")
                    return True
                overlap_b = len(words_a & words_b) / len(words_b) if words_b else 0
                if overlap_b >= 0.65:  # Raised from 0.5 to 0.65
                    if self.debug:
                        print(f"[Similarity] Filtered word overlap reverse ({overlap_b:.2f}): '{a[:60]}...' vs '{prev[:60]}...'")
                    return True
        return False
            
    def ui_update(self):
        now = time.time()
        # After 5 seconds with no LLM response, show [No API response]
        if self.use_large_model and self._llm_request_start_time and not self._llm_5sec_message_shown:
            if now - self._llm_request_start_time >= 5:
                self._add_status_message("No API response", duration_sec=15)
                self._llm_5sec_message_shown = True
        # Update overlay status bar
        active_status = self._get_active_status_messages()
        self.overlay.set_status_messages(active_status)
        # Audio: show loading message while model warms up
        if self.transcription_mode == "audio" and getattr(self, "_audio_model_loading", False):
            self.overlay.update_text("Audio Model Loading...", allow_show=True, partial_text=None)
            return
        # Audio: show "Awaiting audio" after model loads until first audio detected
        if self.transcription_mode == "audio" and getattr(self, "_audio_awaiting", False):
            if not getattr(self, "_audio_paused", False):  # Only show if not paused
                self.overlay.update_text("Awaiting audio...", allow_show=True, partial_text=None)
                return
        # Audio: sync play/pause button with Space/Enter state
        if self.transcription_mode == "audio" and hasattr(self.overlay, "update_play_pause_state"):
            self.overlay.update_play_pause_state()
        # Repaint OCR box so it turns red immediately when paused (mixed content, language mismatch, etc.)
        if self.region_selector and self.region_selector.isVisible():
            self.region_selector.update()
        # Update info pill (per-model word count stack)
        self.overlay.set_info_pill_text(self._session_word_count_by_model)
        # Magnetic snap: when overlay overlaps OCR and allow_overlap is off, animate it above/below
        if getattr(self, "_snap_overlay_requested", False) and not getattr(self, "_snap_animating", False):
            region = getattr(self, "_snap_region", None) or (self.region_selector.get_region() if self.region_selector else self.region)
            if region and self._overlay_rect:
                self._snap_overlay_requested = False
                self._snap_animating = True
                ox, oy, ow, oh = self._overlay_rect
                rx, ry, rw, rh = region["left"], region["top"], region["width"], region["height"]
                overlay_center_y = oy + oh / 2
                region_center_y = ry + rh / 2
                gap = 10
                if overlay_center_y < region_center_y:
                    target_y = ry - oh - gap  # Snap above
                else:
                    target_y = ry + rh + gap  # Snap below
                target_y = max(0, target_y)
                start_y = float(oy)
                steps = 12
                step_duration = 18

                def snap_step(step_num=0):
                    if step_num >= steps:
                        self.overlay.move(ox, int(target_y))
                        self._snap_animating = False
                        return
                    t = step_num / steps
                    # Ease-out cubic for smooth deceleration at end
                    eased = 1.0 - (1.0 - t) ** 3
                    curr_y = start_y + (target_y - start_y) * eased
                    self.overlay.move(ox, int(curr_y))
                    next_step = step_num + 1
                    QTimer.singleShot(step_duration, lambda n=next_step: snap_step(n))

                snap_step(0)
        # Cache overlay geometry for capture thread (must run on main thread)
        try:
            g = self.overlay.frameGeometry()
            self._overlay_rect = (g.x(), g.y(), g.width(), g.height())
        except Exception:
            self._overlay_rect = None
        # Show hint only when overlay overlaps OCR region; hide when moved out
        hint = getattr(self.overlay, "_hint", None)
        if hint and self._overlay_rect:
            region = self.region_selector.get_region() if self.region_selector else self.region
            eff_region = self._get_effective_region_for_overlap(region) if region else None
            if eff_region and self._overlap_is_significant(eff_region, self._overlay_rect):
                hint.move(
                    self.overlay.x() + self.overlay.width() - hint.width() - 100,
                    self.overlay.y() - hint.height() + 5,
                )
                hint.show()
            else:
                hint.hide()
        # Learn mode overlay visibility
        learn_o = getattr(self.overlay, "_learn_overlay", None)
        if learn_o and self.learn_mode:
            learn_o.show()
        # Audio: drain queue fully, process finals first so they're not buried by partials
        # OCR: process 1 per tick
        max_drain = 25 if self.transcription_mode == "audio" else 1
        collected = []
        for _ in range(max_drain):
            try:
                item = self.translated_queue.get_nowait()
                collected.append(item)
            except queue.Empty:
                break
        if not collected:
            pass  # Fall through to learn mode check below
        else:
            # Parse items: (text, is_final, original_length) - treat ALL as stack items (OCR-style)
            def parse_item(it):
                if isinstance(it, tuple):
                    if len(it) >= 3:
                        return it[0], it[2]
                    if len(it) == 2:
                        return it[0], len(it[0])
                    return it[0], len(it[0])
                return it, len(it)
            parsed = [parse_item(it) for it in collected]
            now = time.time()
            for text, original_length in parsed:
                sentence_text = text.strip() if text else ""
                if not sentence_text:
                    continue
                if self._translation_similar_to_any(sentence_text):
                    if self.debug:
                        print(f"[Overlay] Skipped similar: '{sentence_text[:60]}...'")
                    continue
                self._display_stack.append(sentence_text)
                if self.transcription_mode == "ocr" and getattr(self, "tts_enabled", False) and sentence_text:
                    self.tts_engine.speak(sentence_text, lang=self.target_lang)
                while len(self._display_stack) > 2:
                    popped = self._display_stack.pop(0)
                    self._recent_translations.append((popped, now))
                if len(self._recent_translations) > 20:
                    self._recent_translations = self._recent_translations[-20:]
                self._last_translation_time = now
            # Display: last 2 from stack (same as OCR)
            display_lines = list(self._display_stack)[-2:]
            display = "\n".join(display_lines) if display_lines else ""
            self.overlay.update_text(display, allow_show=not self._hiding_for_capture, partial_text=None)
            if self.debug and display_lines:
                print(f"[Overlay] {str(display_lines[-1])[:60]}...")
        if not collected:
            # Still check keyword queue even if no translation
            if self.learn_mode:
                try:
                    while True:
                        keywords = self.keyword_queue.get_nowait()
                        learn_o = getattr(self.overlay, "_learn_overlay", None)
                        if learn_o:
                            if self.debug:
                                words_list = ", ".join([f"{kw.get('word', '')} ({kw.get('pinyin', '')})" for kw in keywords])
                                print(f"[Learn Panel] Displaying keywords: {words_list}")
                            learn_o.update_keywords(keywords)
                            if not hasattr(self, "_recent_keywords"):
                                self._recent_keywords = []
                            self._recent_keywords.append(keywords)
                            if len(self._recent_keywords) > 5:
                                self._recent_keywords.pop(0)
                except queue.Empty:
                    pass
            return
        # Learn mode: drain any pending keywords and update learn overlay
        if self.learn_mode:
            try:
                while True:
                    keywords = self.keyword_queue.get_nowait()
                    learn_o = getattr(self.overlay, "_learn_overlay", None)
                    if learn_o:
                        if self.debug:
                            words_list = ", ".join([f"{kw.get('word', '')} ({kw.get('pinyin', '')})" for kw in keywords])
                            print(f"[Learn Panel] Displaying keywords: {words_list}")
                        learn_o.update_keywords(keywords)
                        if not hasattr(self, "_recent_keywords"):
                            self._recent_keywords = []
                        self._recent_keywords.append(keywords)
                        if len(self._recent_keywords) > 5:
                            self._recent_keywords.pop(0)
            except queue.Empty:
                pass

    def start_threads(self):
        print(f"[Start Threads] transcription_mode={self.transcription_mode}")
        # Start capture thread only for OCR mode
        if self.transcription_mode == "ocr":
            print("[Start Threads] Starting OCR mode threads (capture + OCR)")
            t = threading.Thread(target=self.capture_thread, daemon=True)
            t.start()
            print("[Start Threads] Capture thread started")
            t = threading.Thread(target=self.ocr_thread, daemon=True)
            t.start()
            print("[Start Threads] OCR thread started")
        else:  # audio mode
            print(f"[Start Threads] Starting audio mode thread (transcription_mode={self.transcription_mode})")
            t = threading.Thread(target=self.audio_transcription_thread, daemon=True)
            t.start()
            print("[Start Threads] Audio transcription thread started")
        
        # Translation thread always runs
        t = threading.Thread(target=self.translation_thread, daemon=True)
        t.start()
        
        # LLM reconnect thread always runs
        t = threading.Thread(target=self.llm_reconnect_thread, daemon=True)
        t.start()


class MainControlWindow(QMainWindow):
    """Main application window with OCR and Audio tabs"""
    start_ocr = pyqtSignal()
    start_audio = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, screen_w, screen_h):
        super().__init__()
        self.setWindowTitle("Bilibili OCR & Audio Translator")
        self.setMinimumSize(700, 550)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Helvetica Neue', 'Segoe UI', Arial, sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid #313244;
                background: #1e1e2e;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #313244;
                color: #a6adc8;
                padding: 12px 28px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
                font-size: 14px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                font-weight: bold;
            }
            QTabBar::tab:first:selected {
                background: #00a1d6;
                color: white;
            }
            QTabBar::tab:last:selected {
                background: #89b4fa;
                color: #1e1e2e;
            }
            QLabel {
                font-size: 14px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 6px 8px;
                color: #cdd6f4;
                selection-background-color: #585b70;
                font-size: 13px;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
            QPushButton#StopButton {
                background-color: #f38ba8;
            }
            QPushButton#StopButton:hover {
                background-color: #eba0ac;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #fab387;
            }
            #ocr_tab {
                background-color: #1a1b26;
            }
            #audio_tab {
                background-color: #1e1e2e;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("🎬 Bilibili Live Translator")
        header.setStyleSheet("font-size: 28px; font-weight: bold; color: #cdd6f4;")
        main_layout.addWidget(header)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # OCR Tab
        self.ocr_tab = QWidget()
        self.ocr_tab.setObjectName("ocr_tab")
        self.init_ocr_tab()
        self.tabs.addTab(self.ocr_tab, "📝 OCR Mode")

        # Audio Tab
        self.audio_tab = QWidget()
        self.audio_tab.setObjectName("audio_tab")
        self.init_audio_tab()
        self.tabs.addTab(self.audio_tab, "🎙️ Audio Mode")

        # Status Bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 14px; color: #a6e3a1; padding: 8px 0;")
        main_layout.addWidget(self.status_label)

        # Footer buttons
        footer_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start Translator")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-size: 16px;
                padding: 12px 32px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #94e2d5;
            }
        """)
        self.start_btn.clicked.connect(self._on_start_clicked)

        self.stop_btn = QPushButton("⏹ Stop Translator")
        self.stop_btn.setObjectName("StopButton")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8;
                color: #1e1e2e;
                font-size: 16px;
                padding: 12px 32px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #eba0ac;
            }
        """)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.stop_btn.hide()

        footer_layout.addStretch()
        footer_layout.addWidget(self.start_btn)
        footer_layout.addWidget(self.stop_btn)
        footer_layout.addStretch()
        main_layout.addLayout(footer_layout)

        # Center window on screen
        self.move((screen_w - self.width()) // 2, (screen_h - self.height()) // 2)

    def init_ocr_tab(self):
        """Initialize OCR mode settings tab"""
        layout = QVBoxLayout(self.ocr_tab)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        info = QLabel("OCR mode extracts subtitles from a selected screen region.\n\n"
                     "• Select subtitle region on screen\n"
                     "• Configure OCR and translation settings\n"
                     "• Press Start to begin real-time translation")
        info.setStyleSheet("font-size: 14px; line-height: 1.6; color: #bac2de;")
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch()

    def init_audio_tab(self):
        """Initialize Audio mode settings tab"""
        layout = QVBoxLayout(self.audio_tab)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Audio Device section
        device_group = QWidget()
        device_layout = QFormLayout(device_group)

        self.device_combo = QComboBox()
        self._populate_audio_devices()
        device_layout.addRow("Audio Input Device:", self.device_combo)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self._populate_audio_devices)
        device_layout.addRow("", refresh_btn)
        layout.addWidget(device_group)

        # ASR Settings
        asr_group = QWidget()
        asr_layout = QFormLayout(asr_group)

        self.asr_backend = QComboBox()
        self.asr_backend.addItems(["whisper", "funasr", "mlx"])
        asr_layout.addRow("ASR Backend:", self.asr_backend)
        
        # FunASR Model selection (only shown when FunASR backend is selected)
        self.funasr_model_combo = QComboBox()
        self.funasr_model_combo.setEditable(True)
        self.funasr_model_combo.setToolTip("FunASR model name (must include namespace: iic/ or FunAudioLLM/)")
        # Add recommended models from the user's request
        self.funasr_model_combo.addItems([
            "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",  # Chinese (Offline) - Default
            "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online",  # Chinese (Streaming)
            "iic/speech_UniASR_asr_2pass-en-16k-common-vocab1080-tensorflow1-online",  # English (Streaming)
            "iic/SenseVoiceSmall",  # Multi-language
            "FunAudioLLM/SenseVoiceSmall",  # Multi-language
            "FunAudioLLM/Fun-ASR-Nano-2512",  # Latest 31-language model (dialects, accents, lyrics)
            "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",  # Chinese with VAD and punctuation
            "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",  # Chinese SEACO model
        ])
        # Set default model
        try:
            from audio_config import config
            self.funasr_model_combo.setCurrentText(config.funasr_model)
        except:
            pass
        self.funasr_model_combo.setVisible(False)  # Hidden by default, shown when FunASR is selected
        asr_layout.addRow("FunASR Model:", self.funasr_model_combo)
        
        # Show/hide FunASR model based on backend selection
        def update_funasr_visibility():
            is_funasr = self.asr_backend.currentText() == "funasr"
            self.funasr_model_combo.setVisible(is_funasr)
            # Update label visibility
            for i in range(asr_layout.rowCount()):
                item = asr_layout.itemAt(i, QFormLayout.LabelRole)
                if item and item.widget() and item.widget().text() == "FunASR Model:":
                    item.widget().setVisible(is_funasr)
                    break
        
        self.asr_backend.currentTextChanged.connect(update_funasr_visibility)
        update_funasr_visibility()  # Initial update

        self.silence_threshold = QDoubleSpinBox()
        self.silence_threshold.setRange(0.001, 1.0)
        self.silence_threshold.setSingleStep(0.001)
        self.silence_threshold.setValue(0.005)
        self.silence_threshold.setDecimals(3)
        asr_layout.addRow("Silence Threshold:", self.silence_threshold)

        self.silence_duration = QDoubleSpinBox()
        self.silence_duration.setRange(0.1, 5.0)
        self.silence_duration.setSingleStep(0.1)
        self.silence_duration.setValue(1.0)
        asr_layout.addRow("Silence Duration (s):", self.silence_duration)
        layout.addWidget(asr_group)

        info = QLabel("Audio mode transcribes system audio in real-time.\n\n"
                     "• Select your audio input device (use BlackHole for system audio on macOS)\n"
                     "• Adjust silence detection settings as needed\n"
                     "• Press Start to begin audio transcription and translation")
        info.setStyleSheet("font-size: 14px; line-height: 1.6; color: #bac2de;")
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch()

    def _populate_audio_devices(self):
        """Populate audio input devices list"""
        import sounddevice as sd
        self.device_combo.clear()
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                self.device_combo.addItem(f"{dev['name']} (ID: {i})", i)
        # Set default input device
        default_input = sd.default.device[0]
        if default_input >= 0:
            self.device_combo.setCurrentIndex(self.device_combo.findData(default_input))

    def _on_start_clicked(self):
        current_tab = self.tabs.currentIndex()
        self.status_label.setText("Starting...")
        self.start_btn.hide()
        self.stop_btn.show()
        if current_tab == 0:
            self.start_ocr.emit()
        else:
            self.start_audio.emit()

    def _on_stop_clicked(self):
        self.status_label.setText("Stopping...")
        self.stop_btn.hide()
        self.start_btn.show()
        self.stop_requested.emit()

    def set_status(self, text, color="#a6e3a1"):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"font-size: 14px; color: {color}; padding: 8px 0;")

def main():
    import sys

    print("Starting...")

    app = QApplication(sys.argv)
    app.setApplicationName("BilibiliOCR")
    
    # Install global event filter to show tooltips immediately on mouse enter
    class ImmediateTooltipFilter(QObject):
        def eventFilter(self, obj, event):
            if event.type() == QEvent.Enter:
                # Check if widget has a tooltip
                tooltip_text = None
                if hasattr(obj, 'toolTip'):
                    tooltip_text = obj.toolTip()
                elif isinstance(obj, QWidget):
                    tooltip_text = obj.toolTip()
                
                if tooltip_text:
                    # Get global mouse position
                    global_pos = QCursor.pos()
                    # Show tooltip immediately
                    QToolTip.showText(global_pos, tooltip_text, obj)
            return False  # Don't consume the event
    
    tooltip_filter = ImmediateTooltipFilter()
    app.installEventFilter(tooltip_filter)
    # Accessory mode (no Dock icon) is required for overlay above fullscreen video
    if "--no-accessory" not in sys.argv:
        _mac_set_activation_policy_accessory()

    if "--redownload-piper" in sys.argv:
        from tts_engine import redownload_piper_voices
        print("[TTS] Redownloading Piper voice models...")
        redownload_piper_voices()
        print("[TTS] Done. Continuing with app startup.")

    if not _has_any_api_key():
        show_api_keys_dialog()
        if not _has_any_api_key():
            print("No API keys configured. Add at least one (DeepL, Google, Caiyun, Niutrans, or an LLM key) to translate.")
            sys.exit(1)

    result = show_language_dialog()
    if result[0] is None:
        sys.exit(0)
    source_lang, target_lang, use_large_model, llm_provider, llm_model, learn_mode, learn_mode_provider, learn_mode_model, transcription_mode, audio_device_index, audio_asr_backend, audio_funasr_model, ocr_backend, tts_backend, tts_voice, tts_speed = result
    source_lang = source_lang or "zh"
    target_lang = target_lang or "en"
    print(f"[Main] Audio settings: device={audio_device_index}, backend={audio_asr_backend}, funasr_model={audio_funasr_model}")
    
    # Store audio settings for use in transcription
    audio_settings = {
        "device_index": audio_device_index,
        "asr_backend": audio_asr_backend,
        "funasr_model": audio_funasr_model
    }

    region = None
    region_selector = None
    # Only show region selector for OCR mode
    if transcription_mode == "ocr" and "--no-select" not in sys.argv:
        print("Position the red frame over subtitles. Enter to confirm, Esc to cancel. Frame stays for repositioning.")
        region, region_selector = select_region(app)
        if not region or region["width"] < 10 or region["height"] < 10:
            print("Invalid selection. Using default.")
            region = None
            region_selector = None
    elif transcription_mode == "audio":
        print("Audio transcription mode: No region selection needed.")

    screen = app.primaryScreen().geometry()
    screen_h, screen_w = screen.height(), screen.width()

    # Debug terminal: capture stdout/stderr, toggle with F12
    debug_emitter = _DebugOutputEmitter()
    debug_terminal = DebugTerminal(screen_w, screen_h)
    debug_emitter.text_written.connect(debug_terminal.append, Qt.QueuedConnection)
    _tee_stdout = _TeeStream(sys.stdout, debug_emitter)
    _tee_stderr = _TeeStream(sys.stderr, debug_emitter)
    sys.stdout = _tee_stdout
    sys.stderr = _tee_stderr

    if region is None:
        # Default region (used for overlay positioning even in audio mode)
        region = {
            "left": max(0, (screen_w - 800) // 2),
            "top": screen_h - 150,
            "width": 800,
            "height": 120,
        }

    debug = "--debug" in sys.argv

    overlay_h = 220  # Height for 2 sentences (each up to 2 lines) + half-line gap between
    overlay_below = region["top"] + region["height"] + 10
    overlay_above = region["top"] - overlay_h - 10
    if overlay_below + overlay_h <= screen_h:
        overlay_top = overlay_below
    elif overlay_above >= 0:
        overlay_top = overlay_above
    else:
        overlay_top = max(0, screen_h - overlay_h - 20)

    overlay_below_ocr = overlay_top == overlay_below
    overlay = SubtitleOverlay(
        left=region["left"],
        top=overlay_top,
        width=region["width"],
        height=overlay_h,
        screen_w=screen_w,
        transcription_mode=transcription_mode,
        below_ocr=overlay_below_ocr,
    )
    # hint = QLabel("*Move the translation outside the OCR area to prevent flickering.")
    # hint.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
    # hint.setAttribute(Qt.WA_TranslucentBackground)
    # hint.setStyleSheet("color: rgba(255,255,255,0.9); font-size: 11px;")
    # hint.adjustSize()
    # To move the hint horizontally, change the first argument (X).
    # To move it vertically, change the second argument (Y).
    # Example: Increase "- 20" to move hint further left, or decrease to move right.
    #          Increase "- 10" to move hint further up, or decrease to move down.

    # Create Learn mode overlay (visible only when Chinese detected)
    learn_overlay = None
    if debug:
        print(f"[Main] learn_mode={learn_mode}, creating learn overlay...")
    if learn_mode:
        learn_overlay = LearnOverlay(
            left=screen_w - 470,  # Right side margin (wider widget)
            top=60,
            width=450,  # Increased width for longer definitions
            height=450,
        )
        learn_overlay.show()  # Show immediately when learn_mode is enabled
        overlay._learn_overlay = learn_overlay
        if debug:
            print(f"[Main] Learn overlay created and shown at ({screen_w - 320}, 60)")
    else:
        if debug:
            print(f"[Main] learn_mode is False, skipping learn overlay creation")

    region_ref = region.copy()
    if region_selector:
        region_selector._region_ref = region_ref

        def on_region_changed(new_region):
            overlay.resize(new_region["width"], overlay.height())
            overlay.set_region_size(new_region["width"], screen_w)

        region_selector.region_changed.connect(on_region_changed)

    def quit_all():
        overlay.close()
        if region_selector:
            region_selector.close()
        app.quit()

    def request_quit(translator_ref=None):
        """Central quit handler. Stops TTS, flushes session JSON if enabled, then quits."""
        if translator_ref and hasattr(translator_ref, "tts_engine"):
            translator_ref.tts_engine.stop()
            if hasattr(translator_ref.tts_engine, "shutdown"):
                translator_ref.tts_engine.shutdown()
        if translator_ref and getattr(translator_ref, "session_output_enabled", False) and getattr(translator_ref, "_session_output_buffer", []):
            translator_ref._flush_session_output()
        side_btns.close()
        quit_all()

    overlay._quit_all = lambda: request_quit(None)  # Will be updated after translator exists
    if region_selector:
        region_selector._quit_all = lambda: request_quit(None)
    if learn_overlay:
        learn_overlay._quit_all = lambda: request_quit(None)

    side_btns = _SideButtons(screen_w, screen_h)
    side_btns.show()

    settings = get_app_settings()
    translator = TranslatorApp(
        region_ref, overlay, debug=debug, region_selector=region_selector,
        source_lang=source_lang, target_lang=target_lang,
        use_large_model=use_large_model, llm_provider=llm_provider, llm_model=llm_model,
        learn_mode=learn_mode,
        learn_mode_provider=learn_mode_provider,
        learn_mode_model=learn_mode_model,
        detect_mixed_content=settings.get("detect_mixed_content", False),
        max_words_for_translation=settings.get("max_words_for_translation", 50),
        max_words_enabled=settings.get("max_words_enabled", False),
        allow_overlap=settings.get("allow_overlap", False),
        auto_detect_text_region=settings.get("auto_detect_text_region", False),
        session_output_enabled=settings.get("session_output_enabled", False),
        session_output_path=settings.get("session_output_path", "") or "",
        transcription_mode=transcription_mode,
        audio_device_index=audio_settings.get("device_index"),
        audio_asr_backend=audio_settings.get("asr_backend", "whisper"),
        audio_funasr_model=audio_settings.get("funasr_model"),
        ocr_backend=ocr_backend or "vision",
        tts_backend=tts_backend or "piper",
        tts_voice=tts_voice,
        tts_speed=tts_speed,
    )
    overlay._translator_app = translator
    # Sync initial play/pause button state after app reference is set
    if transcription_mode == "audio" and hasattr(overlay, "update_play_pause_state"):
        overlay.update_play_pause_state()
    if region_selector:
        region_selector._translator_app = translator
        region_selector.region_changed.connect(lambda r: translator._reset_mixed_content_tracking())

    side_btns.set_callbacks(lambda: request_quit(translator), translator)

    overlay._quit_all = lambda: request_quit(translator)
    if region_selector:
        region_selector._quit_all = lambda: request_quit(translator)
    if learn_overlay:
        learn_overlay._quit_all = lambda: request_quit(translator)
    translator._request_quit = lambda: request_quit(translator)

    # Startup: show which API/model is used
    if use_large_model:
        model = llm_model or "?"
        print(f"[Translation] API: LLM ({llm_provider or 'siliconflow_com'}), model: {model}")
    else:
        _mt_checks = [("DeepL", "DEEPL_AUTH_KEY"), ("Google", "GOOGLE_TRANSLATE_API_KEY"), ("Caiyun", "CAIYUN_TOKEN"), ("Niutrans", "NIUTRANS_APIKEY")]
        mt_first = next((n for n, env in _mt_checks if os.environ.get(env)), "none configured")
        print(f"[Translation] API: MT ({mt_first})")

    overlay.show()
    overlay.setFocus()
    if learn_overlay:
        learn_overlay.show()
        learn_overlay.raise_()
    overlay.destroyed.connect(app.quit)
    translator.start_threads()

    # Global Space/Enter for OCR pause (works regardless of focus); F12 toggles debug terminal
    key_filter = _GlobalKeyFilter(translator, debug_terminal=debug_terminal)
    app.installEventFilter(key_filter)

    def bring_front():
        overlay.raise_()
        # hint.raise_()
        if learn_overlay:
            learn_overlay.raise_()
        side_btns.raise_()
        overlay.activateWindow()
        _mac_set_fullscreen_overlay(overlay)
        # _mac_set_fullscreen_overlay(hint)
        if learn_overlay:
            _mac_set_fullscreen_overlay(learn_overlay)
        _mac_set_fullscreen_overlay(side_btns)
    QTimer.singleShot(300, bring_front)

    timer = QTimer()
    timer.timeout.connect(translator.ui_update)
    timer.start(100)  # 10 Hz UI refresh

    if transcription_mode == "ocr":
        print(f"OCR mode: Translating region ({region['width']}x{region['height']}) at ({region['left']}, {region['top']}). Space to pause, Enter to resume, Esc to quit.")
    else:
        print(f"Audio transcription mode: Listening to audio input. Space to pause, Enter to resume, Esc to quit.")
    if debug:
        print("Debug: Watch terminal for [OCR], [Translation], [Overlay] messages.")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()