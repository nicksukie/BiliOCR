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
import threading
import queue
import uuid

import requests
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QDialog, QDialogButtonBox, QLineEdit, QFormLayout, QCheckBox, QListWidget, QListWidgetItem, QMenu, QWidgetAction, QRadioButton, QButtonGroup
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint, QEventLoop, pyqtSignal, QMetaObject
from PyQt5.QtGui import QFont, QPainter, QColor, QPen

from capture_mac import create_capture, DynamicRegionCapture
from vision_ocr import VisionOCR


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
    ("Auto (detect)", "auto"),
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
        "BAIDU_APP_ID", "BAIDU_APP_SECRET",
        "YOUDAO_APP_KEY", "YOUDAO_APP_SECRET",
        "OPENAI_API_KEY", "SILICONFLOW_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY",
    ))


def show_api_keys_dialog(parent=None):
    """Show dialog to enter API keys. Saves to .env in app dir."""
    dlg = QDialog(parent)
    dlg.setWindowTitle("API Keys")
    dlg.setMinimumWidth(420)
    layout = QVBoxLayout(dlg)

    info = QLabel("At least one service needed. Keys are saved locally.")
    info.setWordWrap(True)
    layout.addWidget(info)

    form = QFormLayout()
    pw_edits = []
    deepl = QLineEdit()
    deepl.setPlaceholderText("DeepL API key (500K chars/month free)")
    deepl.setText(os.environ.get("DEEPL_AUTH_KEY", ""))
    deepl.setEchoMode(QLineEdit.Password)
    pw_edits.append(deepl)
    form.addRow("DeepL:", deepl)

    baidu_id = QLineEdit()
    baidu_id.setPlaceholderText("Baidu App ID")
    baidu_id.setText(os.environ.get("BAIDU_APP_ID", ""))
    form.addRow("Baidu App ID:", baidu_id)
    baidu_secret = QLineEdit()
    baidu_secret.setPlaceholderText("Baidu App Secret")
    baidu_secret.setText(os.environ.get("BAIDU_APP_SECRET", ""))
    baidu_secret.setEchoMode(QLineEdit.Password)
    pw_edits.append(baidu_secret)
    form.addRow("Baidu Secret:", baidu_secret)

    youdao_key = QLineEdit()
    youdao_key.setPlaceholderText("Youdao App Key")
    youdao_key.setText(os.environ.get("YOUDAO_APP_KEY", ""))
    form.addRow("Youdao Key:", youdao_key)
    youdao_secret = QLineEdit()
    youdao_secret.setPlaceholderText("Youdao App Secret")
    youdao_secret.setText(os.environ.get("YOUDAO_APP_SECRET", ""))
    youdao_secret.setEchoMode(QLineEdit.Password)
    pw_edits.append(youdao_secret)
    form.addRow("Youdao Secret:", youdao_secret)

    google_key = QLineEdit()
    google_key.setPlaceholderText("Google Cloud Translation API key (500K chars/month free)")
    google_key.setText(os.environ.get("GOOGLE_TRANSLATE_API_KEY", ""))
    google_key.setEchoMode(QLineEdit.Password)
    pw_edits.append(google_key)
    form.addRow("Google:", google_key)

    layout.addWidget(QLabel("--- LLM (for Large Model translation) ---"))
    siliconflow_key = QLineEdit()
    siliconflow_key.setPlaceholderText("SiliconFlow (CN) - api.siliconflow.com")
    siliconflow_key.setText(os.environ.get("SILICONFLOW_API_KEY", ""))
    siliconflow_key.setEchoMode(QLineEdit.Password)
    pw_edits.append(siliconflow_key)
    form.addRow("SiliconFlow:", siliconflow_key)

    openai_key = QLineEdit()
    openai_key.setPlaceholderText("OpenAI / GPT")
    openai_key.setText(os.environ.get("OPENAI_API_KEY", ""))
    openai_key.setEchoMode(QLineEdit.Password)
    pw_edits.append(openai_key)
    form.addRow("OpenAI:", openai_key)

    deepseek_key = QLineEdit()
    deepseek_key.setPlaceholderText("DeepSeek (CN)")
    deepseek_key.setText(os.environ.get("DEEPSEEK_API_KEY", ""))
    deepseek_key.setEchoMode(QLineEdit.Password)
    pw_edits.append(deepseek_key)
    form.addRow("DeepSeek:", deepseek_key)

    anthropic_key = QLineEdit()
    anthropic_key.setPlaceholderText("Anthropic Claude (US)")
    anthropic_key.setText(os.environ.get("ANTHROPIC_API_KEY", ""))
    anthropic_key.setEchoMode(QLineEdit.Password)
    pw_edits.append(anthropic_key)
    form.addRow("Anthropic:", anthropic_key)

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

    if dlg.exec_() != QDialog.Accepted:
        return

    env_path = os.path.join(_app_dir(), ".env")
    lines = []
    env_keys = (
        "DEEPL_AUTH_KEY", "GOOGLE_TRANSLATE_API_KEY", "BAIDU_APP_ID", "BAIDU_APP_SECRET",
        "YOUDAO_APP_KEY", "YOUDAO_APP_SECRET",
        "SILICONFLOW_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY",
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
    add("BAIDU_APP_ID", baidu_id.text())
    add("BAIDU_APP_SECRET", baidu_secret.text())
    add("YOUDAO_APP_KEY", youdao_key.text())
    add("YOUDAO_APP_SECRET", youdao_secret.text())
    add("GOOGLE_TRANSLATE_API_KEY", google_key.text())
    add("SILICONFLOW_API_KEY", siliconflow_key.text())
    add("OPENAI_API_KEY", openai_key.text())
    add("DEEPSEEK_API_KEY", deepseek_key.text())
    add("ANTHROPIC_API_KEY", anthropic_key.text())

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
    from_sel = _LanguageSelector(_LANG_OPTIONS, 1, dlg)  # Chinese
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
    small_rb = QRadioButton("Small model (MT: DeepL, Google, Baidu, Youdao)")
    large_rb = QRadioButton("Large model (LLM)")
    small_rb.setChecked(True)
    model_grp.addButton(small_rb)
    model_grp.addButton(large_rb)
    layout.addWidget(small_rb)
    layout.addWidget(large_rb)

    llm_label = QLabel("LLM provider:")
    _LLM_PROVIDERS = [
        ("SiliconFlow (CN)", "siliconflow"),
        ("OpenAI (US)", "openai"),
        ("DeepSeek (CN)", "deepseek"),
        ("Anthropic (US)", "anthropic"),
    ]
    llm_sel = _LanguageSelector(_LLM_PROVIDERS, 0, dlg)
    llm_sel.setObjectName("lang_selector")
    llm_row = QHBoxLayout()
    llm_row.addWidget(llm_label)
    llm_row.addWidget(llm_sel, 1)
    llm_widget = QWidget()
    llm_widget.setLayout(llm_row)
    layout.addWidget(llm_widget)
    llm_widget.setVisible(False)
    large_rb.toggled.connect(llm_widget.setVisible)

    api_btn = QPushButton("API Keys...")
    api_btn.clicked.connect(lambda: show_api_keys_dialog(dlg))
    layout.addWidget(api_btn)

    btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)
    layout.addWidget(btns)

    if dlg.exec_() != QDialog.Accepted:
        return None, None, False, None
    use_large = large_rb.isChecked()
    llm_provider = _LLM_PROVIDERS[llm_sel.get_index()][1] if use_large else None
    return (
        _LANG_OPTIONS[from_sel.get_index()][1],
        _LANG_OPTIONS_TARGET[to_sel.get_index()][1],
        use_large,
        llm_provider,
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
        if self._active and not self._needs_reconfirm:
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


# --- Overlay ---


class SubtitleOverlay(QWidget):
    def __init__(self, left=400, top=780, width=800, height=100):
        super().__init__()

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(left, top, width, height)
        self.setFocusPolicy(Qt.StrongFocus)
        self._drag_start = None
        self.setCursor(Qt.OpenHandCursor)

        layout = QVBoxLayout()
        self.label = QLabel("Waiting for subtitles... (Esc to quit)")
        self.label.setWordWrap(True)
        self.label.setFont(QFont("Arial", 16))
        self.label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 180);
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_start = (e.globalPos(), self.frameGeometry().topLeft())

    def mouseMoveEvent(self, e):
        if self._drag_start:
            delta = e.globalPos() - self._drag_start[0]
            self.move(self._drag_start[1] + delta)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
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

    def update_text(self, text, allow_show=True):
        """Update text. allow_show=False keeps overlay hidden during brief capture hide."""
        self.label.setText(text or "Waiting for subtitles...")
        if allow_show:
            self.setVisible(bool(text))


# --- Main app ---


# Human-readable language names for LLM prompts
_LANG_NAMES = {code: lbl for lbl, code in _LANG_OPTIONS}
_LANG_NAMES["auto"] = "the detected language"


class TranslatorApp:
    def __init__(self, region, overlay, debug=False, region_selector=None, source_lang="auto", target_lang="en",
                 use_large_model=False, llm_provider=None):
        self.region = region
        self.overlay = overlay
        self.region_selector = region_selector
        self.debug = debug
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.use_large_model = bool(use_large_model)
        self.llm_provider = llm_provider or "siliconflow"

        self.capture_queue = queue.Queue(maxsize=1)
        self.text_queue = queue.Queue(maxsize=5)
        self.translated_queue = queue.Queue(maxsize=5)

        self.last_hash = None
        self.last_text = None
        self.running = True
        self._display_stack = []
        self._last_translation_time = 0
        self._stack_window_sec = 3.0
        self._last_ocr_time = 0
        self._translation_cache = {}  # source_text -> translated
        self._subtitle_buffer = []
        self._buffer_updated_at = 0
        self._debounce_sec = 0.8
        self._use_buffer = False  # True = accumulate context; False = immediate (more reliable)
        self._translation_fail_warned = False
        self._recent_translations = []  # [(text, timestamp)] for dedup beyond stack
        try:
            g = overlay.frameGeometry()
            self._overlay_rect = (g.x(), g.y(), g.width(), g.height())
        except Exception:
            self._overlay_rect = None
        self._hiding_for_capture = False

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

    def _llm_translate_prompt(self, text):
        """Strict translation prompt - no assistant behavior, only output the translation."""
        src = _LANG_NAMES.get(self.source_lang, self.source_lang)
        tgt = _LANG_NAMES.get(self.target_lang, self.target_lang)
        return (
            "You are a translation tool. Translate the input text to " + tgt + ".\n"
            "Reply with ONLY the translated text. No explanations, no commentary, no prefixes like 'Translation:'.\n\n"
            "Input:\n" + text
        )

    def _translate_llm_openai_compat(self, text, base_url, api_key_env, model, extra_headers=None):
        """OpenAI-compatible chat completion (SiliconFlow, OpenAI, DeepSeek)."""
        key = os.environ.get(api_key_env)
        if not key:
            return None
        prompt = self._llm_translate_prompt(text)
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        r = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500,
            },
            timeout=15,
        )
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"]
        return out.strip() if isinstance(out, str) else str(out).strip()

    def _translate_siliconflow(self, text):
        """SiliconFlow (CN) - OpenAI compatible. Set SILICONFLOW_API_KEY."""
        return self._translate_llm_openai_compat(
            text,
            "https://api.siliconflow.com/v1",
            "SILICONFLOW_API_KEY",
            "Qwen/Qwen2.5-7B-Instruct",
        )

    def _translate_openai(self, text):
        """OpenAI GPT. Set OPENAI_API_KEY."""
        return self._translate_llm_openai_compat(
            text,
            "https://api.openai.com/v1",
            "OPENAI_API_KEY",
            "gpt-4o-mini",
        )

    def _translate_deepseek(self, text):
        """DeepSeek (CN). Set DEEPSEEK_API_KEY."""
        return self._translate_llm_openai_compat(
            text,
            "https://api.deepseek.com/v1",
            "DEEPSEEK_API_KEY",
            "deepseek-chat",
        )

    def _translate_anthropic(self, text):
        """Anthropic Claude. Set ANTHROPIC_API_KEY. Uses Messages API."""
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        prompt = self._llm_translate_prompt(text)
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        r.raise_for_status()
        blocks = r.json().get("content", [])
        for b in blocks:
            if b.get("type") == "text":
                return b.get("text", "").strip()
        return None

    def _translate_llm(self, text):
        """LLM translation by selected provider."""
        providers = {
            "siliconflow": self._translate_siliconflow,
            "openai": self._translate_openai,
            "deepseek": self._translate_deepseek,
            "anthropic": self._translate_anthropic,
        }
        fn = providers.get(self.llm_provider, self._translate_siliconflow)
        return fn(text)

    def translate(self, text):
        """Translate: LLM if use_large_model else traditional MT (DeepL → Google → Baidu → Youdao)."""
        if text in self._translation_cache:
            return self._translation_cache[text]
        if self.use_large_model:
            try:
                result = self._translate_llm(text)
                if result:
                    self._translation_cache[text] = result
                    return result
            except Exception as ex:
                if self.debug:
                    print(f"[Translate] LLM ({self.llm_provider}) failed: {ex}")
        else:
            names = ("DeepL", "Google", "Baidu", "Youdao")
            fns = (self._translate_deepl, self._translate_google, self._translate_baidu, self._translate_youdao)
            for name, fn in zip(names, fns):
                try:
                    result = fn(text)
                    if result:
                        self._translation_cache[text] = result
                        return result
                except Exception as ex:
                    if self.debug:
                        print(f"[Translate] {name} failed: {ex}")
        if self.debug:
            print(f"[Translate] All APIs failed for: {text[:40]}...")
        if not self._translation_fail_warned:
            self._translation_fail_warned = True
            print(f"[Translation failed: {text[:15]}...]")
        fallback = f"[Translation failed: {text[:15]}...]"
        self._translation_cache[text] = fallback
        return fallback

    def _rects_overlap(self, region, overlay_rect):
        """True if rects share any area."""
        if not region or not overlay_rect:
            return False
        rx, ry, rw, rh = region["left"], region["top"], region["width"], region["height"]
        ox, oy, ow, oh = overlay_rect
        return not (rx + rw <= ox or ox + ow <= rx or ry + rh <= oy or oy + oh <= ry)

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

    def capture_thread(self):
        if self.region_selector:
            def get_region():
                try:
                    return self.region_selector.get_region()
                except RuntimeError:
                    return self.region
            cap = DynamicRegionCapture(get_region, debug=self.debug)
        else:
            cap = DynamicRegionCapture(lambda: self.region, debug=self.debug)
        # When overlay overlaps capture region: hide before capture (only way to get content underneath)
        # Throttle to ~2 fps to minimize flicker
        overlap_capture_interval = 0.5  # seconds between hide-capture-show when overlapping
        last_overlap_capture = 0.0
        while self.running:
            region = cap.get_region()
            if region is None:
                region = self.region
            # Only hide when overlay meaningfully overlaps (inside box), not just near/touching
            overlap = bool(self._overlay_rect and self._overlap_is_significant(region, self._overlay_rect))
            now = time.time()
            if overlap:
                if now - last_overlap_capture >= overlap_capture_interval:
                    last_overlap_capture = now
                    self._hiding_for_capture = True
                    try:
                        QMetaObject.invokeMethod(
                            self.overlay, "hide", Qt.BlockingQueuedConnection
                        )
                    except Exception:
                        pass
                    frame = cap.capture()
                    try:
                        QMetaObject.invokeMethod(
                            self.overlay, "show", Qt.BlockingQueuedConnection
                        )
                    except Exception:
                        pass
                    self._hiding_for_capture = False
                else:
                    frame = None  # skip capture this cycle when throttled
            else:
                frame = cap.capture()
            if self.capture_queue.full():
                try:
                    self.capture_queue.get_nowait()
                except queue.Empty:
                    pass
            if frame is not None:
                self.capture_queue.put(frame)
            time.sleep(0.1)

    def _ocr_matches_overlay(self, text):
        """Skip OCR text that matches our overlay – capture region can overlap the overlay window."""
        if not text or not self._display_stack:
            return False
        a = text.strip().lower()
        if not a or len(a) < 4:
            return False
        displayed = "\n".join(self._display_stack).lower()
        if a in displayed or displayed in a:
            return True
        words_a = set(w for w in a.split() if len(w) > 1)
        words_d = set(w for w in displayed.replace("\n", " ").split() if len(w) > 1)
        if not words_a:
            return False
        overlap = len(words_a & words_d) / len(words_a)
        return overlap >= 0.7

    def _texts_similar(self, a, b):
        """Check if two source texts are similar (OCR variants). Used for cache lookup and OCR dedup."""
        if not a or not b:
            return True
        a, b = a.strip(), b.strip()
        if a == b:
            return True
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if shorter in longer:
            return len(longer) - len(shorter) <= 5
        if len(longer) - len(shorter) > 8:
            return False
        has_cjk = any("\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff" for c in a + b)
        if has_cjk:
            chars_a, chars_b = set(a), set(b)
            overlap = len(chars_a & chars_b) / min(len(chars_a), len(chars_b)) if chars_a and chars_b else 0
            if overlap >= 0.72 and len(longer) / max(1, len(shorter)) <= 1.5:
                return True
        diffs = sum(1 for x, y in zip(a, b) if x != y)
        max_len = max(len(a), len(b))
        return diffs <= max(4, max_len // 3)

    def _is_similar_to_last(self, text):
        """OCR returns variants (重/蛋/虫, 王不/每个). Uses _texts_similar."""
        return self.last_text and self._texts_similar(text, self.last_text)

    def _add_to_buffer(self, text):
        """Accumulate segments. Progressive reveal: replace. Same sentence: append. New subtitle: reset."""
        text = text.strip()
        if not text or self._ocr_matches_overlay(text):
            return
        now = time.time()
        self._buffer_updated_at = now
        if self._subtitle_buffer:
            last = self._subtitle_buffer[-1]
            if text == last:
                return
            if text.startswith(last) or last in text:
                # Progressive reveal: "你" -> "你好" -> "你好 世界"
                self._subtitle_buffer[-1] = text if len(text) > len(last) else last
                return
            if not (last in text or text in last):
                # Completely different = new subtitle, start fresh
                self._subtitle_buffer.clear()
        self._subtitle_buffer.append(text)
        if len(self._subtitle_buffer) > 4:
            self._subtitle_buffer.pop(0)

    def _flush_buffer(self):
        """Join buffered segments and send for translation. Skip if same as last."""
        if not self._subtitle_buffer:
            return
        combined = " ".join(self._subtitle_buffer)
        self._subtitle_buffer.clear()
        if combined and combined != self.last_text:
            self.last_text = combined
            if self.debug:
                print(f"[OCR combined] {combined}")
            self.text_queue.put(combined)

    def ocr_thread(self):
        ocr = VisionOCR()
        while self.running:
            try:
                frame = self.capture_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.region_selector and getattr(self.region_selector, "_needs_reconfirm", False):
                continue
            changed = self.has_changed(frame)
            force_ocr = (time.time() - self._last_ocr_time) > 0.5
            if changed or force_ocr:
                self._last_ocr_time = time.time()
                try:
                    text = ocr.process(frame)
                except Exception as ex:
                    if self.debug:
                        print(f"[OCR error] {ex}")
                    continue
                if text and text.strip():
                    if self._ocr_matches_overlay(text):
                        if self.debug:
                            print(f"[OCR] skipping (overlay echo): {text[:50]}...")
                        continue
                    if self._use_buffer:
                        self._add_to_buffer(text)
                    else:
                        if not self._is_similar_to_last(text):
                            self.last_text = text
                            if self.debug:
                                print(f"[OCR] {text}")
                            try:
                                self.text_queue.put_nowait(text)
                            except queue.Full:
                                try:
                                    self.text_queue.get_nowait()
                                except queue.Empty:
                                    pass
                                self.text_queue.put_nowait(text)
            if self._use_buffer:
                now = time.time()
                if self._subtitle_buffer and (now - self._buffer_updated_at) >= self._debounce_sec:
                    self._flush_buffer()

    def translation_thread(self):
        while self.running:
            try:
                text = self.text_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                translated = self.translate(text)
            except Exception as ex:
                translated = f"[err] {text}"
                if self.debug:
                    print(f"[Translate error] {ex}")
            if self.debug:
                print(f"[Translation] {translated}")
            try:
                self.translated_queue.put_nowait(translated)
            except queue.Full:
                try:
                    self.translated_queue.get_nowait()
                except queue.Empty:
                    pass
                self.translated_queue.put_nowait(translated)

    def _translation_similar_to_any(self, new_text):
        """Skip if new translation is similar to stack or recently shown (reduces paraphrase repetition)."""
        a = new_text.strip().lower()
        if not a:
            return True
        words_a = set(w for w in a.split() if len(w) > 1)
        now = time.time()
        candidates = list(self._display_stack) + [t for t, ts in self._recent_translations if now - ts < 12]
        self._recent_translations = [(t, ts) for t, ts in self._recent_translations if now - ts < 12]
        for prev in candidates:
            if new_text == prev:
                return True
            b = prev.strip().lower()
            if not b:
                continue
            if a in b or b in a:
                return True
            words_b = set(w for w in b.split() if len(w) > 1)
            if not words_a:
                continue
            overlap = len(words_a & words_b) / len(words_a)
            if overlap >= 0.5:
                return True
            overlap_b = len(words_a & words_b) / len(words_b) if words_b else 0
            if overlap_b >= 0.5:
                return True
        return False

    def ui_update(self):
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
            if region and self._overlap_is_significant(region, self._overlay_rect):
                hint.move(
                    self.overlay.x() + self.overlay.width() - hint.width() - 100,
                    self.overlay.y() - hint.height() + 5,
                )
                hint.show()
            else:
                hint.hide()
        try:
            text = self.translated_queue.get_nowait()
        except queue.Empty:
            return
        if self._translation_similar_to_any(text):
            return
        now = time.time()
        if self._display_stack and (now - self._last_translation_time) > self._stack_window_sec:
            for t in self._display_stack:
                self._recent_translations.append((t, self._last_translation_time))
            if len(self._recent_translations) > 20:
                self._recent_translations = self._recent_translations[-20:]
            self._display_stack.clear()
        self._display_stack.append(text)
        if len(self._display_stack) > 3:
            self._display_stack.pop(0)
        self._last_translation_time = now
        display = "\n".join(self._display_stack)
        self.overlay.update_text(display, allow_show=not self._hiding_for_capture)
        if self.debug:
            print(f"[Overlay] {display[:60]}...")

    def start_threads(self):
        for target in (self.capture_thread, self.ocr_thread, self.translation_thread):
            t = threading.Thread(target=target, daemon=True)
            t.start()


def main():
    import sys

    print("Starting...")

    app = QApplication(sys.argv)
    app.setApplicationName("BilibiliOCR")
    # Accessory mode (no Dock icon) is required for overlay above fullscreen video
    if "--no-accessory" not in sys.argv:
        _mac_set_activation_policy_accessory()

    if not _has_any_api_key():
        show_api_keys_dialog()
        if not _has_any_api_key():
            print("No API keys configured. Add at least one (DeepL, Google, Baidu, Youdao, or an LLM key) to translate.")
            sys.exit(1)

    result = show_language_dialog()
    if result[0] is None:
        sys.exit(0)
    source_lang, target_lang, use_large_model, llm_provider = result
    source_lang = source_lang or "zh"
    target_lang = target_lang or "en"

    region = None
    region_selector = None
    if "--no-select" not in sys.argv:
        print("Position the red frame over subtitles. Enter to confirm, Esc to cancel. Frame stays for repositioning.")
        region, region_selector = select_region(app)
        if not region or region["width"] < 10 or region["height"] < 10:
            print("Invalid selection. Using default.")
            region = None
            region_selector = None

    screen = app.primaryScreen().geometry()
    screen_h, screen_w = screen.height(), screen.width()

    if region is None:
        region = {
            "left": max(0, (screen_w - 800) // 2),
            "top": screen_h - 150,
            "width": 800,
            "height": 120,
        }

    debug = "--debug" in sys.argv

    overlay_h = 150
    overlay_below = region["top"] + region["height"] + 10
    overlay_above = region["top"] - overlay_h - 10
    if overlay_below + overlay_h <= screen_h:
        overlay_top = overlay_below
    elif overlay_above >= 0:
        overlay_top = overlay_above
    else:
        overlay_top = max(0, screen_h - overlay_h - 20)

    overlay = SubtitleOverlay(
        left=region["left"],
        top=overlay_top,
        width=region["width"],
        height=overlay_h,
    )
    hint = QLabel("*Move the translation outside the OCR area to prevent flickering.")
    hint.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
    hint.setAttribute(Qt.WA_TranslucentBackground)
    hint.setStyleSheet("color: rgba(255,255,255,0.9); font-size: 11px;")
    hint.adjustSize()
    # To move the hint horizontally, change the first argument (X).
    # To move it vertically, change the second argument (Y).
    # Example: Increase "- 20" to move hint further left, or decrease to move right.
    #          Increase "- 10" to move hint further up, or decrease to move down.
    hint.move(
        overlay.x() + overlay.width() - hint.width() - 100,  # X: increase to move left
        overlay.y() - hint.height() + 5                      # Y: increase to move down
    )
    overlay._hint = hint
    region_ref = region.copy()
    if region_selector:
        region_selector._region_ref = region_ref

        def on_region_changed(new_region):
            overlay.resize(new_region["width"], overlay.height())

        region_selector.region_changed.connect(on_region_changed)

    def quit_all():
        overlay.close()
        if region_selector:
            region_selector.close()
        app.quit()

    overlay._quit_all = quit_all
    if region_selector:
        region_selector._quit_all = quit_all

    translator = TranslatorApp(
        region_ref, overlay, debug=debug, region_selector=region_selector,
        source_lang=source_lang, target_lang=target_lang,
        use_large_model=use_large_model, llm_provider=llm_provider,
    )

    overlay.show()
    overlay.setFocus()
    overlay.destroyed.connect(app.quit)
    translator.start_threads()

    def bring_front():
        overlay.raise_()
        hint.raise_()
        overlay.activateWindow()
        _mac_set_fullscreen_overlay(overlay)
        _mac_set_fullscreen_overlay(hint)
    QTimer.singleShot(300, bring_front)

    timer = QTimer()
    timer.timeout.connect(translator.ui_update)
    timer.start(100)  # 10 Hz UI refresh

    print(f"Translating region ({region['width']}x{region['height']}) at ({region['left']}, {region['top']}). Drag frame to reposition. Esc to quit.")
    if debug:
        print("Debug: Watch terminal for [OCR], [Translation], [Overlay] messages.")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()