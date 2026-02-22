"""Screen capture: Quartz (native) or mss fallback."""
import os
import numpy as np


def _get_primary_display_bounds():
    """Get primary display dimensions (width, height)."""
    try:
        import Quartz
        bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        return int(bounds.size.width), int(bounds.size.height)
    except Exception:
        try:
            import mss
            with mss.mss() as sct:
                m = sct.monitors[0]
                return m["width"], m["height"]
        except Exception:
            return 1920, 1080


class MSSCapture:
    """Fallback using mss - avoids Quartz, more stable on some Macs."""

    def __init__(self, left, top, width, height):
        self.region = {"left": left, "top": top, "width": width, "height": height}

    def capture(self):
        try:
            import mss
            with mss.mss() as sct:
                frame = np.array(sct.grab(self.region))
                return np.ascontiguousarray(frame[:, :, [2, 1, 0]])  # BGRA -> RGB
        except Exception:
            return None


def create_capture(left, top, width, height, debug=False):
    """Use mss if USE_MSS=1 (avoids Quartz crashes), else native Quartz."""
    if os.environ.get("USE_MSS") == "1":
        return MSSCapture(left, top, width, height)
    return MacOSCapture(left, top, width, height, debug=debug)


class DynamicRegionCapture:
    """Capture that reads region from a getter - for live repositioning."""

    def __init__(self, get_region, debug=False):
        self.get_region = get_region
        self._capture = None
        self._last_key = None
        self._debug = debug

    def capture(self):
        r = self.get_region()
        if r is None:
            return None
        key = (r.get("left"), r.get("top"), r.get("width"), r.get("height"))
        if key != self._last_key:
            self._last_key = key
            self._capture = create_capture(
                r["left"], r["top"], r["width"], r["height"],
                debug=self._debug,
            )
        return self._capture.capture()


class MacOSCapture:
    """Captures a screen region. Accepts (left, top, width, height) in top-left coords."""

    def __init__(self, left, top, width, height, debug=False):
        import Quartz
        screen_w, screen_h = _get_primary_display_bounds()
        y_quartz = screen_h - top - height
        self.rect = Quartz.CGRectMake(left, y_quartz, width, height)
        self.width = width
        self.height = height

    def capture(self):
        import Quartz
        try:
            image = Quartz.CGWindowListCreateImage(
                self.rect,
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID,
                Quartz.kCGWindowImageDefault,
            )

            if image is None:
                return None

            w = Quartz.CGImageGetWidth(image)
            h = Quartz.CGImageGetHeight(image)
            bpr = Quartz.CGImageGetBytesPerRow(image)
            provider = Quartz.CGImageGetDataProvider(image)
            data = Quartz.CGDataProviderCopyData(provider)
            if data is None or len(data) == 0:
                return None
            # Copy to numpy immediately - avoid vm_copy issues from CFData lifecycle
            arr = np.copy(np.frombuffer(data, dtype=np.uint8))
            del data  # Release CFData
            pixels = arr.reshape(h, bpr // 4, 4)
            return np.ascontiguousarray(pixels[:, :w, 1:4].copy())  # RGB only
        except Exception:
            return None
