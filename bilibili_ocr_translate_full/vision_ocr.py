"""Apple Vision framework OCR - Mac native, no external models."""
import io
import Vision
import Quartz
import numpy as np
from Foundation import NSData
from AppKit import NSBitmapImageRep
from PIL import Image


def _numpy_to_ciimage(arr):
    """Convert numpy array (H, W, 3) RGB to CIImage for Vision framework."""
    # Ensure RGB and contiguous
    if arr.shape[2] == 4:
        pil_img = Image.fromarray(arr[:, :, :3])
    else:
        pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data = buf.getvalue()
    nsdata = NSData.dataWithBytes_length_(data, len(data))
    rep = NSBitmapImageRep.imageRepWithData_(nsdata)
    ciimage = Quartz.CIImage.alloc().initWithBitmapImageRep_(rep)
    return ciimage


class VisionOCR:
    def __init__(self, languages=None):
        self.request = Vision.VNRecognizeTextRequest.alloc().init()
        langs = languages or ["zh-Hans", "en"]
        self.request.setRecognitionLanguages_(langs)
        self.request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        if hasattr(self.request, 'setUsesLanguageCorrection_'):
            self.request.setUsesLanguageCorrection_(True)

    def process(self, image_np, return_boxes=False):
        """Takes numpy array (H, W, 3) RGB, returns recognized text.
        If return_boxes=True, also returns list of (y_top, y_bottom) in pixel coords (Vision uses bottom-left origin)."""
        if image_np is None or image_np.size == 0:
            return ("", [], []) if return_boxes else ("", [])

        h_img, w_img = image_np.shape[:2]
        ciimage = _numpy_to_ciimage(np.ascontiguousarray(image_np))
        options = {}
        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
            ciimage, options
        )

        error = None
        success = handler.performRequests_error_([self.request], None)

        if not success:
            return ("", [], []) if return_boxes else ("", [])

        # Collect top candidate per observation (for pick_best); also build best string
        obs_candidates = []
        results = []
        boxes = []  # (y_top, y_bottom) in pixel coords, top-left origin
        for observation in self.request.results():
            cands = observation.topCandidates_(5)
            if cands:
                best_cand = cands[0]
                if best_cand.confidence() > 0.5:
                    results.append(best_cand.string())
                    obs_candidates.append([c.string() for c in cands if c.confidence() > 0.3])
                    if return_boxes:
                        try:
                            r = observation.boundingBox()
                            # Vision: origin bottom-left, normalized 0-1
                            if hasattr(Quartz, 'CGRectGetMinY'):
                                y_n = Quartz.CGRectGetMinY(r)
                                h_n = Quartz.CGRectGetHeight(r)
                            else:
                                y_n = getattr(getattr(r, 'origin', None), 'y', 0) or 0
                                h_n = getattr(getattr(r, 'size', None), 'height', 0) or 0
                            y_top = (1.0 - float(y_n) - float(h_n)) * h_img
                            y_bot = (1.0 - float(y_n)) * h_img
                            boxes.append((max(0, int(y_top)), min(h_img, int(y_bot))))
                        except Exception:
                            pass
        text = " ".join(results).strip()
        if return_boxes:
            return text, obs_candidates, boxes
        return text, obs_candidates
