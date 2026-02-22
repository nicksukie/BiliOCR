"""OCR provider abstraction supporting multiple backends."""
import numpy as np
import base64
import io
import os
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import requests
except ImportError:
    requests = None


class OCRProvider:
    """Base class for OCR providers."""
    
    def process(self, image_np, return_boxes=False):
        """
        Process image and return OCR results.
        
        Args:
            image_np: numpy array (H, W, 3) RGB
            return_boxes: if True, return bounding boxes
            
        Returns:
            If return_boxes=False: (text, candidates_list)
            If return_boxes=True: (text, candidates_list, boxes_list)
        """
        raise NotImplementedError


class VisionOCRProvider(OCRProvider):
    """Apple Vision framework OCR - Mac native, fast, no external models."""
    
    def __init__(self, languages=None):
        try:
            from vision_ocr import VisionOCR
            self.ocr = VisionOCR(languages=languages)
        except ImportError:
            raise ImportError("VisionOCR requires macOS and Vision framework")
    
    def process(self, image_np, return_boxes=False):
        return self.ocr.process(image_np, return_boxes=return_boxes)


class EasyOCRProvider(OCRProvider):
    """EasyOCR - small local model, supports many languages."""
    
    def __init__(self, languages=None):
        try:
            import easyocr
            # Default to Chinese and English if not specified
            langs = languages or ['ch_sim', 'en']
            self.reader = easyocr.Reader(langs, gpu=False)  # Use gpu=True if CUDA available
            print("[EasyOCR] Model loaded successfully")
        except ImportError:
            raise ImportError("EasyOCR not installed. Install with: pip install easyocr")
        except Exception as e:
            print(f"[EasyOCR] Failed to initialize: {e}")
            raise
    
    def process(self, image_np, return_boxes=False):
        if image_np is None or image_np.size == 0:
            return ("", [], []) if return_boxes else ("", [])
        
        try:
            # EasyOCR expects BGR format
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                # Assume RGB, convert to BGR
                image_bgr = image_np[:, :, ::-1].copy()
            else:
                image_bgr = image_np
            
            results = self.reader.readtext(image_bgr, detail=1 if return_boxes else 0)
            
            if return_boxes:
                text_parts = []
                candidates = []
                boxes = []
                for detection in results:
                    if isinstance(detection, tuple) and len(detection) >= 2:
                        bbox, text, confidence = detection[0], detection[1], detection[2] if len(detection) > 2 else 1.0
                        if confidence > 0.5:
                            text_parts.append(text)
                            candidates.append([text])  # EasyOCR doesn't provide multiple candidates
                            # bbox is list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                            if bbox:
                                y_coords = [p[1] for p in bbox]
                                boxes.append((int(min(y_coords)), int(max(y_coords))))
                text = " ".join(text_parts).strip()
                return text, candidates, boxes
            else:
                text_parts = []
                candidates = []
                for detection in results:
                    if isinstance(detection, str):
                        text_parts.append(detection)
                        candidates.append([detection])
                    elif isinstance(detection, tuple) and len(detection) >= 2:
                        text = detection[1]
                        confidence = detection[2] if len(detection) > 2 else 1.0
                        if confidence > 0.5:
                            text_parts.append(text)
                            candidates.append([text])
                text = " ".join(text_parts).strip()
                return text, candidates
        except Exception as e:
            print(f"[EasyOCR] Error during processing: {e}")
            return ("", [], []) if return_boxes else ("", [])


def create_ocr_provider(backend="vision", languages=None, **kwargs):
    """
    Factory function to create OCR provider.
    
    Args:
        backend: "vision" (default), "easyocr"
        languages: list of language codes (for vision/easyocr)
        **kwargs: additional provider-specific arguments
    
    Returns:
        OCRProvider instance
    """
    backend = backend.lower()
    
    if backend == "vision":
        return VisionOCRProvider(languages=languages)
    elif backend == "easyocr":
        return EasyOCRProvider(languages=languages)
    else:
        raise ValueError(f"Unknown OCR backend: {backend}")
