import mss
import numpy as np
import cv2
import easyocr
import time
import json
from PIL import Image
import hashlib
from datetime import datetime

class SubtitleTranslator:
    def __init__(self):
        # Configure your subtitle region (top-left x, y, width, height)
        # You'll calibrate this later
        self.region = {"top": 900, "left": 400, "width": 800, "height": 120}
        self.threshold = 5.0  # Pixel difference threshold
        self.last_hash = None
        self.last_text = ""
        self.stable_count = 0
        
        # Initialize EasyOCR (first run downloads models)
        print("Loading OCR model...")
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)  # Use gpu=True if you have CUDA
        
    def capture_region(self):
        """Capture specific screen region"""
        with mss.mss() as sct:
            screenshot = np.array(sct.grab(self.region))
            return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    
    def has_changed(self, img):
        """Check if image changed significantly using perceptual hash"""
        # Resize for faster comparison
        small = cv2.resize(img, (64, 16))
        # Simple mean hash
        avg = np.mean(small)
        hash_val = "".join(['1' if pixel > avg else '0' for pixel in small.flatten()])
        
        if hash_val != self.last_hash:
            self.last_hash = hash_val
            return True
        return False
    
    def extract_text(self, img):
        """OCR the image"""
        results = self.reader.readtext(img, detail=0, paragraph=True)
        return " ".join(results).strip()
    
    def translate(self, text):
        """Stub - implement with DeepL/OpenAI"""
        # TODO: Add your translation API here
        return f"[EN] {text}"
    
    def run(self):
        print("Starting capture... Press Ctrl+C to stop")
        prev_img = None
        
        while True:
            try:
                img = self.capture_region()
                
                # Only OCR if changed
                if self.has_changed(img):
                    text = self.extract_text(img)
                    
                    # Debounce: wait for text to stabilize (avoid mid-sentence captures)
                    if text == self.last_text:
                        self.stable_count += 1
                        if self.stable_count == 2 and text:  # Stable for 2 frames
                            translation = self.translate(text)
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] {translation}")
                    else:
                        self.stable_count = 0
                        self.last_text = text
                
                time.sleep(0.3)  # 3-4 FPS is enough for subtitles
                
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    app = SubtitleTranslator()
    app.run()