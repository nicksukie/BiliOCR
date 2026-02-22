import sys
import os


import numpy as np
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtCore import QObject, pyqtSignal

from audio_capture import AudioCapture
from audio_transcriber import Transcriber

class AudioPipelineSignals(QObject):
    """Signals for audio pipeline events"""
    transcription_received = pyqtSignal(str)  # Raw transcribed text
    translation_received = pyqtSignal(str)    # Translated text
    error_occurred = pyqtSignal(str)          # Error messages
    no_audio_detected = pyqtSignal()          # No audio received for >10s

class AudioPipeline:
    """Audio processing pipeline that captures, transcribes, and translates audio"""
    
    def __init__(self, translator_callback, device_index=None, asr_backend="whisper", 
                 model_size="base", language="zh", sample_rate=16000,
                 silence_threshold=0.005, silence_duration=1.0, status_callback=None):
        """
        Initialize audio pipeline
        
        Args:
            translator_callback: Function that takes text string and returns translated string
            device_index: Audio input device index
            asr_backend: ASR backend to use (whisper, funasr, mlx)
            model_size: ASR model size
            language: Source language code
            sample_rate: Audio sample rate
            silence_threshold: RMS threshold for silence detection
            silence_duration: Seconds of silence before cutting phrase
            status_callback: Optional callback function(status_message, duration_sec, is_good_news) for status messages
        """
        self.signals = AudioPipelineSignals()
        self.translator_callback = translator_callback
        self.status_callback = status_callback
        self.running = False
        self.thread = None
        
        # Initialize components
        self.audio_capture = AudioCapture(
            device_index=device_index,
            sample_rate=sample_rate,
            silence_threshold=silence_threshold,
            silence_duration=silence_duration,
            max_phrase_duration=30.0,
            streaming_mode=True,
            streaming_step_size=0.2
        )
        
        # Initialize Transcriber
        print(f"[AudioPipeline] Initializing Transcriber with backend={asr_backend}")
        self.transcriber = Transcriber(
            backend=asr_backend,
            model_size=model_size,
            device="auto",
            compute_type="float16" if asr_backend != "mlx" else "int8",
            language=language,
            status_callback=self.status_callback
        )
        
        # Warmup model
        self.transcriber.warmup()
        
        # State
        self.buffer = np.array([], dtype=np.float32)
        self.last_final_text = ""
        self.translation_executor = ThreadPoolExecutor(max_workers=2)
        
    def start(self):
        """Start the audio processing pipeline"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        print("[AudioPipeline] Started")
        
    def stop(self):
        """Stop the audio processing pipeline"""
        if not self.running:
            return
            
        print("[AudioPipeline] Stopping...")
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.audio_capture.stop()
        self.translation_executor.shutdown(wait=False)
        print("[AudioPipeline] Stopped")
        
    def _processing_loop(self):
        """Main audio processing loop"""
        print("[AudioPipeline] Processing loop started")
        
        buffer_duration = 0.0
        last_update_time = time.time()
        phrase_start_time = time.time()
        
        # Generator yielding small audio chunks
        audio_gen = self.audio_capture.generator()
        
        try:
            for audio_chunk in audio_gen:
                if not self.running:
                    break
                    
                self.buffer = np.concatenate([self.buffer, audio_chunk])
                now = time.time()
                buffer_duration = len(self.buffer) / self.audio_capture.sample_rate
                
                # Check silence for finalization
                is_silence = False
                min_silence_dur = self.audio_capture.silence_duration
                
                if buffer_duration > min_silence_dur:
                    # Check tail of silence duration
                    tail = self.buffer[-int(self.audio_capture.sample_rate * min_silence_dur):]
                    rms = np.sqrt(np.mean(tail**2))
                    if rms < self.audio_capture.silence_threshold:
                        is_silence = True
                        
                # VAD Logic
                standard_cut = (is_silence and buffer_duration > 2.0)
                soft_limit_cut = False
                
                if buffer_duration > 6.0:
                    # Check shorter silence tail (0.4s)
                    short_tail_samps = int(self.audio_capture.sample_rate * 0.4)
                    if len(self.buffer) > short_tail_samps:
                        t_rms = np.sqrt(np.mean(self.buffer[-short_tail_samps:]**2))
                        if t_rms < self.audio_capture.silence_threshold:
                            soft_limit_cut = True
                            
                hard_limit_cut = (buffer_duration > self.audio_capture.max_phrase_duration)
                should_finalize = standard_cut or soft_limit_cut or hard_limit_cut
                
                if should_finalize and buffer_duration > 0.5:
                    # Finalize this chunk
                    final_buffer = self.buffer.copy()
                    
                    # Check if buffer is not pure silence
                    overall_rms = np.sqrt(np.mean(final_buffer**2))
                    if overall_rms < self.audio_capture.silence_threshold:
                        print(f"[AudioPipeline] Skipped silent chunk ({buffer_duration:.2f}s)")
                    else:
                        # Process in background thread
                        self.translation_executor.submit(
                            self._process_final_chunk, 
                            final_buffer, 
                            self.last_final_text
                        )
                    
                    # Reset buffer
                    self.buffer = np.array([], dtype=np.float32)
                    last_update_time = now
                    
        except Exception as e:
            print(f"[AudioPipeline] Processing error: {e}")
            self.signals.error_occurred.emit(f"Audio processing error: {str(e)}")
            
        print("[AudioPipeline] Processing loop exited")
        
    def _process_final_chunk(self, audio_buffer, prompt):
        """Process a final audio chunk: transcribe -> translate"""
        try:
            # Transcribe
            transcription = self.transcriber.transcribe(audio_buffer, prompt=prompt)
            if not transcription or len(transcription.strip()) < 2:
                return
                
            print(f"[AudioPipeline] Transcribed: {transcription[:100]}...")
            self.signals.transcription_received.emit(transcription)
            
            # Translate using the main app's translation system
            if self.translator_callback:
                translation = self.translator_callback(transcription)
                if translation:
                    print(f"[AudioPipeline] Translated: {translation[:100]}...")
                    self.signals.translation_received.emit(translation)
                    
            # Update context
            self.last_final_text = transcription
            
        except Exception as e:
            print(f"[AudioPipeline] Chunk processing error: {e}")