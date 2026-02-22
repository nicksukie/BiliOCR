"""
Streaming text reconciler for incremental segmentation.
Implements stable/unstable buffer algorithm to prevent disjointed translations.
Uses stability over time only—no punctuation-based logic.
"""
import re
import time
from difflib import SequenceMatcher


def _count_words(text):
    """Count words: CJK chars + Latin tokens. Handles mixed text."""
    if not text or not text.strip():
        return 0
    cjk = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))
    latin = len(re.findall(r'[a-zA-Z]+', text))
    return cjk + latin


class StreamingReconciler:
    """
    Maintains stable and unstable text buffers to prevent translating incomplete sentences.
    Based on the industry-standard approach for live subtitling.
    """
    
    def __init__(self, stability_threshold=0.4, debug=False):
        """
        Args:
            stability_threshold: Seconds of stability before committing text (default 0.4s for responsiveness)
            debug: Enable debug logging
        """
        self.stability_threshold = stability_threshold
        self.debug = debug
        
        # Buffers
        self.stable_buffer = []  # List of finalized sentences/chunks
        self.unstable_buffer = ""  # Current incomplete sentence being OCR'd
        self.last_unstable = ""  # Last OCR result for comparison
        
        # Timing
        self.last_change_time = time.time()
        self.stability_start_time = None
        self.unstable_buffer_start_time = None  # When unstable buffer was first created
        
        # Translation cache to avoid re-translating stable text
        self.translated_stable = []  # Translated versions of stable_buffer
        
    def ingest(self, new_text):
        """
        Process new OCR text frame.
        
        Args:
            new_text: Raw OCR result from current frame
            
        Returns:
            tuple: (should_translate, text_to_translate, is_final)
                - should_translate: Whether to trigger translation now
                - text_to_translate: The text unit to translate
                - is_final: Whether this is a finalized sentence (vs partial)
        """
        if not new_text or not new_text.strip():
            return False, None, False
            
        now = time.time()
        new_text = new_text.strip()
        
        # Case 1: Text is identical to last frame - it might be stabilizing
        if new_text == self.last_unstable:
            if self.stability_start_time is None:
                self.stability_start_time = now
                if self.debug:
                    print(f"[Reconciler] Text stabilized, starting timer")
            
            elapsed_stable = now - self.stability_start_time
            
            # Check if we've been stable long enough
            if elapsed_stable >= self.stability_threshold:
                # Text has been stable - check if we should commit
                # Update unstable buffer to match current stable text
                if self.unstable_buffer != new_text:
                    self.unstable_buffer = new_text
                    if self.debug:
                        print(f"[Reconciler] Updated unstable_buffer to match stable text")
                
                if self.debug:
                    print(f"[Reconciler] Stable for {elapsed_stable:.2f}s, threshold={self.stability_threshold}")
                text_to_commit = self.unstable_buffer
                if self.debug:
                    print(f"[Reconciler] COMMITTING: '{text_to_commit}'")
                self._commit_unstable()
                return True, text_to_commit, True
        else:
            # Text changed - update unstable buffer
            old_buffer = self.unstable_buffer
            was_empty = not self.unstable_buffer
            self.unstable_buffer = self._merge_with_overlap(self.unstable_buffer, new_text)
            self.last_change_time = now
            self.stability_start_time = None  # Reset stability timer
            
            # Track when unstable buffer was first created
            if was_empty and self.unstable_buffer:
                self.unstable_buffer_start_time = now
            
            # Reduced debug output
            # if self.debug and old_buffer != self.unstable_buffer:
            #     print(f"[Reconciler] Text changed, merged")
            
        self.last_unstable = new_text
        
        # Timeout: text has been in buffer long enough - send it (stability = full sentence captured)
        max_wait_time = self.stability_threshold * 2.0
        if self.unstable_buffer and self.unstable_buffer_start_time:
            time_in_buffer = now - self.unstable_buffer_start_time
            if time_in_buffer >= max_wait_time:
                text_to_translate = self.unstable_buffer
                if self.debug:
                    print(f"[Reconciler] TIMEOUT - translating after {time_in_buffer:.2f}s: '{text_to_translate}'")
                self._commit_unstable()
                return True, text_to_translate, True
        
        # Also check: if text has been stable for a shorter time but buffer is substantial, translate
        if self.unstable_buffer and new_text == self.last_unstable:
            if self.stability_start_time:
                elapsed_stable = now - self.stability_start_time
                # If stable for at least 0.2s and buffer is substantial, translate
                if elapsed_stable >= 0.2 and len(self.unstable_buffer) >= 6:
                    if elapsed_stable >= self.stability_threshold:
                        if self.debug:
                            print(f"[Reconciler] Early commit: stable for {elapsed_stable:.2f}s, buffer length {len(self.unstable_buffer)}")
                        text_to_translate = self.unstable_buffer
                        self._commit_unstable()
                        return True, text_to_translate, True
        
        # Reduced debug output to prevent OCR capture
        # if self.debug:
        #     print(f"[Reconciler] No translation trigger yet (stable={new_text == self.last_unstable}, buffer='{self.unstable_buffer[:50] if self.unstable_buffer else ''}')")
        
        # Return partial text for display (optional - can be None if you don't want partials)
        return False, None, False
    
    def _merge_with_overlap(self, old, new):
        """
        Handle mid-sentence updates intelligently.
        Example: "Hello wor" -> "Hello world" (find overlap and merge)
        
        Args:
            old: Previous unstable buffer
            new: New OCR text
            
        Returns:
            Merged text
        """
        if not old:
            return new
        
        # Correction: same sentence, different OCR variant - replace, don't concatenate
        len_ratio = len(new) / max(1, len(old))
        if 0.6 <= len_ratio <= 1.5:
            matcher = SequenceMatcher(None, old, new, autojunk=False)
            if matcher.ratio() >= 0.5:
                return new
        
        # Common case: new text is a continuation (starts with old)
        if new.startswith(old):
            return new
        
        # Common case: new text contains old (progressive reveal)
        if old in new:
            return new
        
        # Check if new continues from the end of old (overlap at boundaries)
        # Try suffixes of old matching prefixes of new
        min_overlap = 2  # Minimum overlap to consider
        max_check = min(len(old), len(new), 20)  # Don't check too far back
        
        for i in range(max_check, min_overlap - 1, -1):
            if old[-i:] == new[:i]:
                # Found overlap - merge
                return old + new[i:]
        
        # Use difflib to find the best overlap (for corrections)
        matcher = SequenceMatcher(None, old, new, autojunk=False)
        match = matcher.find_longest_match(0, len(old), 0, len(new))
        
        if match.size >= min_overlap:
            overlap_start_old = match.a
            overlap_start_new = match.b
            
            # If overlap is at the end of old and start of new, it's a continuation
            if overlap_start_old + match.size == len(old):
                # New text continues from old
                return old + new[overlap_start_new + match.size:]
            elif overlap_start_old == 0 and overlap_start_new == 0:
                # New text replaces old (complete rewrite)
                return new
            else:
                # Partial overlap in middle - prefer new text but keep prefix
                return old[:overlap_start_old] + new
        
        # No clear overlap - new might be a correction or completely different
        # If new is significantly longer, assume it's a correction
        if len(new) > len(old) * 0.7:
            return new
        
        # If old is much longer, keep old (might be OCR error)
        if len(old) > len(new) * 1.5:
            return old
        
        # Last resort: return new (assume it's a correction)
        return new
    
    def _commit_unstable(self):
        """Move unstable buffer to stable buffer."""
        if self.unstable_buffer:
            self.stable_buffer.append(self.unstable_buffer)
            # Keep only last 5 sentences for context
            if len(self.stable_buffer) > 5:
                self.stable_buffer.pop(0)
            self.unstable_buffer = ""
            self.stability_start_time = None
            self.unstable_buffer_start_time = None
    
    def get_current_text(self):
        """
        Get current display text (stable + unstable).
        
        Returns:
            str: Combined stable and unstable text
        """
        stable_text = " ".join(self.stable_buffer) if self.stable_buffer else ""
        if stable_text and self.unstable_buffer:
            return stable_text + " " + self.unstable_buffer
        elif self.unstable_buffer:
            return self.unstable_buffer
        else:
            return stable_text

    def reset(self):
        """Reset all buffers (e.g., when switching videos)."""
        self.stable_buffer = []
        self.unstable_buffer = ""
        self.last_unstable = ""
        self.last_change_time = time.time()
        self.stability_start_time = None
        self.unstable_buffer_start_time = None
        self.translated_stable = []


class LLMReconciler:
    """
    Reconciler optimized for LLM translation. Semantics only—no punctuation logic.
    LLMs can read larger chunks; we accumulate OCR text and send when stable.
    """
    def __init__(self, stability_threshold=0.12, max_buffer_time=0.6, debug=False):
        """
        Args:
            stability_threshold: Seconds of stability before committing
            max_buffer_time: Send buffered text after this many seconds (timeout)
            debug: Enable debug logging
        """
        self.stability_threshold = stability_threshold
        self.max_buffer_time = max_buffer_time
        self.debug = debug
        self.buffer = ""
        self.last_frame = ""
        self.stability_start = None
        self.buffer_start_time = None

    def ingest(self, new_text):
        """
        Process new OCR frame. Accumulate, send when stable or timeout.
        Returns:
            tuple: (should_translate, text_to_translate, is_final)
        """
        if not new_text or not new_text.strip():
            return False, None, False

        now = time.time()
        new_text = new_text.strip()
        merged = self._merge(new_text)
        self.last_frame = new_text

        if merged and not self.buffer:
            self.buffer_start_time = now

        # Text changed
        if merged != self.buffer:
            self.stability_start = now

        self.buffer = merged

        # Stability: text has been unchanged long enough
        elapsed = now - (self.stability_start or now)
        if elapsed >= self.stability_threshold and merged:
            if self.debug:
                print(f"[LLM Reconciler] Stable send: '{merged[:50]}...'")
            self.buffer = ""
            self.buffer_start_time = None
            self.stability_start = None
            return True, merged.strip(), True

        # Timeout: send anyway after max_buffer_time
        if merged and len(merged.strip()) >= 2 and self.buffer_start_time:
            if now - self.buffer_start_time >= self.max_buffer_time:
                if merged == new_text:
                    if self.debug:
                        print(f"[LLM Reconciler] Timeout send: '{merged[:50]}...'")
                    self.buffer = ""
                    self.buffer_start_time = None
                    return True, merged.strip(), True

        return False, None, False

    def _merge(self, new_text):
        """Merge new OCR frame with buffer (continuation, overlap, or replacement)."""
        if not self.buffer:
            return new_text
        old, new = self.buffer, new_text

        # Same sentence, OCR correction—replace
        if 0.6 <= len(new) / max(1, len(old)) <= 1.5:
            matcher = SequenceMatcher(None, old, new, autojunk=False)
            if matcher.ratio() >= 0.5:
                return new

        if new.startswith(old):
            return new
        if old in new:
            return new

        # Overlap at boundary
        for i in range(min(len(old), len(new), 25), 1, -1):
            if old[-i:] == new[:i]:
                return old + new[i:]

        # New continues or replaces
        matcher = SequenceMatcher(None, old, new, autojunk=False)
        match = matcher.find_longest_match(0, len(old), 0, len(new))
        if match.size >= 2 and match.a + match.size == len(old):
            return old + new[match.b + match.size:]
        if len(new) > len(old) * 0.7:
            return new
        return old + " " + new

    def reset(self):
        """Reset buffers."""
        self.buffer = ""
        self.last_frame = ""
        self.stability_start = None


class AudioReconciler:
    """
    Reconciler optimized for audio transcription.
    Within X seconds, checks Y times for sentence completion. Whichever happens first:
    - Sentence completes (punctuation) -> send (if words >= min_words)
    - X seconds pass or Y checks done -> send anyway (if words >= min_words)
    All output is final (white). Once sent, buffer is discarded.
    """
    # Sentence-ending punctuation (CJK + Latin) - not comma, only true sentence enders
    SENTENCE_ENDINGS = frozenset('.!?。！？')

    def __init__(self, period_sec=2.0, num_checks=4, min_words=7, debug=False):
        """
        Args:
            period_sec: Max seconds before forcing send (X)
            num_checks: Number of completion checks in that period (Y)
            min_words: Minimum word count before sending
            debug: Enable debug logging
        """
        self.period_sec = period_sec
        self.num_checks = num_checks
        self.min_words = min_words
        self.debug = debug
        self.buffer = ""
        self.period_start = None
        self.check_count = 0

    def ingest(self, transcribed_text):
        """
        Process new transcription. Called each time we transcribe.
        Returns:
            tuple: (should_send, text_to_send, is_final)
            Always is_final=True when should_send.
        """
        if not transcribed_text or not transcribed_text.strip():
            return False, None, True

        now = time.time()
        text = transcribed_text.strip()
        word_count = _count_words(text)

        if self.period_start is None:
            self.period_start = now

        self.buffer = text
        self.check_count += 1
        elapsed = now - self.period_start

        # Must have minimum words before sending
        if word_count < self.min_words:
            return False, None, True

        # Check 1: Sentence complete (ends with punctuation)
        if self._is_sentence_complete(text):
            if self.debug:
                print(f"[AudioReconciler] Sentence complete after {self.check_count} checks ({word_count} words): '{text[:50]}...'")
            out = self.buffer
            self._reset()
            return True, out, True

        # Check 2: Period expired or max checks reached
        if elapsed >= self.period_sec or self.check_count >= self.num_checks:
            if self.debug:
                print(f"[AudioReconciler] Timeout ({elapsed:.1f}s, {self.check_count} checks, {word_count} words): '{text[:50]}...'")
            out = self.buffer
            self._reset()
            return True, out, True

        return False, None, True

    def _is_sentence_complete(self, text):
        """True if text ends with sentence-ending punctuation."""
        if not text:
            return False
        return text[-1] in self.SENTENCE_ENDINGS

    def _reset(self):
        """Discard buffer after send."""
        self.buffer = ""
        self.period_start = None
        self.check_count = 0

    def reset(self):
        """Reset state (e.g. new phrase)."""
        self._reset()
