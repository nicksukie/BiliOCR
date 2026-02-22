"""
Extract keywords from Chinese text: word segmentation, pinyin, definitions.
Uses conventional methods: jieba (TF-IDF), pypinyin, CEDICT dictionary.
Supports multiple dictionary sources for better coverage.
Extracts: common words (nouns, verbs, adjectives, adverbs), proper nouns (person/place names), and chengyu (4-character idioms).
"""
import re
import jieba
import jieba.analyse
import jieba.posseg
import pypinyin
from pypinyin import Style
from cepy_dict import entries

# Try to import additional dictionary sources
try:
    from pycccedict.cccedict import CcCedict
    HAS_PYCCCEDICT = True
except ImportError:
    HAS_PYCCCEDICT = False
    CcCedict = None


class KeywordExtractor:
    """Conventional Chinese keyword extraction using TF-IDF, pinyin, and CEDICT."""

    def __init__(self):
        """Initialize CEDICT and default settings."""
        # CEDICT from cepy-dict returns (entry_text, traditional, simplified, pinyin, definitions)
        # definitions can be a list or string - convert to string
        # Map simplified to (simplified, pinyin, definitions_string)
        self.cedict = {}
        # Build traditional-to-simplified character mapping for conversion
        self.trad_to_simp = {}
        for entry_text, trad, simp, py, defn in entries():
            # Convert definitions to string if it's a list
            if isinstance(defn, list):
                defn_str = "; ".join(defn) if defn else ""
            else:
                defn_str = str(defn) if defn else ""
            # Store as (simplified, pinyin, definition) - use simplified as key and value
            self.cedict[simp] = (simp, py, defn_str)
            # Build character-level traditional-to-simplified mapping
            if trad != simp:
                for trad_char, simp_char in zip(trad, simp):
                    if trad_char != simp_char:
                        self.trad_to_simp[trad_char] = simp_char
        
        # Initialize additional dictionary sources if available
        self.pycccedict_dict = None
        if HAS_PYCCCEDICT and CcCedict:
            try:
                self.pycccedict_dict = CcCedict()
            except Exception:
                pass  # Fallback to CEDICT only

    def _convert_traditional_to_simplified(self, text: str) -> str:
        """Convert traditional Chinese characters to simplified using CEDICT mapping."""
        if not text:
            return text
        result = []
        for char in text:
            # Convert if mapping exists, otherwise keep original
            result.append(self.trad_to_simp.get(char, char))
        return ''.join(result)

    def extract_keywords(self, text: str, target_lang: str, translate_word_fn=None) -> list[dict]:
        """
        Extract keywords from Chinese text.
        Returns: [{word, pinyin, definition}, ...]
        translate_word_fn: optional callable(word) -> str. When word not in CEDICT, uses this
            to get definition (e.g. LLM/MT translation). If None, uses _fallback_definition.
        """
        # Skip if no Chinese
        if not self._has_chinese(text):
            return []

        # Convert traditional Chinese to simplified before processing
        text = self._convert_traditional_to_simplified(text)

        # Extract keywords using multiple methods:
        # 1. TF-IDF for common nouns, verbs, adjectives, adverbs
        # 2. Proper nouns (person names nr, place names ns, other proper nouns nz)
        # 3. Chengyu (4-character idioms)
        
        all_keywords = []
        
        # Method 1: TF-IDF extraction for common words
        # POS tags: n=noun, v=verb, a=adjective, d=adverb
        tfidf_keywords = jieba.analyse.extract_tags(text, topK=8, withWeight=False, allowPOS=('n', 'v', 'a', 'd'))
        all_keywords.extend(tfidf_keywords)
        
        # Method 2: Extract proper nouns using jieba POS tagging
        # POS tags: nr=person name, ns=place name, nz=other proper noun
        words_with_pos = jieba.posseg.cut(text)
        proper_nouns = []
        for word, flag in words_with_pos:
            if flag in ('nr', 'ns', 'nz') and len(word) >= 2:
                proper_nouns.append(word)
        all_keywords.extend(proper_nouns)
        
        # Method 3: Extract chengyu (4-character idioms)
        # Chengyu are usually 4 characters and often appear as fixed phrases
        # Use jieba to find 4-character words that are recognized as single units
        words_with_pos = list(jieba.posseg.cut(text))
        chengyu_found = set()
        for i in range(len(words_with_pos) - 3):
            # Check for consecutive 4-character sequences
            if i + 3 < len(words_with_pos):
                # Try to find 4-character sequences
                potential_chengyu = ''.join([words_with_pos[j].word for j in range(i, min(i+4, len(words_with_pos)))])
                if len(potential_chengyu) == 4 and self._is_chinese_only(potential_chengyu):
                    # Check if jieba recognizes this as a single word when we look at the full text
                    # Re-segment to see if this 4-char sequence appears as one word
                    full_seg = list(jieba.cut(text))
                    if potential_chengyu in full_seg:
                        chengyu_found.add(potential_chengyu)
        
        # Also check for standalone 4-character patterns that might be chengyu
        chengyu_pattern = re.compile(r'[\u4e00-\u9fff]{4}')
        for match in chengyu_pattern.finditer(text):
            chengyu = match.group()
            # Only add if it's recognized as a word by jieba
            words_in_text = list(jieba.cut(text))
            if chengyu in words_in_text and chengyu not in chengyu_found:
                chengyu_found.add(chengyu)
        
        all_keywords.extend(list(chengyu_found))
        
        # Filter and deduplicate
        filtered_keywords = []
        seen = set()
        for kw in all_keywords:
            kw = kw.strip()
            if not kw or kw in seen:
                continue
            # Keep words that are 2+ characters, or common single characters
            if len(kw) >= 2:
                filtered_keywords.append(kw)
                seen.add(kw)
            elif len(kw) == 1 and self._is_common_character(kw) and kw not in seen:
                filtered_keywords.append(kw)
                seen.add(kw)
        
        # Limit to top 8 keywords (increased from 5 to accommodate proper nouns and chengyu)
        keywords = filtered_keywords[:8]

        results = []
        for word in keywords:
            # Only use local dictionary (CEDICT) if target language is English
            # For other languages, go straight to LLM translation
            use_local_dict = (target_lang == "en")
            
            if use_local_dict:
                entry = self._lookup_cedict(word)
                if entry:
                    # Found in CEDICT
                    simp, pinyin, definition = entry
                    # Ensure definition is a string (safety check)
                    if isinstance(definition, list):
                        definition = "; ".join(definition) if definition else ""
                    else:
                        definition = str(definition) if definition else ""
                    results.append({
                        "word": simp,  # Use simplified form from CEDICT (not original word which might be traditional)
                        "pinyin": self._normalize_pinyin(pinyin),
                        "definition": definition,
                    })
                    continue  # Found in dictionary, skip LLM lookup
            
            # Not in dictionary (or target language is not English): use LLM/MT translation
            # Ensure word is simplified (should already be after text conversion, but double-check)
            word_simp = self._convert_traditional_to_simplified(word)
            pinyin_list = pypinyin.lazy_pinyin(word_simp, style=Style.TONE)
            pinyin_str = " ".join(pinyin_list)
            if translate_word_fn:
                try:
                    fallback_def = translate_word_fn(word_simp)
                    if not fallback_def or not str(fallback_def).strip():
                        fallback_def = self._fallback_definition(word_simp, target_lang)
                except Exception:
                    fallback_def = self._fallback_definition(word_simp, target_lang)
            else:
                fallback_def = self._fallback_definition(word_simp, target_lang)
            results.append({
                "word": word_simp,  # Use simplified form
                "pinyin": pinyin_str,  # Already has tone symbols from Style.TONE
                "definition": fallback_def,
            })
        return results

    def _has_chinese(self, text: str) -> bool:
        """Check if text contains CJK unified ideographs."""
        return any('\u4e00' <= c <= '\u9fff' for c in text)
    
    def _is_chinese_only(self, text: str) -> bool:
        """Check if text contains only Chinese characters."""
        return all('\u4e00' <= c <= '\u9fff' for c in text) and len(text) > 0

    def _lookup_cedict(self, word: str) -> tuple | None:
        """
        Look up word in dictionary sources (CEDICT, then pycccedict if available).
        Returns (simplified, pinyin, definitions) or None if not found.
        Only returns exact matches - no random character combinations.
        """
        # Try CEDICT first
        if word in self.cedict:
            simp, py, defn = self.cedict[word]
            return (word, py, defn)
        
        # Try pycccedict as fallback if available
        if self.pycccedict_dict:
            try:
                entry = self.pycccedict_dict.get_entry(word)
                if entry:
                    pinyin = entry.get('pinyin', '')
                    definitions = entry.get('definitions', [])
                    defn_str = "; ".join(definitions) if isinstance(definitions, list) else str(definitions)
                    if defn_str:
                        return (word, pinyin, defn_str)
            except Exception:
                pass  # Fallback to None
        
        return None
    
    def _is_common_character(self, char: str) -> bool:
        """Check if a single character is common enough to include."""
        # Common single-character words that are meaningful
        common_chars = {
            '的', '了', '是', '我', '你', '他', '她', '它', '这', '那',
            '在', '有', '和', '就', '不', '人', '都', '来', '到', '说',
            '要', '会', '能', '可以', '好', '很', '也', '还', '又', '只'
        }
        return char in common_chars

    def _normalize_pinyin(self, pinyin: str) -> str:
        """Strip whitespace, convert tone numbers to tone symbols."""
        pinyin = pinyin.strip()
        # Convert tone numbers (1-5) to tone symbols
        # Tone marks: 1=ā, 2=á, 3=ǎ, 4=à, 5=neutral (no mark)
        tone_map = {
            '1': {'a': 'ā', 'e': 'ē', 'i': 'ī', 'o': 'ō', 'u': 'ū', 'ü': 'ǖ', 'v': 'ǖ'},
            '2': {'a': 'á', 'e': 'é', 'i': 'í', 'o': 'ó', 'u': 'ú', 'ü': 'ǘ', 'v': 'ǘ'},
            '3': {'a': 'ǎ', 'e': 'ě', 'i': 'ǐ', 'o': 'ǒ', 'u': 'ǔ', 'ü': 'ǚ', 'v': 'ǚ'},
            '4': {'a': 'à', 'e': 'è', 'i': 'ì', 'o': 'ò', 'u': 'ù', 'ü': 'ǜ', 'v': 'ǜ'},
        }
        
        import re
        # Pattern: syllable with tone number (e.g., "ni3", "hao3", "zhong1")
        def replace_tone(match):
            syllable = match.group(1)
            tone = match.group(2)
            
            if tone == '5' or tone not in tone_map:
                # Neutral tone or invalid - remove number
                return syllable
            
            # Find the vowel to mark (priority: a > e > o > i > u > ü)
            vowels = ['a', 'e', 'o', 'i', 'u', 'ü', 'v']
            for vowel in vowels:
                if vowel in syllable.lower():
                    # Replace the vowel with its tone-marked version
                    if vowel == 'v':  # v is used for ü in some systems
                        vowel = 'ü'
                    tone_marked = tone_map[tone][vowel]
                    syllable = syllable.replace(vowel, tone_marked).replace(vowel.upper(), tone_marked.upper())
                    # Remove the tone number
                    return syllable
            
            # No vowel found (shouldn't happen), just remove tone number
            return syllable
        
        # Replace patterns like "ni3", "hao3", "zhong1" etc.
        pinyin = re.sub(r'([a-züv]+)([1-5])', replace_tone, pinyin, flags=re.IGNORECASE)
        
        return pinyin

    def _fallback_definition(self, word: str, target_lang: str) -> str:
        """
        Fallback definition when word not in CEDICT.
        Returns simple placeholder. Target_lang used for potential future improvement.
        """
        # Simple: use word length and first char meaning as placeholder
        # Could improve by doing: translate(word) if we have fallback text
        if len(word) == 1:
            # Common characters: basic meaning placeholders
            if word in ("你", "您"):
                return "you (fml)"
            if word in ("我", "咱"):
                return "I/my"
            if word == "是":
                return "is"
            if word == "不":
                return "not"
            if word in ("很"):
                return "very"
            if any(char in word for char in ["人", "者", "客"]):
                return "person"
            if "好" in word:
                return "good"
            if "天" in word:
                return "day/sky"
            return "common character"
        elif len(word) == 2:
            if any(char in word for char in ["国", "家"]):
                return "country"
            if any(char in word for char in ["学", "校"]):
                return "school/study"
            if any(char in word for char in ["老", "师"]):
                return "teacher/old"
            return "common word"
        else:
            return "common phrase"


def extract_keywords(text: str, target_lang: str, translate_word_fn=None) -> list[dict]:
    """
    Public function for extraction.
    Returns list of keyword dictionaries: {word, pinyin, definition}.
    translate_word_fn: optional callable(word) -> str for words not in dictionary (LLM/MT).
    """
    extractor = KeywordExtractor()
    return extractor.extract_keywords(text, target_lang, translate_word_fn=translate_word_fn)