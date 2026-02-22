"""
Lightweight probabilistic OCR correction.
Uses Vision's alternative candidates + jieba word segmentation score to pick the most likely text.
No confusion dictionary - purely probabilistic from OCR candidates and language model.
"""
import itertools
import jieba


def _jieba_score(text: str) -> float:
    """Score by segmentation quality: prefer more/longer multi-char words (known vocabulary)."""
    if not text or not text.strip():
        return 0.0
    try:
        words = jieba.lcut(text.strip())
        total = 0.0
        for w in words:
            if len(w) >= 2:
                total += len(w) ** 1.5
            else:
                total += 0.3
        return total
    except Exception:
        return 0.0


def _has_chinese(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def pick_best(obs_candidates: list, max_combinations: int = 27) -> str:
    """
    From Vision's per-observation candidate lists, pick the combination with highest jieba score.
    obs_candidates: [[c1, c2, c3], [c1, c2]] for 2 observations.
    """
    if not obs_candidates or not any(obs_candidates):
        return ""
    # Limit candidates per obs to avoid explosion
    limited = [c[:5] for c in obs_candidates if c]
    if not limited:
        return ""
    # Build combinations; cap total
    combos = list(itertools.islice(itertools.product(*limited), max_combinations))
    if not combos:
        return limited[0][0] if limited[0] else ""
    best_text = " ".join(combos[0]).strip() if len(combos[0]) > 1 else combos[0][0]
    best_score = _jieba_score(best_text) if _has_chinese(best_text) else 0.0
    for combo in combos[1:]:
        text = " ".join(combo).strip() if len(combo) > 1 else combo[0]
        if not _has_chinese(text):
            continue
        score = _jieba_score(text)
        if score > best_score:
            best_score = score
            best_text = text
    return best_text


def correct(text: str, obs_candidates: list = None) -> str:
    """
    Probabilistic correction. If obs_candidates from Vision, pick best combination.
    Otherwise return text unchanged (no correction without candidates).
    """
    if obs_candidates and len(obs_candidates) > 0 and any(obs_candidates):
        return pick_best(obs_candidates)
    return text or ""
