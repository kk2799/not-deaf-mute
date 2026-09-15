"""Parse a model's free-form answer into a predicted label.

Used as a fallback / cross-check alongside ``choice_logprobs`` scoring, and for
the error-type analysis (评测 Q5). Strategy, in priority order:
  1. a standalone option letter (``A``..``D``) — most prompts ask for one;
  2. a class-name substring match (English or Chinese);
  3. ``None`` (unparseable → recorded for error analysis).
"""

from __future__ import annotations

import re

from .templates import LETTERS


def parse_answer(
    text: str,
    letters: list[str],
    label_order: list[int],
    label_names: dict[int, str],
) -> tuple[str | None, int | None]:
    """Return ``(letter, label_id)`` or ``(None, None)`` if unparseable."""
    if not text:
        return None, None
    t = text.strip()

    # 1) leading or standalone option letter, e.g. "C", "C. Oil tanker", "Answer: C"
    valid = set(letters)
    m = re.search(r"(?<![A-Za-z])([" + "".join(letters) + r"])(?![A-Za-z])", t)
    if m and m.group(1) in valid:
        letter = m.group(1)
        return letter, label_order[LETTERS.index(letter)]

    # 2) class-name substring (match longer names first to avoid partial hits)
    low = t.lower()
    candidates = sorted(label_names.items(), key=lambda kv: -len(kv[1]))
    for lid, name in candidates:
        if name.lower() in low:
            letter = LETTERS[[k for k, v in enumerate(label_order) if v == lid][0]]
            return letter, lid

    return None, None
