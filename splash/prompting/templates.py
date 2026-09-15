"""Prompt templates for 评测 A/B (LALM/VLM prompting, zero-shot & few-shot).

Design choices:
* **Lettered multiple-choice** (A/B/C/D …) so ``choice_logprobs`` scores a single
  next token — robust and cheap. The letter↔label order is fixed to the dataset's
  canonical label order by default; per-sample letter shuffling (to neutralise
  position bias) is a planned refinement.
* **Structured HF content** (``{"type":"text"/"audio"/"image"}``) so the same
  template serves audio (评测 B) and spectrogram-image (评测 A) routes — only the
  media item type differs.
* Bilingual (``lang``): English default, Chinese for the 中/英 prompt ablation.
* Optional text enrichment (ShipsEar differentiator) via ``text_enrich``.
"""

from __future__ import annotations

import string

# --- option / label helpers ------------------------------------------------

LETTERS = list(string.ascii_uppercase)


def option_layout(label_names: dict[int, str]) -> tuple[list[str], list[int], str]:
    """Map labels to option letters in canonical label-id order.

    Returns ``(letters, label_order, option_text)`` where:
    * ``letters``      — ["A","B","C",...] (the answer choices to score)
    * ``label_order``  — label ids in the same order (letter i ↔ label_order[i])
    * ``option_text``  — "A. Cargo ship\nB. Passenger ship\n..." for the prompt
    """
    ids = sorted(label_names.keys())
    letters = LETTERS[: len(ids)]
    option_text = "\n".join(f"{l}. {label_names[i]}" for l, i in zip(letters, ids))
    return letters, ids, option_text


def letter_to_label(letter: str, label_order: list[int]) -> int:
    return label_order[LETTERS.index(letter)]


# --- instruction strings (bilingual) ---------------------------------------

_INSTR = {
    "en": (
        "You are an expert in underwater acoustics. Listen to the underwater "
        "recording and identify the source ship type. Choose the best option and "
        "reply with ONLY the single letter ({letters}).\n{options}{enrich}"
    ),
    "zh": (
        "你是水声专家。请听这段水下录音，识别声源的船舶类型。"
        "从下列选项中选择最合适的一项，只回复单个字母（{letters}）。\n{options}{enrich}"
    ),
}


def _enrichment_block(text: str | None, lang: str) -> str:
    if not text:
        return ""
    head = "Reference description" if lang == "en" else "参考描述"
    return f"\n{head}: {text}"


def _media_item(media_type: str, data) -> dict:
    assert media_type in {"audio", "image"}
    return {"type": media_type, media_type: data}


# --- prompt builders -------------------------------------------------------

def build_zero_shot(
    media: "np.ndarray | object",
    label_names: dict[int, str],
    media_type: str = "audio",
    lang: str = "en",
    enrich_text: str | None = None,
) -> list[dict]:
    """One-turn zero-shot prompt: instruction + options [+enrichment] + media."""
    letters, _ids, options = option_layout(label_names)
    instr = _INSTR[lang].format(letters="/".join(letters), options=options,
                                enrich=_enrichment_block(enrich_text, lang))
    return [{"role": "user", "content": [{"type": "text", "text": instr}, _media_item(media_type, media)]}]


def build_few_shot(
    media: "np.ndarray | object",
    support: list[dict],
    label_names: dict[int, str],
    media_type: str = "audio",
    lang: str = "en",
    enrich_text: str | None = None,
) -> list[dict]:
    """In-context few-shot prompt: k labelled support examples then the query.

    Each support example becomes a user/assistant turn. ``support`` items must
    carry ``audio``/``image`` (matching ``media_type``) and ``label_id``.
    """
    letters, _ids, options = option_layout(label_names)
    instr = _INSTR[lang].format(letters="/".join(letters), options=options,
                                enrich=_enrichment_block(enrich_text, lang))
    messages: list[dict] = []
    ask = lambda data: [{"type": "text", "text": instr}, _media_item(media_type, data)]
    for s in support:
        messages.append({"role": "user", "content": ask(s[media_type])})
        lid = s["label_id"]
        letter = LETTERS[[k for k, v in enumerate(sorted(label_names)) if v == lid][0]]
        messages.append({"role": "assistant", "content": f"{letter}. {label_names[lid]}"})
    messages.append({"role": "user", "content": ask(media)})
    return messages


# --- pairwise similarity (few-shot for small-context models, e.g. Qwen2-Audio) ---

_PAIRWISE_INSTR = {
    "en": (
        "You are an underwater acoustics expert. Recording 1 is a known example of a "
        "{label}. Recording 2 is an unknown recording. Do Recording 1 and Recording 2 "
        "come from the same type of vessel? Reply with only 'yes' or 'no'."
    ),
    "zh": (
        "你是水声专家。录音1是{label}的已知样本。录音2是一段未知录音。"
        "录音1与录音2是否来自同一种船舶？只回答 'yes' 或 'no'。"
    ),
}


def build_pairwise(media_known, media_query, known_label: str, media_type: str = "audio", lang: str = "en") -> list[dict]:
    """Two-media comparison prompt: is ``media_query`` the same class as the known
    ``media_known`` (label ``known_label``)? Scored over {'yes','no'} via
    ``choice_logprobs``. Only 2 media items per prompt → fits small contexts (the
    basis of pairwise few-shot for context-limited LALMs)."""
    instr = _PAIRWISE_INSTR[lang].format(label=known_label)
    return [{"role": "user", "content": [
        {"type": "text", "text": instr},
        {"type": "text", "text": "Recording 1:"}, _media_item(media_type, media_known),
        {"type": "text", "text": "Recording 2:"}, _media_item(media_type, media_query),
    ]}]


def build_prompt(
    media,
    label_names: dict[int, str],
    regime: str = "zero_shot",
    support: list[dict] | None = None,
    media_type: str = "audio",
    lang: str = "en",
    enrich_text: str | None = None,
) -> list[dict]:
    """Dispatch to zero-shot or few-shot builder based on ``regime``."""
    if regime == "zero_shot" or not support:
        return build_zero_shot(media, label_names, media_type, lang, enrich_text)
    return build_few_shot(media, support, label_names, media_type, lang, enrich_text)


# ---- fusion prompts (multi-modal: audio+image or image+image) -----

_MEDIA_LABELS = {
    "audio": "an audio recording",
    "mel": "a mel spectrogram",
    "stft": "an STFT spectrogram",
    "demon": "a DEMON spectrogram",
}

_FUSION_INSTR = {
    "en": (
        "You are an underwater acoustics expert. Here {is_are} {media_desc} of the same "
        "underwater recording. Use ALL available information to identify the source ship type. "
        "Choose the best option and reply with ONLY the single letter ({letters}).\n{options}"
    ),
}


def _media_desc(fusion_parts: list[str]) -> str:
    """Build a human-readable description of the media combo."""
    descs = [_MEDIA_LABELS.get(p, p) for p in fusion_parts]
    if len(descs) == 1:
        return descs[0]
    if len(descs) == 2:
        return descs[0] + " and " + descs[1]
    return ", ".join(descs[:-1]) + ", and " + descs[-1]


def build_fusion_prompt(
    media_list: list[tuple],
    label_names: dict[int, str],
    regime: str = "zero_shot",
    support: list[dict] | None = None,
    lang: str = "en",
    fusion_parts: list[str] | None = None,
) -> list[dict]:
    """Build a prompt with MULTIPLE media items (fusion: audio+image).

    ``media_list`` = [(media_data, media_type), ...] e.g. [(wav, "audio"), (img, "image")].
    ``fusion_parts`` = ["audio", "mel"] for instruction text.
    """
    letters, label_order, options = option_layout(label_names)
    parts = fusion_parts or [mt for _, mt in media_list]
    instr = _FUSION_INSTR[lang].format(
        is_are="are" if len(media_list) > 1 else "is",
        media_desc=_media_desc(parts),
        letters="/".join(letters), options=options,
    )

    def _content(ml):
        c = [{"type": "text", "text": instr}]
        for data, mtype in ml:
            c.append(_media_item(mtype, data))
        return c

    if regime == "zero_shot" or not support:
        return [{"role": "user", "content": _content(media_list)}]

    messages = []
    for s in support:
        messages.append({"role": "user", "content": _content(s["media_list"])})
        lid = s["label_id"]
        letter = LETTERS[[k for k, v in enumerate(sorted(label_names)) if v == lid][0]]
        messages.append({"role": "assistant", "content": f"{letter}. {label_names[lid]}"})
    messages.append({"role": "user", "content": _content(media_list)})
    return messages
