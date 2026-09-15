"""Text-enhancement variants for prompting (the ShipsEar differentiator).

Three variants, contrasted in 评测 A/B (vs. Cao 2026 who synthesised text):
  * ``none``      — plain prompt (baseline)
  * ``synthetic`` — canonical, hand-written description per ship type
  * ``real``      — ShipsEar's native ``Notes`` text (from ``shipsear_meta``)

On DeepShip (no native text) only ``none``/``synthetic`` apply; ``real`` is a
no-op there. The variants are injected via ``build_prompt(..., enrich_text=...)``.
"""

from __future__ import annotations

# Canonical, domain-grounded descriptions per ship type (English).
# Used for the "synthetic" variant and as a reproducible analogue of Cao's
# Gemini-generated descriptions.
SYNTHETIC_DESCRIPTIONS: dict[str, str] = {
    "Cargo ship": (
        "Large displacement-hull vessel with slow-speed low-RPM diesel propulsion; "
        "broad-band cavitation noise with a low fundamental blade-pass frequency."
    ),
    "Passenger ship": (
        "Medium-size vessel with higher-RPM machinery and auxiliary generators; "
        "broad-band noise dominated by mid-frequency propulsion and hotel loads."
    ),
    "Oil tanker": (
        "Very large slow single-screw vessel; strong low-frequency tonals from a "
        "large slow-speed diesel engine and a distinctive low blade-pass rate."
    ),
    "Tug boat": (
        "Small powerful vessel with high-RPM engines and often CP propellers; "
        "high-frequency broad-band noise with rapidly varying tonals under load."
    ),
}


def synthetic_text(label_name: str) -> str | None:
    return SYNTHETIC_DESCRIPTIONS.get(label_name)


def get_enrich_text(
    variant: str,
    label_name: str | None = None,
    real_meta_text: str | None = None,
) -> str | None:
    """Resolve the enrichment text for a given variant.

    * ``none``      → None
    * ``synthetic`` → canonical description for the *gold* ship type (used in the
                      text-enhanced prompting experiment; pass the candidate type
                      when per-option enrichment is desired later)
    * ``real``      → ShipsEar native notes (caller supplies ``real_meta_text``)
    """
    if variant == "none":
        return None
    if variant == "synthetic" and label_name:
        return synthetic_text(label_name)
    if variant == "real" and real_meta_text:
        return real_meta_text
    return None
