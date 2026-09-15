"""Class-label mappings for each UATR dataset.

Each dataset maps integer label ids (== folder names on disk) to
``(name_en, name_zh)``. The bilingual names directly support the
中/英 prompt ablation (评测 A/B).

DeepShip mapping confirmed from the dataset's own label screenshot
(``微信图片_20250211090311.png``): 0=Cargo, 1=Passenger, 2=Oil tanker, 3=Tug.
ShipsEar-12class mapping is best-effort from the same screenshot; the local
folder layout (A/B/C) does not line up with the 12 types, so ShipsEar manifest
generation is treated separately (see ``shipsear_meta.py``).
"""

from __future__ import annotations

# label_id -> (name_en, name_zh)
LABELS: dict[str, dict[int, tuple[str, str]]] = {
    "deepship": {
        0: ("Cargo ship", "货船"),
        1: ("Passenger ship", "客船"),
        2: ("Oil tanker", "油轮"),
        3: ("Tug boat", "拖船"),
    },
    # ShipsEar 12-class — canonical types (from ShipsEar_meta.xlsx), label_id = sorted order.
    # Filenames encode the type as a prefix (e.g. "Oceanliner_22.wav"); normalized at parse.
    "shipsear": {
        0: ("Dredger", "挖泥船"),
        1: ("Fishboat", "渔船"),
        2: ("Motorboat", "摩托艇"),
        3: ("Mussel boat", "贻贝船"),
        4: ("Natural ambient noise", "自然背景噪声"),
        5: ("Ocean liner", "远洋客轮"),
        6: ("Passengers", "客船"),
        7: ("Pilot ship", "引航船"),
        8: ("RORO", "滚装船"),
        9: ("Sailboat", "帆船"),
        10: ("Trawler", "拖网渔船"),
        11: ("Tugboat", "拖船"),
    },
}


def class_names(dataset: str, lang: str = "en") -> dict[int, str]:
    """Return ``{label_id: name}`` for the requested language (``en`` or ``zh``)."""
    table = LABELS[dataset]
    idx = 0 if lang.startswith("en") else 1
    return {k: v[idx] for k, v in table.items()}


def num_classes(dataset: str) -> int:
    return len(LABELS[dataset])


def label_to_id(dataset: str, name: str, lang: str = "en") -> int | None:
    """Reverse lookup: class name -> id. Returns ``None`` if not found."""
    names = class_names(dataset, lang)
    for lid, nm in names.items():
        if nm.lower() == name.lower():
            return lid
    return None
