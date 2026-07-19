"""Output artifacts: crops grouped per person, montages and a JSON summary."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .embedder import FaceSample

logger = logging.getLogger(__name__)


def save_crops(
    samples: list[FaceSample],
    person_ids: list[str | None],
    out_dir: str | Path,
    max_per_cluster: int = 40,
):
    """Write face crops into output/persons/<person_id>/ folders."""
    import cv2

    out = Path(out_dir) / "persons"
    counts: dict[str, int] = {}
    for sample, person in zip(samples, person_ids):
        if sample.crop is None:
            continue
        key = person if person is not None else "_noise"
        if counts.get(key, 0) >= max_per_cluster:
            continue
        folder = out / key
        folder.mkdir(parents=True, exist_ok=True)
        idx = counts.get(key, 0)
        name = f"{idx:04d}_q{sample.quality:.2f}_f{sample.frame_idx}.jpg"
        cv2.imwrite(str(folder / name), sample.crop)
        counts[key] = idx + 1
    n_identities = sum(1 for key in counts if key != "_noise")
    logger.info("Saved crops for %d identities to %s", n_identities, out)


def save_montages(out_dir: str | Path, tile: int = 112, cols: int = 8):
    """Build one montage image per person folder for quick visual inspection."""
    import cv2

    persons_dir = Path(out_dir) / "persons"
    if not persons_dir.exists():
        return
    montages_dir = Path(out_dir) / "montages"
    montages_dir.mkdir(parents=True, exist_ok=True)
    for folder in sorted(p for p in persons_dir.iterdir() if p.is_dir()):
        crops = sorted(folder.glob("*.jpg"))[: cols * 4]
        if not crops:
            continue
        tiles = []
        for path in crops:
            img = cv2.imread(str(path))
            if img is None:
                continue
            tiles.append(cv2.resize(img, (tile, tile)))
        if not tiles:
            continue
        rows = int(np.ceil(len(tiles) / cols))
        canvas = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)
        for i, timg in enumerate(tiles):
            r, c = divmod(i, cols)
            canvas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = timg
        cv2.imwrite(str(montages_dir / f"{folder.name}.jpg"), canvas)
    logger.info("Montages written to %s", montages_dir)


def save_summary(
    out_dir: str | Path,
    samples: list[FaceSample],
    person_ids: list[str | None],
    extra: dict | None = None,
):
    """JSON summary: per-person counts, sources, quality stats."""
    persons: dict[str, dict] = {}
    for sample, person in zip(samples, person_ids):
        key = person if person is not None else "_noise"
        entry = persons.setdefault(
            key, {"faces": 0, "sources": set(), "mean_quality": []}
        )
        entry["faces"] += 1
        entry["sources"].add(sample.source)
        entry["mean_quality"].append(sample.quality)

    for entry in persons.values():
        entry["sources"] = sorted(entry["sources"])
        q = entry.pop("mean_quality")
        entry["mean_quality"] = round(float(np.mean(q)), 3) if q else None

    summary = {
        "total_faces": len(samples),
        "identities": sum(1 for k in persons if k != "_noise"),
        "noise_faces": persons.get("_noise", {}).get("faces", 0),
        "persons": persons,
    }
    if extra:
        summary.update(extra)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Summary written to %s", out / "summary.json")
    return summary
