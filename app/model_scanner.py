from __future__ import annotations

import os
from pathlib import Path


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def list_model_entries(models_dir: Path) -> list[str]:
    """Return relative paths (posix) for selectable model entries under models_dir.

    Supports:
    - Diffusers folders (contains model_index.json)
    - Single-file weights (.safetensors, .ckpt)
    """
    if not models_dir.exists():
        return []

    results: list[str] = []

    # Use os.walk so we can prune traversal when we find a Diffusers directory.
    for root, dirs, files in os.walk(models_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(models_dir)

        if rel_root != Path(".") and _is_hidden(rel_root):
            dirs[:] = []
            continue

        # If this directory is a Diffusers model directory, add it and skip descending.
        if (root_path / "model_index.json").exists():
            results.append(rel_root.as_posix())
            dirs[:] = []
            continue

        # Prune hidden child dirs.
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for filename in files:
            if filename.startswith("."):
                continue
            path = root_path / filename
            if path.suffix.lower() in {".safetensors", ".ckpt"}:
                results.append(path.relative_to(models_dir).as_posix())

    return sorted(set(results))
