"""
ImageLoader — robust multi-format image reader.

Tries the given path first, then falls back to sibling extensions so callers
never need to know whether a stem is stored as .jpg, .png, .jpeg, or .bmp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Priority order: most common first
_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]


class ImageLoader:
    """Stateless helper for reading face images in any common format."""

    @staticmethod
    def load(path: Path | str) -> Optional[np.ndarray]:
        """
        Load a BGR frame from `path`.

        If the file at the given path cannot be read, tries the same stem
        with every extension in _EXTENSIONS before returning None.
        """
        path = Path(path)
        frame = cv2.imread(str(path))
        if frame is not None:
            return frame

        # Fallback: try other extensions in the same directory
        for ext in _EXTENSIONS:
            if ext == path.suffix.lower():
                continue
            alt = path.with_suffix(ext)
            if alt.exists():
                frame = cv2.imread(str(alt))
                if frame is not None:
                    return frame

        return None

    @staticmethod
    def find(stem: str, directory: Path | str) -> Optional[Path]:
        """
        Find an image file by stem in `directory`, trying all extensions.
        Returns the first matching Path, or None if nothing found.
        """
        directory = Path(directory)
        for ext in _EXTENSIONS:
            p = directory / (stem + ext)
            if p.exists():
                return p
        return None

    @staticmethod
    def collect(directory: Path | str, recursive: bool = False) -> list[Path]:
        """
        List all image files under `directory`.
        Returns paths sorted for reproducibility.
        """
        directory = Path(directory)
        paths: list[Path] = []
        glob = directory.rglob if recursive else directory.glob
        for ext in _EXTENSIONS:
            paths.extend(glob(f"*{ext}"))
            paths.extend(glob(f"*{ext.upper()}"))
        return sorted(set(paths))
