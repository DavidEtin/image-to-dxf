"""
Create simple test images in the folder test_images/.

    python make_test_images.py

Each image checks one feature of img2dxf.py.
"""
from pathlib import Path

import cv2
import numpy as np

OUT = Path("test_images")
BLACK = 0


def blank(w: int = 400, h: int = 400) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def circle() -> np.ndarray:
    """Simplest case: one solid disk."""
    img = blank()
    cv2.circle(img, (200, 200), 150, BLACK, -1)
    return img


def letter_o() -> np.ndarray:
    """A ring. In stencil mode its middle would fall out -> needs a bridge."""
    img = blank()
    cv2.circle(img, (200, 200), 150, BLACK, -1)
    cv2.circle(img, (200, 200), 90, 255, -1)
    return img


def thin_lines() -> np.ndarray:
    """A thick bar with lines of thickness 1..8 px. The thin ones must be thickened."""
    img = blank(500, 300)
    cv2.rectangle(img, (20, 220), (480, 280), BLACK, -1)
    for i, t in enumerate(range(1, 9)):
        x = 50 + i * 55
        cv2.line(img, (x, 40), (x, 220), BLACK, t)
    return img


def eye() -> np.ndarray:
    """Dark disk, white hole, dark pupil inside -> the pupil needs a bridge."""
    img = blank()
    cv2.circle(img, (200, 200), 160, BLACK, -1)
    cv2.ellipse(img, (200, 200), (110, 60), 0, 0, 360, 255, -1)
    cv2.circle(img, (200, 200), 35, BLACK, -1)
    return img


def specks() -> np.ndarray:
    """A square with small dust dots around it -> dots must be removed."""
    img = blank()
    cv2.rectangle(img, (100, 100), (300, 300), BLACK, -1)
    rng = np.random.default_rng(1)
    for _ in range(25):
        x, y = rng.integers(20, 380, size=2)
        cv2.circle(img, (int(x), int(y)), 1, BLACK, -1)
    return img


def text() -> np.ndarray:
    """Text for stencil mode: letters like A and R have inner islands."""
    img = blank(900, 220)
    cv2.putText(img, "STEEL ART", (20, 160), cv2.FONT_HERSHEY_DUPLEX, 4, BLACK, 18)
    return img


def main() -> None:
    OUT.mkdir(exist_ok=True)
    images = {
        "circle.png": circle(),
        "letter_o.png": letter_o(),
        "thin_lines.png": thin_lines(),
        "eye.png": eye(),
        "specks.png": specks(),
        "text.png": text(),
    }
    for name, img in images.items():
        # x3 so the images are big enough for real-size output (>= 4 px/mm)
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT / name), img)
        print(f"created {OUT / name}")


if __name__ == "__main__":
    main()
