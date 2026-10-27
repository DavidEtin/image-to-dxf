"""
img2dxf - turn a dark-on-light silhouette image into a cut-ready DXF file (mm).

Pipeline (each step is a small function you can test on its own):

    load image -> black/white mask -> crop -> scale to real size
    -> (stencil mode: build plate) -> remove tiny parts -> thicken thin parts
    -> fill tiny holes -> add bridges to floating islands -> outlines -> DXF

In the mask, 255 means "metal that stays" and 0 means "cut away / empty".

Example:
    python img2dxf.py test_images/circle.png circle.dxf --width-mm 200
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import ezdxf
import numpy as np
from ezdxf import units

METAL = 255
EMPTY = 0
CUT_LAYER = "CUT"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """All the knobs. Values left as None get a default based on thickness."""

    width_mm: float                      # final width of the design
    thickness_mm: float = 2.0            # steel thickness
    min_width_mm: float | None = None    # thinnest allowed metal line
    min_hole_mm: float | None = None     # smaller holes get filled
    min_detail_mm: float | None = None   # smaller metal specks get removed
    bridge_width_mm: float | None = None # width of added bridges
    mode: str = "silhouette"             # "silhouette" or "stencil"
    margin_mm: float = 15.0              # plate border in stencil mode
    simplify_mm: float = 0.1             # how far outlines may move when simplified
    px_per_mm: float = 5.0               # working resolution
    threshold: int | None = None         # 0-255, None = automatic (Otsu)

    def __post_init__(self) -> None:
        if self.width_mm <= 0:
            raise ValueError("width_mm must be positive")
        if self.thickness_mm <= 0:
            raise ValueError("thickness_mm must be positive")
        if self.mode not in ("silhouette", "stencil"):
            raise ValueError("mode must be 'silhouette' or 'stencil'")
        if self.px_per_mm <= 0:
            raise ValueError("px_per_mm must be positive")
        t = self.thickness_mm
        # Laser rule of thumb: features should be at least as wide as the steel is thick.
        if self.min_width_mm is None:
            self.min_width_mm = t
        if self.min_hole_mm is None:
            self.min_hole_mm = t
        if self.min_detail_mm is None:
            self.min_detail_mm = 1.5 * t
        if self.bridge_width_mm is None:
            self.bridge_width_mm = 1.5 * t
        self.bridge_width_mm = max(self.bridge_width_mm, self.min_width_mm)


@dataclass
class Report:
    """What the conversion did, so you can check it before cutting."""

    mode: str = ""
    width_mm: float = 0.0
    height_mm: float = 0.0
    outlines: int = 0
    small_parts_removed: int = 0
    thin_area_added_mm2: float = 0.0
    small_holes_filled: int = 0
    bridges_added: int = 0
    source_px_per_mm: float = 0.0
    warnings: list[str] = field(default_factory=list)


# Below this many image pixels per mm, cut edges come out visibly stepped.
MIN_SOURCE_PX_PER_MM = 4.0


def source_resolution(content_width_px: int, width_mm: float) -> float:
    """How many pixels of the original image end up in each mm of steel."""
    return content_width_px / width_mm


def resolution_warning(content_width_px: int, width_mm: float,
                       minimum: float = MIN_SOURCE_PX_PER_MM) -> str | None:
    """Return a warning if the image is too small for the requested size, else None."""
    ppm = source_resolution(content_width_px, width_mm)
    if ppm >= minimum:
        return None
    max_width = int(content_width_px / minimum)
    return (f"Image is low resolution: {ppm:.1f} px/mm (recommended >= {minimum:g}). "
            f"Edges may look stepped. Use a bigger image, or a width up to ~{max_width} mm.")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def mm_to_px(mm: float, px_per_mm: float) -> int:
    """Convert millimetres to a whole number of pixels (at least 1)."""
    return max(1, int(round(mm * px_per_mm)))


def circle_area_px(diameter_px: int) -> float:
    """Area of a circle with the given diameter, in pixels."""
    return math.pi * (diameter_px / 2) ** 2


def disk(diameter_px: int) -> np.ndarray:
    """Round structuring element ("brush") for morphology."""
    d = max(1, int(diameter_px))
    if d % 2 == 0:
        d += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


# ---------------------------------------------------------------------------
# Loading and saving images (works with Hebrew/Unicode paths on Windows)
# ---------------------------------------------------------------------------

def load_grayscale(path: str | Path) -> np.ndarray:
    """Read an image as grayscale. Transparent areas become white."""
    if not Path(path).is_file():
        raise ValueError(f"File not found: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def save_image(path: str | Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    buf.tofile(str(path))


# ---------------------------------------------------------------------------
# Image -> mask at real size
# ---------------------------------------------------------------------------

def find_threshold(gray: np.ndarray, threshold: int | None = None) -> int:
    """Pick the gray level that separates dark (metal) from light (background)."""
    if threshold is not None:
        return int(threshold)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = gray[gray <= otsu]
    light = gray[gray > otsu]
    if dark.size == 0 or light.size == 0:
        return int(otsu)
    # Halfway between the average dark and average light value. This stays
    # sensible even for pure black/white images, where Otsu alone returns 0.
    return int((float(dark.mean()) + float(light.mean())) / 2)


def to_mask(gray: np.ndarray, threshold: int) -> np.ndarray:
    """Dark pixels (<= threshold) become METAL, the rest EMPTY."""
    return np.where(gray <= threshold, METAL, EMPTY).astype(np.uint8)


def crop_to_content(gray: np.ndarray, mask: np.ndarray, dust_fraction: float = 0.005
                    ) -> np.ndarray:
    """
    Cut away the empty margin around the shape.
    Dust (pieces smaller than dust_fraction of all the dark area) is ignored,
    so a few dirty pixels don't change the final size.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise ValueError("No dark shape found in the image")
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= dust_fraction * areas.sum()]
    x0 = min(stats[i, cv2.CC_STAT_LEFT] for i in keep)
    y0 = min(stats[i, cv2.CC_STAT_TOP] for i in keep)
    x1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in keep)
    y1 = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in keep)
    return gray[y0:y1, x0:x1]


def scale_to_width(gray: np.ndarray, width_px: int) -> np.ndarray:
    """Resize so the image is exactly width_px wide, keeping proportions."""
    h, w = gray.shape[:2]
    height_px = max(1, int(round(h * width_px / w)))
    # INTER_LINEAR when enlarging: smooth edges without the "ringing" noise of cubic.
    interp = cv2.INTER_AREA if width_px < w else cv2.INTER_LINEAR
    return cv2.resize(gray, (width_px, height_px), interpolation=interp)


def pad(mask: np.ndarray, px: int) -> np.ndarray:
    """Add an empty border so nothing touches the image edge."""
    return cv2.copyMakeBorder(mask, px, px, px, px, cv2.BORDER_CONSTANT, value=EMPTY)


def make_stencil(design: np.ndarray, margin_px: int) -> np.ndarray:
    """Stencil mode: a solid plate with the design cut OUT of it."""
    plate = np.full((design.shape[0] + 2 * margin_px, design.shape[1] + 2 * margin_px),
                    METAL, dtype=np.uint8)
    inner = plate[margin_px:margin_px + design.shape[0], margin_px:margin_px + design.shape[1]]
    inner[design == METAL] = EMPTY
    return plate


# ---------------------------------------------------------------------------
# Making it cuttable
# ---------------------------------------------------------------------------

def remove_small_parts(mask: np.ndarray, min_area_px: float) -> tuple[np.ndarray, int]:
    """Delete metal pieces smaller than min_area_px. Returns (mask, how many removed)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = mask.copy()
    removed = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            out[labels == i] = EMPTY
            removed += 1
    return out, removed


def thicken_thin_parts(mask: np.ndarray, min_width_px: int) -> tuple[np.ndarray, int]:
    """
    Find metal that is thinner than min_width_px and grow it.

    How: an "opening" with a round brush of that size erases everything the
    brush can't fit into. Whatever disappeared is "too thin". We grow (dilate)
    those thin parts by the brush size and add them back.
    Returns (mask, number of pixels added).
    """
    if min_width_px < 2:
        return mask.copy(), 0
    brush = disk(min_width_px)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, brush)
    # Grow "opened" by 1 px so the jagged pixel edge of big shapes
    # doesn't count as thin.
    opened = cv2.dilate(opened, np.ones((3, 3), np.uint8))
    thin = cv2.bitwise_and(mask, cv2.bitwise_not(opened))
    thin = _keep_long_parts(thin, min_length_px=min_width_px)
    grown = cv2.dilate(thin, brush)
    out = cv2.bitwise_or(mask, grown)
    added = int(np.count_nonzero(out) - np.count_nonzero(mask))
    return out, added


def _keep_long_parts(mask: np.ndarray, min_length_px: int) -> np.ndarray:
    """
    Keep only pieces at least min_length_px long. Real thin lines are long;
    the rounded-off tips of sharp corners are short, so they get ignored.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        length = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if length >= min_length_px:
            out[labels == i] = METAL
    return out


def fill_small_holes(mask: np.ndarray, min_area_px: float) -> tuple[np.ndarray, int]:
    """Fill closed holes smaller than min_area_px. Returns (mask, how many filled)."""
    empty = cv2.bitwise_not(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(empty, connectivity=4)
    outside = _labels_on_border(labels)
    out = mask.copy()
    filled = 0
    for i in range(1, n):
        if i in outside:
            continue
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            out[labels == i] = METAL
            filled += 1
    return out, filled


def _labels_on_border(labels: np.ndarray) -> set[int]:
    edge = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    return set(int(v) for v in np.unique(edge))


def find_floating_islands(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """
    Find metal pieces that sit inside a hole of another piece
    (like the middle of an "O" in a stencil). They would fall out when cut.
    Returns (label image, list of island label ids).
    """
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    empty = cv2.bitwise_not(mask)
    _, empty_labels = cv2.connectedComponents(empty, connectivity=4)
    outside_ids = _labels_on_border(empty_labels) - {0}
    outside = np.isin(empty_labels, list(outside_ids)) & (empty > 0)
    near_outside = cv2.dilate(outside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    anchored = set(int(v) for v in np.unique(labels[near_outside]))
    islands = [i for i in range(1, n) if i not in anchored]
    return labels, islands


def add_bridge(mask: np.ndarray, island: np.ndarray, width_px: int
               ) -> tuple[np.ndarray, tuple[int, int] | None, tuple[int, int] | None]:
    """
    Connect one island (boolean array) to the nearest other metal with a
    straight bridge. Returns (mask, start point, end point).
    """
    other = (mask > 0) & ~island
    if not other.any():
        return mask.copy(), None, None
    # Distance from every pixel to the nearest "other" metal pixel.
    src = np.where(other, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(src, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(island)
    k = int(np.argmin(dist[ys, xs]))
    p = (int(xs[k]), int(ys[k]))
    # Nearest "other" pixel to p (only edge pixels need checking).
    edge = other & ~(cv2.erode(other.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    oy, ox = np.nonzero(edge)
    j = int(np.argmin((ox - p[0]) ** 2 + (oy - p[1]) ** 2))
    q = (int(ox[j]), int(oy[j]))
    out = mask.copy()
    cv2.line(out, p, q, METAL, thickness=max(1, width_px), lineType=cv2.LINE_8)
    return out, p, q


def add_bridges(mask: np.ndarray, width_px: int, max_bridges: int = 500) -> tuple[np.ndarray, int]:
    """Keep adding bridges until no floating islands are left."""
    out = mask.copy()
    count = 0
    while count < max_bridges:
        labels, islands = find_floating_islands(out)
        if not islands:
            break
        out, p, _ = add_bridge(out, labels == islands[0], width_px)
        if p is None:
            break
        count += 1
    return out, count


# ---------------------------------------------------------------------------
# Mask -> outlines -> DXF
# ---------------------------------------------------------------------------

def extract_contours(mask: np.ndarray, simplify_px: float) -> list[np.ndarray]:
    """Trace every edge between metal and empty, and simplify it a little."""
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    result = []
    for c in contours:
        simple = cv2.approxPolyDP(c, simplify_px, True).reshape(-1, 2)
        if len(simple) >= 3:
            result.append(simple.astype(np.float64))
    return result


def contours_to_mm(contours: list[np.ndarray], height_px: int, px_per_mm: float
                   ) -> list[list[tuple[float, float]]]:
    """
    Pixels -> millimetres. Image Y grows downward, CAD Y grows upward, so we
    flip it. The drawing is moved so its bottom-left corner is at (0, 0).
    """
    if not contours:
        return []
    flipped = []
    for c in contours:
        x = c[:, 0] / px_per_mm
        y = (height_px - 1 - c[:, 1]) / px_per_mm
        flipped.append(np.column_stack([x, y]))
    all_pts = np.vstack(flipped)
    min_x, min_y = all_pts.min(axis=0)
    return [[(round(float(px - min_x), 3), round(float(py - min_y), 3)) for px, py in c]
            for c in flipped]


def bounding_size(polylines: list[list[tuple[float, float]]]) -> tuple[float, float]:
    """Width and height (mm) of everything together."""
    if not polylines:
        return 0.0, 0.0
    pts = np.array([p for line in polylines for p in line])
    w, h = pts.max(axis=0) - pts.min(axis=0)
    return round(float(w), 2), round(float(h), 2)


def write_dxf(polylines: list[list[tuple[float, float]]], path: str | Path) -> None:
    """Save closed outlines to a DXF file with millimetre units."""
    doc = ezdxf.new("R2010")
    doc.units = units.MM
    doc.header["$MEASUREMENT"] = 1  # metric
    doc.layers.add(CUT_LAYER, color=1)
    msp = doc.modelspace()
    for pts in polylines:
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": CUT_LAYER})
    doc.saveas(str(path))


def make_preview(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """
    Picture of the result: gray = metal, orange = metal we added
    (thickening, bridges), pink = metal we removed (small specks).
    """
    img = np.full((*after.shape, 3), 255, dtype=np.uint8)
    img[after > 0] = (90, 90, 90)
    img[(after > 0) & (before == 0)] = (0, 140, 255)
    img[(after == 0) & (before > 0)] = (190, 170, 255)
    return img


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------

def convert(gray: np.ndarray, s: Settings
            ) -> tuple[list[list[tuple[float, float]]], np.ndarray, Report]:
    """Run every step. Returns (outlines in mm, preview image, report)."""
    ppm = s.px_per_mm
    report = Report(mode=s.mode)

    thr = find_threshold(gray, s.threshold)
    cropped = crop_to_content(gray, to_mask(gray, thr))
    report.source_px_per_mm = round(source_resolution(cropped.shape[1], s.width_mm), 1)
    low_res = resolution_warning(cropped.shape[1], s.width_mm)
    if low_res:
        report.warnings.append(low_res)
    design = to_mask(scale_to_width(cropped, mm_to_px(s.width_mm, ppm)), thr)

    min_width_px = mm_to_px(s.min_width_mm, ppm)
    bridge_px = mm_to_px(s.bridge_width_mm, ppm)
    border = max(min_width_px, bridge_px) + 5

    if s.mode == "stencil":
        metal = make_stencil(design, mm_to_px(s.margin_mm, ppm))
    else:
        metal = design
    metal = pad(metal, border)
    before = metal.copy()

    metal, report.small_parts_removed = remove_small_parts(
        metal, circle_area_px(mm_to_px(s.min_detail_mm, ppm)))
    metal, added_px = thicken_thin_parts(metal, min_width_px)
    report.thin_area_added_mm2 = round(added_px / ppm ** 2, 1)
    metal, report.small_holes_filled = fill_small_holes(
        metal, circle_area_px(mm_to_px(s.min_hole_mm, ppm)))
    metal, report.bridges_added = add_bridges(metal, bridge_px)

    contours = extract_contours(metal, s.simplify_mm * ppm)
    polylines = contours_to_mm(contours, metal.shape[0], ppm)
    report.outlines = len(polylines)
    report.width_mm, report.height_mm = bounding_size(polylines)

    n_parts = cv2.connectedComponents(metal, connectivity=8)[0] - 1
    if n_parts > 1:
        report.warnings.append(f"Design is {n_parts} separate pieces (not one part).")
    if report.thin_area_added_mm2 > 0:
        report.warnings.append("Thin areas were thickened - check orange areas in the preview.")
    if report.bridges_added:
        report.warnings.append("Bridges were added - check orange lines in the preview.")
    return polylines, make_preview(before, metal), report


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a silhouette image to a cut-ready DXF (mm).")
    p.add_argument("input", help="input image (png/jpg)")
    p.add_argument("output", help="output DXF file")
    p.add_argument("--width-mm", type=float, required=True, help="final design width in mm")
    p.add_argument("--thickness-mm", type=float, default=2.0, help="steel thickness (default 2)")
    p.add_argument("--min-width-mm", type=float, help="thinnest metal line (default = thickness)")
    p.add_argument("--min-hole-mm", type=float, help="fill holes smaller than this (default = thickness)")
    p.add_argument("--min-detail-mm", type=float, help="remove specks smaller than this (default = 1.5x thickness)")
    p.add_argument("--bridge-width-mm", type=float, help="bridge width (default = 1.5x thickness)")
    p.add_argument("--mode", choices=["silhouette", "stencil"], default="silhouette",
                   help="silhouette: the dark shape is the part. stencil: the dark shape is cut out of a plate")
    p.add_argument("--margin-mm", type=float, default=15.0, help="plate border in stencil mode (default 15)")
    p.add_argument("--simplify-mm", type=float, default=0.1, help="outline simplification (default 0.1)")
    p.add_argument("--px-per-mm", type=float, default=5.0, help="working resolution (default 5)")
    p.add_argument("--threshold", type=int, help="dark/light cut-off 0-255 (default: automatic)")
    p.add_argument("--no-preview", action="store_true", help="do not save the preview PNG")
    return p.parse_args(argv)


def settings_from_args(a: argparse.Namespace) -> Settings:
    return Settings(
        width_mm=a.width_mm, thickness_mm=a.thickness_mm, min_width_mm=a.min_width_mm,
        min_hole_mm=a.min_hole_mm, min_detail_mm=a.min_detail_mm,
        bridge_width_mm=a.bridge_width_mm, mode=a.mode, margin_mm=a.margin_mm,
        simplify_mm=a.simplify_mm, px_per_mm=a.px_per_mm, threshold=a.threshold,
    )


def print_report(r: Report, dxf_path: str, preview_path: str | None) -> None:
    print(f"Saved DXF:      {dxf_path}")
    if preview_path:
        print(f"Saved preview:  {preview_path}")
    print(f"Mode:           {r.mode}")
    print(f"Size:           {r.width_mm} x {r.height_mm} mm")
    print(f"Image quality:  {r.source_px_per_mm} px/mm (recommended >= {MIN_SOURCE_PX_PER_MM:g})")
    print(f"Outlines:       {r.outlines}")
    print(f"Specks removed: {r.small_parts_removed}")
    print(f"Thin area fixed:{r.thin_area_added_mm2} mm2")
    print(f"Holes filled:   {r.small_holes_filled}")
    print(f"Bridges added:  {r.bridges_added}")
    for w in r.warnings:
        print(f"WARNING: {w}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = settings_from_args(args)
        gray = load_grayscale(args.input)
        polylines, preview, report = convert(gray, settings)
        if not polylines:
            raise ValueError("No outlines found - is the image dark-on-light?")
        write_dxf(polylines, args.output)
        preview_path = None
        if not args.no_preview:
            preview_path = str(Path(args.output).with_name(Path(args.output).stem + "_preview.png"))
            save_image(preview_path, preview)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print_report(report, args.output, preview_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
