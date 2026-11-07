import numpy as np
from img2dxf import thicken_thin_parts


def make_line(width_px):
    """A white horizontal bar on black, width_px tall."""
    mask = np.zeros((100, 200), np.uint8)
    top = 50 - width_px // 2
    mask[top:top + width_px, 20:180] = 255
    return mask


def width_in_middle(mask):
    """How many metal pixels the bar has in its middle column."""
    return int(np.count_nonzero(mask[:, 100]))


def test_thin_line_becomes_minimum_width():
    out, added = thicken_thin_parts(make_line(7), 15)
    assert added > 0
    assert 14 <= width_in_middle(out) <= 16


def test_thick_line_is_not_changed():
    out, added = thicken_thin_parts(make_line(30), 15)
    assert added == 0