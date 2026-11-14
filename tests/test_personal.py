"""Tests for personalization: transparent PNG input, stand tab, joining pieces."""
import cv2
import numpy as np

from img2dxf import add_stand, load_grayscale


def save_cutout(path, color):
    """A 'remove background' result: a colored square on a transparent background."""
    img = np.zeros((200, 200, 4), np.uint8)       # fully transparent
    img[50:150, 50:150, :3] = color               # the subject's color
    img[50:150, 50:150, 3] = 255                  # the subject is visible
    cv2.imwrite(str(path), img)


def test_transparent_png_becomes_solid_silhouette(tmp_path):
    # A white shirt must not turn into a hole: color does not matter, only visibility.
    path = tmp_path / "white_subject.png"
    save_cutout(path, (255, 255, 255))
    gray = load_grayscale(path)
    assert gray[100, 100] == 0      # inside the subject = metal (black)
    assert gray[10, 10] == 255      # transparent background = empty (white)


def test_ignore_alpha_keeps_old_behavior(tmp_path):
    # With use_alpha=False the colors count again, so a white subject vanishes.
    path = tmp_path / "white_subject.png"
    save_cutout(path, (255, 255, 255))
    gray = load_grayscale(path, use_alpha=False)
    assert gray[100, 100] == 255


def test_stand_tab_is_added_below_the_design():
    mask = np.zeros((100, 200), np.uint8)
    mask[20:80, 60:140] = 255                     # the design, 80 px wide
    out = add_stand(mask, tab_px=30, min_len_px=40, overlap_px=5)
    assert out.shape[0] > mask.shape[0]           # the image grew downwards
    assert out[95, 100] == 255                    # metal right under the design
    assert np.count_nonzero(out[105]) >= 40       # the tab is at least min_len_px long


def test_frame_holds_a_floating_piece():
    from img2dxf import add_frame, join_parts
    mask = np.zeros((100, 100), np.uint8)
    mask[45:55, 45:55] = 255                      # a piece in the middle, touching nothing
    framed = add_frame(mask, frame_px=5)
    assert framed[0, 50] == 255 and framed[50, 0] == 255   # the border is metal
    joined, bridges = join_parts(framed, bridge_px=3)
    pieces = cv2.connectedComponents((joined > 0).astype(np.uint8))[0] - 1
    assert pieces == 1                            # frame and piece are now one part
    assert bridges == 1


def test_interactive_mode_builds_the_command():
    from img2dxf import ask_for_args
    answers = iter(['"C:\\photos\\dana.png"', "", "150", "y", "", "", "n"])
    args = ask_for_args(ask=lambda question: next(answers))
    assert args[0] == "C:\\photos\\dana.png"          # quotes from drag-and-drop are removed
    assert args[1].endswith("dana.dxf")               # output name defaults to the image name
    assert args[2:4] == ["--width-mm", "150"]
    assert "--frame-mm" in args and "6" in args       # frame chosen, default width 6
    assert "--stand-mm" in args                       # stand tab: default yes
    assert "--join-parts" not in args                 # answered no
