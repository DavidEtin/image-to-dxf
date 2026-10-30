# steel-art: image to cut-ready DXF

Converts a dark-on-light silhouette image into a DXF file at real size (mm),
prepared for CNC laser cutting of steel art.

It does more than trace outlines. It also makes the design cuttable:

- **Thickens thin lines** below a minimum width, so they don't warp or break
- **Adds bridges** to floating islands (like the middle of an "A" in a stencil), so they don't fall out
- **Removes tiny details** and **fills tiny holes** the laser can't cut cleanly
- **Warns about low-resolution images** that would give stepped edges
- Saves a **preview image** where everything it changed is shown in orange

## Machine

Defaults are set for steel cut on a **TRUMPF TruLaser 5030 fiber** (full sheet 3000 x 1500 mm), 3 mm steel, electrostatic powder coating.
The minimum line width is still an estimate: cut the calibration coupon from the web demo to measure it.

Web demo: https://davidetin.github.io/image-to-dxf/

## Setup (Windows)

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Usage

```
python make_test_images.py
python img2dxf.py test_images\circle.png circle.dxf --width-mm 200
python img2dxf.py test_images\text.png text.dxf --width-mm 400 --mode stencil
```

### Modes

| Mode | Meaning |
|------|---------|
| `silhouette` (default) | The dark shape is the steel part |
| `stencil` | The dark shape is cut out of a steel plate |

### Main options

| Option | Default | Meaning |
|--------|---------|---------|
| `--width-mm` | (required) | Final width of the design |
| `--thickness-mm` | 3 | Steel thickness; other limits are based on it |
| `--min-width-mm` | = thickness | Thinnest allowed metal line |
| `--bridge-width-mm` | 1.5 x thickness | Width of added bridges |
| `--margin-mm` | 15 | Plate border in stencil mode |

Run `python img2dxf.py --help` for all options.

## Roadmap

- [x] Image to DXF with cut-readiness fixes
- [ ] Automated tests with pytest
- [ ] CI with GitHub Actions
- [ ] Simple web interface, Docker, cloud deployment
