from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "logo_reversprom_horizontal_dark.png"


@dataclass(frozen=True)
class Bg:
    rgb: tuple[int, int, int]


def _sample_bg(img: Image.Image) -> Bg:
    """
    Background is assumed to be solid. We sample corners and take the median-ish value.
    """
    im = img.convert("RGB")
    w, h = im.size
    pts = [
        (2, 2),
        (w - 3, 2),
        (2, h - 3),
        (w - 3, h - 3),
        (w // 2, 2),
        (w // 2, h - 3),
    ]
    colors = [im.getpixel(p) for p in pts]
    colors.sort(key=lambda c: (c[0] + c[1] + c[2], c[0], c[1], c[2]))
    return Bg(rgb=colors[len(colors) // 2])


def _extract_alpha(img: Image.Image) -> Image.Image:
    """
    Create alpha mask by luminance threshold.
    Works well when logo is significantly brighter than background,
    even if the background isn't perfectly flat.
    """
    rgb = img.convert("RGB")
    gray = rgb.convert("L")

    # Threshold tuned for the current dark background export:
    # background is near-black, logo is bright metallic.
    alpha = gray.point(lambda p: 0 if p < 38 else min(255, int((p - 38) * 5.2)))
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.8))
    return alpha


def _to_rgba_cutout(img: Image.Image) -> Image.Image:
    alpha = _extract_alpha(img)
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def _compose_on_bg(cutout: Image.Image, bg_rgb: tuple[int, int, int], size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, bg_rgb).convert("RGBA")
    x = (size[0] - cutout.size[0]) // 2
    y = (size[1] - cutout.size[1]) // 2
    canvas.alpha_composite(cutout, (x, y))
    return canvas.convert("RGB")

def _invert_under_alpha(cutout: Image.Image) -> Image.Image:
    """
    Invert RGB only where alpha > 0 to create 'dark metal' from 'light metal'
    while keeping the same texture/highlights (inverted).
    """
    rgba = cutout.convert("RGBA")
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))
    inv = ImageChops.invert(rgb)
    out = Image.merge("RGBA", (*inv.split(), a))
    return out


def _bbox_from_alpha(rgba: Image.Image, threshold: int = 8) -> tuple[int, int, int, int]:
    a = rgba.getchannel("A")
    # Binarize for bbox stability
    mask = a.point(lambda p: 255 if p > threshold else 0)
    bbox = mask.getbbox()
    if not bbox:
        return (0, 0, rgba.size[0], rgba.size[1])
    return bbox


def _split_icon_and_text(rgba: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Split into icon (left) and text (right) by scanning alpha columns for a gap.
    """
    a = rgba.getchannel("A")
    w, h = rgba.size
    col = [sum(a.crop((x, 0, x + 1, h)).getdata()) for x in range(w)]
    # Find a wide low-alpha valley after the icon
    # Search middle region for the minimum run
    start = int(w * 0.18)
    end = int(w * 0.55)
    min_x = min(range(start, end), key=lambda x: col[x])
    # Expand around min_x to get a gap region
    gap_thresh = max(1, int(max(col) * 0.001))
    left = min_x
    right = min_x
    while left > 0 and col[left] <= gap_thresh:
        left -= 1
    while right < w - 1 and col[right] <= gap_thresh:
        right += 1
    cut_x = (left + right) // 2
    icon = rgba.crop((0, 0, cut_x, h))
    text = rgba.crop((cut_x, 0, w, h))
    # Tighten each
    icon = icon.crop(_bbox_from_alpha(icon))
    text = text.crop(_bbox_from_alpha(text))
    return icon, text


def _letter_blocks(text_rgba: Image.Image) -> list[tuple[int, int]]:
    """
    Returns list of (x0, x1) blocks for letters based on alpha projection.
    """
    a = text_rgba.getchannel("A")
    w, h = text_rgba.size
    proj = [sum(a.crop((x, 0, x + 1, h)).getdata()) for x in range(w)]
    # Dynamic threshold
    thr = max(1, int(max(proj) * 0.03))
    blocks: list[tuple[int, int]] = []
    in_block = False
    x0 = 0
    for x in range(w):
        on = proj[x] > thr
        if on and not in_block:
            in_block = True
            x0 = x
        if not on and in_block:
            x1 = x
            if x1 - x0 > 6:
                blocks.append((x0, x1))
            in_block = False
    if in_block:
        blocks.append((x0, w))
    return blocks


def _crop_text_parts(text_rgba: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
    """
    Split text 'РЕВЕРСПРОМ' into three RGBA parts: 'Р', 'ЕВЕРС', 'ПРОМ'
    based on detected letter blocks.
    """
    blocks = _letter_blocks(text_rgba)
    if len(blocks) < 10:
        # fallback: treat whole as one line (won't satisfy layout but avoids crash)
        tight = text_rgba.crop(_bbox_from_alpha(text_rgba))
        return tight, Image.new("RGBA", (1, 1), (0, 0, 0, 0)), Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    # First letter 'Р' = block 0
    r0, r1 = blocks[0]
    # 'ЕВЕРС' = blocks 1..5
    e0 = blocks[1][0]
    e1 = blocks[5][1]
    # 'ПРОМ' = blocks 6..9
    p0 = blocks[6][0]
    p1 = blocks[9][1]

    R = text_rgba.crop((r0, 0, r1, text_rgba.size[1])).crop(_bbox_from_alpha(text_rgba.crop((r0, 0, r1, text_rgba.size[1]))))
    EVERS = text_rgba.crop((e0, 0, e1, text_rgba.size[1])).crop(_bbox_from_alpha(text_rgba.crop((e0, 0, e1, text_rgba.size[1]))))
    PROM = text_rgba.crop((p0, 0, p1, text_rgba.size[1])).crop(_bbox_from_alpha(text_rgba.crop((p0, 0, p1, text_rgba.size[1]))))
    return R, EVERS, PROM


def _resize_keep(img: Image.Image, height: int) -> Image.Image:
    w, h = img.size
    if h <= 0:
        return img
    scale = height / h
    return img.resize((max(1, int(w * scale)), height), Image.Resampling.LANCZOS)


def _compose_square_stacked(
    icon: Image.Image,
    r_part: Image.Image,
    evers: Image.Image,
    prom: Image.Image,
    bg_rgb: tuple[int, int, int],
    square: int = 1024,
) -> Image.Image:
    """
    icon on left, text stacked on right:
    Top line:  Р + ЕВЕРС
    Bottom:    ПРОМ (aligned under ЕВЕРС)
    """
    canvas = Image.new("RGB", (square, square), bg_rgb).convert("RGBA")

    pad = 90
    gap = 34
    line_gap = 18

    # Choose a target line height based on available height
    target_line_h = int((square - 2 * pad - line_gap) * 0.42)

    icon_h = int((square - 2 * pad) * 0.62)
    icon_r = _resize_keep(icon, icon_h)

    r_r = _resize_keep(r_part, target_line_h)
    e_r = _resize_keep(evers, target_line_h)
    p_r = _resize_keep(prom, target_line_h)

    text_w = r_r.size[0] + gap + e_r.size[0]
    right_w = square - 2 * pad - icon_r.size[0] - gap
    # If too wide, scale down uniformly
    if text_w > right_w:
        scale = right_w / text_w
        new_h = max(1, int(target_line_h * scale))
        r_r = _resize_keep(r_part, new_h)
        e_r = _resize_keep(evers, new_h)
        p_r = _resize_keep(prom, new_h)

    # Vertical centering for 2 lines
    total_text_h = r_r.size[1] + line_gap + p_r.size[1]
    y0 = (square - total_text_h) // 2

    x_icon = pad
    y_icon = (square - icon_r.size[1]) // 2
    canvas.alpha_composite(icon_r, (x_icon, y_icon))

    x_text = x_icon + icon_r.size[0] + gap
    # Top line
    canvas.alpha_composite(r_r, (x_text, y0))
    x_e = x_text + r_r.size[0] + gap
    canvas.alpha_composite(e_r, (x_e, y0))
    # Bottom line aligned under "ЕВЕРС"
    canvas.alpha_composite(p_r, (x_e, y0 + r_r.size[1] + line_gap))

    return canvas.convert("RGB")


def _fit_into_square(cutout: Image.Image, square: int, padding: int) -> Image.Image:
    """
    Resizes cutout proportionally to fit into a square with padding.
    """
    target = square - 2 * padding
    w, h = cutout.size
    scale = min(target / w, target / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cutout.resize((nw, nh), Image.Resampling.LANCZOS)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Source not found: {SRC}")

    src_img = Image.open(SRC)
    cutout_light = _to_rgba_cutout(src_img)
    cutout_dark = _invert_under_alpha(cutout_light)

    # Keep original horizontal canvas size for horizontal variants
    w, h = src_img.size

    white = (255, 255, 255)
    black = (0, 0, 0)

    # 1) horizontal on white should be DARK metal
    out1 = _compose_on_bg(cutout_dark, white, (w, h))
    (ROOT / "logo_reversprom_horizontal_white_exact.png").write_bytes(_to_png(out1))

    # 2) square variants: stacked layout
    icon, text = _split_icon_and_text(cutout_light)
    r_part, evers, prom = _crop_text_parts(text)

    # White background => DARK metal
    icon_dark = _invert_under_alpha(icon)
    r_dark = _invert_under_alpha(r_part)
    e_dark = _invert_under_alpha(evers)
    p_dark = _invert_under_alpha(prom)
    sq_white = _compose_square_stacked(icon_dark, r_dark, e_dark, p_dark, bg_rgb=white, square=1024)
    (ROOT / "logo_reversprom_square_white_exact.png").write_bytes(_to_png(sq_white))

    # Black background => LIGHT metal
    sq_black = _compose_square_stacked(icon, r_part, evers, prom, bg_rgb=black, square=1024)
    (ROOT / "logo_reversprom_square_dark_exact.png").write_bytes(_to_png(sq_black))

    print("Saved:")
    print(f"- {ROOT / 'logo_reversprom_horizontal_white_exact.png'}")
    print(f"- {ROOT / 'logo_reversprom_square_white_exact.png'}")
    print(f"- {ROOT / 'logo_reversprom_square_dark_exact.png'}")


def _to_png(img: Image.Image) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


if __name__ == "__main__":
    main()

