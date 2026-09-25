"""Re-layout the archived pHash panel without modifying MRI pixel content."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "supplementary" / "figures" / "phash_verification_source.png"
OUTPUT = ROOT / "supplementary" / "figures" / "phash_verification.png"


def font(size):
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def centered_multiline(draw, box, text, text_font):
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=3, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.multiline_text(
        ((left + right - width) / 2, top + (bottom - top - height) / 2),
        text,
        fill="black",
        font=text_font,
        spacing=3,
        align="center",
    )


def main():
    source = Image.open(SOURCE).convert("RGB")
    x_boxes = [(10, 350), (396, 736)]
    y_boxes = [(52, 390), (449, 787), (845, 1183), (1241, 1579)]
    filenames = [("0375.jpg", "186.jpg"), ("0799.jpg", "154.jpg"),
                 ("0074.jpg", "130.jpg"), ("1501.jpg", "192.jpg")]
    panel = 500
    title_h = 74
    gap_x = 48
    gap_y = 24
    margin = 28
    canvas = Image.new("RGB", (2 * panel + gap_x + 2 * margin,
                               4 * (panel + title_h) + 3 * gap_y + 2 * margin), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(26)
    for row, ((y0, y1), names) in enumerate(zip(y_boxes, filenames)):
        for col, ((x0, x1), name) in enumerate(zip(x_boxes, names)):
            x = margin + col * (panel + gap_x)
            y = margin + row * (panel + title_h + gap_y)
            label = f"Benchmark\n{name}" if col == 0 else f"External: pituitary macroadenoma\n{name}"
            centered_multiline(draw, (x, y, x + panel, y + title_h), label, title_font)
            crop = source.crop((x0, y0, x1, y1)).resize((panel, panel), Image.Resampling.LANCZOS)
            canvas.paste(crop, (x, y + title_h))
    canvas.save(OUTPUT, dpi=(300, 300), optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
