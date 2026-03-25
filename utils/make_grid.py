import os
from PIL import Image

BEFORE_DIR = "display/single_object/before"
AFTER_DIR = "display/single_object/after"
OUT_BEFORE = "display/single_object/before_grid.png"
OUT_AFTER = "display/single_object/after_grid.png"

ROWS = 5
COLS = 5


def make_grid(input_dir: str, output_path: str, rows: int = 5, cols: int = 5):
    # Load first image to get size
    first_path = os.path.join(input_dir, "1_1.png")
    first_img = Image.open(first_path).convert("RGB")
    w, h = first_img.size

    grid = Image.new("RGB", (cols * w, rows * h))

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            img_name = f"{r}_{c}.png"
            img_path = os.path.join(input_dir, img_name)

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing image: {img_path}")

            img = Image.open(img_path).convert("RGB")
            if img.size != (w, h):
                img = img.resize((w, h))

            x = (c - 1) * w
            y = (r - 1) * h
            grid.paste(img, (x, y))

    grid.save(output_path)
    print(f"Saved grid to: {output_path}")


if __name__ == "__main__":
    make_grid(BEFORE_DIR, OUT_BEFORE, ROWS, COLS)
    make_grid(AFTER_DIR, OUT_AFTER, ROWS, COLS)