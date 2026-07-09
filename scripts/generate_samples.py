"""Generate sample good and defective product images for testing."""

import os
import cv2
import numpy as np

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")


def make_good_product(size=400):
    """Uniform metallic surface — no defects."""
    img = np.full((size, size, 3), 180, dtype=np.uint8)
    # Subtle gradient
    for i in range(size):
        img[i, :, :] = 175 + int(10 * i / size)
    # Light noise
    noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def make_scratched_product(size=400):
    """Surface with prominent scratch defects."""
    img = make_good_product(size)
    scratches = [
        ((30, 30), (370, 370), (60, 60, 60), 5),
        ((80, 10), (320, 390), (50, 50, 50), 4),
        ((200, 0), (200, 400), (45, 45, 45), 4),
        ((0, 200), (400, 200), (55, 55, 55), 3),
        ((100, 100), (300, 300), (40, 40, 40), 3),
    ]
    for pt1, pt2, color, thickness in scratches:
        cv2.line(img, pt1, pt2, color, thickness)
    return img


def make_texture_defect(size=400):
    """Localized texture anomaly with crack and rough patch."""
    img = make_good_product(size)
    cv2.circle(img, (200, 200), 70, (90, 90, 90), -1)
    cv2.circle(img, (200, 200), 50, (70, 70, 70), 3)
    # Crack lines radiating from defect
    for angle in range(0, 360, 45):
        rad = np.radians(angle)
        x2 = int(200 + 120 * np.cos(rad))
        y2 = int(200 + 120 * np.sin(rad))
        cv2.line(img, (200, 200), (x2, y2), (50, 50, 50), 2)
    # Rough noisy patch
    patch = img[260:380, 60:180]
    patch = np.clip(
        patch.astype(np.int16) + np.random.randint(-60, 60, patch.shape), 0, 255
    ).astype(np.uint8)
    img[260:380, 60:180] = patch
    return img


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    samples = {
        "good_product.png": make_good_product(),
        "defect_scratch.png": make_scratched_product(),
        "defect_texture.png": make_texture_defect(),
    }

    for name, img in samples.items():
        path = os.path.join(SAMPLES_DIR, name)
        cv2.imwrite(path, img)
        print(f"Saved: {path}")

    print(f"\n{len(samples)} sample images generated in {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
