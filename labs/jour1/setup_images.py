"""Download a public-domain test image for the Jour 1 lab.

Uses a freely available sample image from a reliable source.
If download fails, falls back to generating a synthetic scene.
"""
from pathlib import Path

import cv2
import numpy as np


def download_test_image() -> str:
    """Attempt to download a sample image; return path or empty string on failure."""
    url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
        "PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    )
    out = Path(__file__).parent / "assets" / "test_image.png"
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(out))
        img = cv2.imread(str(out))
        if img is not None:
            print(f"Downloaded test image: {out}")
            return str(out)
    except Exception:
        pass
    return ""


def generate_test_image() -> str:
    """Create a realistic synthetic scene with multiple objects and text."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    # Gradient background
    for i in range(300):
        img[i, :] = [int(20 + 0.1 * i), int(20 + 0.05 * i), int(40 + 0.15 * i)]
    # Large white rectangle
    cv2.rectangle(img, (30, 40), (180, 200), (255, 255, 255), -1)
    # Circle
    cv2.circle(img, (300, 150), 60, (255, 200, 0), -1)
    # Smaller shapes
    cv2.rectangle(img, (200, 20), (260, 70), (0, 165, 255), 3)
    cv2.ellipse(img, (100, 250), (50, 30), 0, 0, 360, (128, 255, 128), -1)
    # Text
    cv2.putText(img, "CV Lab", (220, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Noise for texture
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    out = Path(__file__).parent / "assets" / "test_scene.png"
    cv2.imwrite(str(out), img)
    print(f"Generated test image: {out}")
    return str(out)


def generate_low_contrast_image() -> str:
    """Create a dark, low-contrast image for equalizeHist demo."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    # Very dark shapes
    for _ in range(5):
        x1 = int(np.random.randint(10, 200))
        y1 = int(np.random.randint(10, 150))
        x2 = int(np.random.randint(x1 + 20, 280))
        y2 = int(np.random.randint(y1 + 20, 180))
        val = int(np.random.randint(30, 70))
        cv2.rectangle(img, (x1, y1), (x2, y2), (val, val, val), -1)
    noise = np.random.randint(0, 10, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    out = Path(__file__).parent / "assets" / "test_low_contrast.png"
    cv2.imwrite(str(out), img)
    print(f"Generated low-contrast image: {out}")
    return str(out)


if __name__ == "__main__":
    (Path(__file__).parent / "assets").mkdir(parents=True, exist_ok=True)
    download_test_image() or generate_test_image()
    generate_low_contrast_image()
    print("All test images ready in labs/jour1/assets/")
