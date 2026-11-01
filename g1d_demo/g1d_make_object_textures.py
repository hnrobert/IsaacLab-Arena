"""Generate diffuse textures for the generated pickable objects.

The primitives (cups/spheres/books/towels/totes) render FLAT with a
plain diffuse color in RTX (user report); a UsdUVTexture-driven
PreviewSurface gives them a real material look. Output PNGs land in
g1d_demo/assets/materials/ and are bound by bind_object_materials()
in g1d_usd.py.

    python3 g1d_demo/g1d_make_object_textures.py
"""

import os
import random

from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "materials")
SIZE = 256


def _noise(base, spread, size=SIZE):
    rng = random.Random(42)
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            n = rng.randint(-spread, spread)
            px[x, y] = tuple(max(0, min(255, c + n)) for c in base)
    return img


def wood():
    """Warm wood grain: vertical streaks over a noise base."""
    rng = random.Random(7)
    img = _noise((150, 105, 60), 8)
    px = img.load()
    for x in range(SIZE):
        grain = rng.randint(0, 26)
        for y in range(SIZE):
            r, g, b = px[x, y]
            k = grain if (x // rng.choice((3, 5, 7))) % 2 else grain // 2
            px[x, y] = (max(0, r - k), max(0, g - k), max(0, b - k))
    return img


def metal():
    """Brushed steel: horizontal streaks."""
    rng = random.Random(11)
    img = Image.new("RGB", (SIZE, SIZE))
    px = img.load()
    for y in range(SIZE):
        base = 168 + rng.randint(-10, 10)
        for x in range(SIZE):
            c = max(90, min(230, base + rng.randint(-14, 14)))
            px[x, y] = (c, c, c + 6)
    return img


def plastic(color):
    """Glossy plastic: fine speckle."""
    return _noise(color, 6)


def paper():
    """Cardboard paper: fiber noise + faint bands."""
    img = _noise((196, 170, 128), 10)
    px = img.load()
    for y in range(0, SIZE, 32):
        for x in range(SIZE):
            r, g, b = px[x, y]
            px[x, y] = (r - 14, g - 14, b - 12)
    return img


def fabric():
    """Weave: crosshatch threads."""
    img = _noise((228, 226, 218), 5)
    px = img.load()
    for i in range(SIZE):
        if i % 4 == 0:
            for j in range(SIZE):
                r, g, b = px[i, j]
                px[i, j] = (r - 16, g - 16, b - 14)
                r, g, b = px[j, i]
                px[j, i] = (r - 16, g - 16, b - 14)
    return img


def fruit(base, speckle):
    """Fruit skin: speckled tone."""
    rng = random.Random(3)
    img = _noise(base, 12)
    px = img.load()
    for _ in range(900):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        r, g, b = px[x, y]
        d = rng.randint(-speckle, speckle)
        px[x, y] = (max(0, min(255, r + d)), max(0, min(255, g + d // 2)), max(0, min(255, b + d // 3)))
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {
        "wood.png": wood(),
        "metal.png": metal(),
        "paper.png": paper(),
        "fabric.png": fabric(),
        "plastic_blue.png": plastic((38, 96, 205)),
        "plastic_yellow.png": plastic((228, 190, 48)),
        "apple_red.png": fruit((178, 32, 28), 26),
        "orange.png": fruit((232, 138, 24), 22),
        "lime_green.png": fruit((72, 178, 62), 24),
    }
    for name, img in out.items():
        img.save(os.path.join(OUT, name))
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
