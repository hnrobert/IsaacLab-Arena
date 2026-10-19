"""Generate the two missing kit texture families (cone + aisle sign).

The Simple_Warehouse kit shipped with M_TrafficCone.mdl / M_AisleSign.mdl
but their PNGs were never copied - the MDL texture lookups return black,
so both props rendered solid black. This regenerates plausible stand-ins
matching the MDL's expectations (measured UVs: cone v~0.1 = tip,
v 0.31-0.40 = mid-body band zone, v>0.5 = base disc; sign = full-face
board). Run inside the container:

    docker exec isaaclab_arena-latest bash -c \
      "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
       g1d_demo/g1d_make_textures.py"
"""

import numpy as np
import os

from PIL import Image, ImageDraw
from pxr import Usd

MAT = os.path.join(os.path.dirname(__file__), "assets", "warehouse", "Materials")
TEX = os.path.join(MAT, "Textures")
PROPS = os.path.join(os.path.dirname(__file__), "assets", "warehouse", "Props")
S = 512

ORANGE = (250, 108, 8)  # safety-orange cone body
WHITE = (255, 255, 255)
FLAT_N = (128, 128, 255)  # +Z neutral normal
ORM_PLASTIC = (230, 90, 0)  # AO 0.9 / rough 0.35 / metal 0
ORM_BOARD = (230, 120, 0)

# cone band: physical z 0.20..0.30 m of the 0.46 m cone lands at mesh
# v ~0.31..0.40 (tip at v~0.1, base disc above v~0.5); GL v=0 is the
# BOTTOM image row
BAND_V = (0.31, 0.40)


def rows_for_v(v0, v1, size=S):
    return int(round((1.0 - v1) * size)), int(round((1.0 - v0) * size))


def cone():
    d = Image.new("RGB", (S, S), ORANGE)
    m = Image.new("RGB", (S, S), (0, 0, 0))
    y0, y1 = rows_for_v(*BAND_V)
    for img, color in ((d, WHITE), (m, WHITE)):
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, y0, S, y1], fill=color)
    d.save(os.path.join(TEX, "T_TrafficCone_D.png"))
    m.save(os.path.join(TEX, "T_TrafficCone_Stripes.png"))
    Image.new("RGB", (S, S), FLAT_N).save(os.path.join(TEX, "T_TrafficCone_N.png"))
    Image.new("RGB", (S, S), ORM_PLASTIC).save(os.path.join(TEX, "T_TrafficCone_ORM.png"))


def sign():
    # MDL: white-ish D board; Text = white glyph on black. Result renders
    # as a black board with a bright glyph. Upright orientation decided
    # by the mesh's v<->z correlation (v tracks +z -> glyph tip at v=1,
    # i.e. image top row).
    stage = Usd.Stage.Open(os.path.join(PROPS, "S_AisleSign.usd"))
    prim = next(p for p in stage.Traverse() if p.GetTypeName() == "Mesh")
    pts = np.array(prim.GetAttribute("points").Get())
    st = np.array(prim.GetAttribute("primvars:st").Get())
    corr = np.corrcoef(st[:, 1], pts[:, 2])[0, 1]  # v vs z
    flip = corr < 0

    text = Image.new("RGB", (S, S), (0, 0, 0))
    dr = ImageDraw.Draw(text)
    if not flip:  # v=1 (image top) = +z: arrow points up
        dr.polygon(
            [
                (S // 2, 30),
                (S - 130, 240),
                (S // 2 + 60, 240),
                (S // 2 + 60, 420),
                (S // 2 - 60, 420),
                (S // 2 - 60, 240),
                (130, 240),
            ],
            fill=WHITE,
        )
        for yy in (360, 420):
            dr.rectangle([S // 2 - 90, yy - 26, S // 2 + 90, yy + 2], fill=WHITE)
    else:  # v=1 = -z: mirror vertically
        dr.polygon(
            [
                (S // 2, S - 30),
                (S - 130, S - 240),
                (S // 2 + 60, S - 240),
                (S // 2 + 60, S - 420),
                (S // 2 - 60, S - 420),
                (S // 2 - 60, S - 240),
                (130, S - 240),
            ],
            fill=WHITE,
        )
        for yy in (S - 360, S - 420):
            dr.rectangle([S // 2 - 90, S - yy - 2, S // 2 + 90, S - yy + 26], fill=WHITE)
    text.save(os.path.join(TEX, "AisleSign_Text_01.png"))
    Image.new("RGB", (S, S), WHITE).save(os.path.join(TEX, "T_AisleSign_D.png"))
    Image.new("RGB", (S, S), FLAT_N).save(os.path.join(TEX, "T_AisleSign_N.png"))
    Image.new("RGB", (S, S), ORM_BOARD).save(os.path.join(TEX, "T_AisleSign_ORM.png"))
    print(f"sign v/z corr {corr:+.2f} -> glyph flip={flip}", flush=True)


if __name__ == "__main__":
    cone()
    sign()
    for f in (
        "T_TrafficCone_D.png",
        "T_TrafficCone_Stripes.png",
        "T_TrafficCone_N.png",
        "T_TrafficCone_ORM.png",
        "AisleSign_Text_01.png",
        "T_AisleSign_D.png",
        "T_AisleSign_N.png",
        "T_AisleSign_ORM.png",
    ):
        p = os.path.join(TEX, f)
        assert os.path.exists(p) and os.path.getsize(p) > 0, p
    print("cone + sign textures written to", TEX, flush=True)
