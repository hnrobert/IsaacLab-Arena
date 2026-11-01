"""Inspect a downloaded SimReady USD: mesh/material/UV structure."""

import sys

from pxr import Usd, UsdGeom, UsdShade

path = sys.argv[1]
stage = Usd.Stage.Open(path)
n_mesh = n_mat = n_shader = 0
refs = set()
ids = set()
for p in stage.Traverse():
    if p.GetTypeName() == "Mesh":
        n_mesh += 1
    if p.GetTypeName() == "Material":
        n_mat += 1
    if "Shader" in p.GetTypeName():
        n_shader += 1
        i = p.GetAttribute("info:mdl:sourceAsset").Get()
        if i:
            ids.add(str(i))
        i2 = p.GetAttribute("info:sourceAsset").Get()
        if i2:
            ids.add(str(i2))
    for a in p.GetAttributes():
        if a.GetTypeName() == "asset":
            v = a.Get()
            if v and ("mdl" in str(v) or "png" in str(v) or "ref" in str(v)):
                refs.add(f"{a.GetName()}={v}")
print("meshes:", n_mesh, "materials:", n_mat, "shaders:", n_shader)
print("shader sources:", sorted(ids)[:4])
print("asset attrs:", sorted(refs)[:6])
for p in stage.Traverse():
    if p.GetTypeName() == "Mesh":
        pv = UsdGeom.PrimvarsAPI(p).GetPrimvar("st")
        print("first mesh:", p.GetPath(), "st:", pv.IsDefined())
        break
