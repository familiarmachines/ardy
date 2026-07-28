#!/usr/bin/env python3
"""Create a USD wrapper that exposes custom texture inputs as UsdPreviewSurface materials."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdShade


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a USDA layer that sublayers a source USD and adds standard "
            "UsdPreviewSurface texture networks for materials with diffuse_texture inputs."
        )
    )
    parser.add_argument("source", type=Path, help="Source USD/USDZ/USDA file.")
    parser.add_argument("output", type=Path, help="Output USDA wrapper file.")
    return parser.parse_args()


def first_texture_shader(material_prim: Usd.Prim) -> UsdShade.Shader | None:
    for child in material_prim.GetChildren():
        if child.GetTypeName() != "Shader":
            continue
        shader = UsdShade.Shader(child)
        if shader.GetInput("diffuse_texture") or shader.GetInput("diffuse_color_constant"):
            return shader
    return None


def resolve_asset(base_dir: Path, asset: Sdf.AssetPath | None) -> Path | None:
    if asset is None or not asset.path:
        return None
    path = Path(asset.path)
    if not path.is_absolute():
        path = base_dir / asset.path.replace("./", "")
    path = path.resolve()
    return path if path.exists() else None


def color3f(value: object) -> Gf.Vec3f:
    try:
        return Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))  # type: ignore[index]
    except Exception:
        return Gf.Vec3f(0.8, 0.8, 0.8)


def create_st_reader(stage: Usd.Stage, material_path: Sdf.Path) -> UsdShade.Shader:
    shader = UsdShade.Shader.Get(stage, material_path.AppendChild("PreviewPrimvar_st"))
    if shader:
        return shader
    shader = UsdShade.Shader.Define(stage, material_path.AppendChild("PreviewPrimvar_st"))
    shader.CreateIdAttr("UsdPrimvarReader_float2")
    shader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    shader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    return shader


def create_wrapper(source: Path, output: Path) -> tuple[int, int]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source USD not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)

    source_stage = Usd.Stage.Open(str(source))
    layer = Sdf.Layer.CreateAnonymous(output.name)
    layer.subLayerPaths.append(str(source))
    stage = Usd.Stage.Open(layer.identifier)

    diffuse_count = 0
    opacity_count = 0
    base_dir = source.parent
    for prim in source_stage.Traverse():
        if prim.GetTypeName() != "Material":
            continue
        source_shader = first_texture_shader(prim)
        if source_shader is None:
            continue

        material_path = prim.GetPath()
        diffuse_asset = source_shader.GetInput("diffuse_texture")
        diffuse_path = resolve_asset(base_dir, diffuse_asset.Get() if diffuse_asset else None)
        opacity_asset = source_shader.GetInput("opacity_texture")
        opacity_path = resolve_asset(base_dir, opacity_asset.Get() if opacity_asset else None)
        enable_opacity = source_shader.GetInput("enable_opacity_texture")
        color_input = source_shader.GetInput("diffuse_color_constant")

        material = UsdShade.Material.Define(stage, material_path)
        preview = UsdShade.Shader.Define(stage, material_path.AppendChild("PreviewSurface"))
        preview.CreateIdAttr("UsdPreviewSurface")
        preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        preview.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)

        diffuse_color = preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        diffuse_color.Set(color3f(color_input.Get() if color_input else None))
        material.CreateSurfaceOutput().ConnectToSource(
            preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        )

        st_reader = None
        if diffuse_path is not None:
            st_reader = create_st_reader(stage, material_path)
            texture = UsdShade.Shader.Define(stage, material_path.AppendChild("PreviewDiffuseTexture"))
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(diffuse_path)))
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            )
            diffuse_color.ConnectToSource(texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
            texture.CreateOutput("a", Sdf.ValueTypeNames.Float)
            diffuse_count += 1

        if opacity_path is not None and (enable_opacity is None or bool(enable_opacity.Get())):
            st_reader = st_reader or create_st_reader(stage, material_path)
            texture = UsdShade.Shader.Define(stage, material_path.AppendChild("PreviewOpacityTexture"))
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(opacity_path)))
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            )
            preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
                texture.CreateOutput("a", Sdf.ValueTypeNames.Float)
            )
            opacity_count += 1

    layer.Export(str(output))
    return diffuse_count, opacity_count


def main() -> None:
    args = parse_args()
    diffuse_count, opacity_count = create_wrapper(args.source, args.output)
    print(f"Wrote {args.output.expanduser().resolve()}")
    print(f"Converted diffuse textures: {diffuse_count}")
    print(f"Converted opacity textures: {opacity_count}")


if __name__ == "__main__":
    main()
