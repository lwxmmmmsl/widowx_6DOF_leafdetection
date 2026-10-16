"""把用户提供的番茄 GLB 转为 USD，保留叶片 UV 与透明裁切。

本函数应在 AppLauncher 之后调用；尺寸和 Y-up→Z-up 旋转由 scene.yaml 管理。

Convert the user's textured GLB to USD without losing leaf alpha cutouts.

Call after AppLauncher. Geometry remains in source units/Y-up; scene.yaml
provides an explicit scale and rotation. No claim of deformable plant physics.
"""
from pathlib import Path
import numpy as np

def convert_glb(source, directory):
    import trimesh
    from pxr import Usd,UsdGeom,UsdShade,Sdf,Gf
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    scene=trimesh.load(source,force='scene')
    # to_geometry applies node transforms; this supplied GLB has one material.
    if len(scene.geometry)!=1:
        raise ValueError('GLB converter requires one geometry/material; convert multi-material scenes to USD externally')
    mesh=scene.to_geometry(); material=mesh.visual.material
    if mesh.visual.kind!='texture' or material.baseColorTexture is None:
        raise ValueError('Expected the supplied tomato GLB with UV/baseColorTexture')
    texture=directory/'base_color.png'; material.baseColorTexture.save(texture)
    path=directory/'tomato.usda'; stage=Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage,1.)
    root=UsdGeom.Xform.Define(stage,'/Tomato'); stage.SetDefaultPrim(root.GetPrim())
    geom=UsdGeom.Mesh.Define(stage,'/Tomato/Leaves')
    geom.CreatePointsAttr(mesh.vertices.astype(np.float32).tolist())
    geom.CreateFaceVertexCountsAttr([3]*len(mesh.faces))
    geom.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())
    geom.CreateSubdivisionSchemeAttr('none'); geom.CreateDoubleSidedAttr(True)
    geom.CreateNormalsAttr(mesh.vertex_normals.astype(np.float32).tolist()); geom.SetNormalsInterpolation('vertex')
    uv=UsdGeom.PrimvarsAPI(geom).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex')
    uv.Set(np.asarray(mesh.visual.uv,dtype=np.float32).tolist())
    mat=UsdShade.Material.Define(stage,'/Tomato/Material')
    shader=UsdShade.Shader.Define(stage,'/Tomato/Material/Surface'); shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(float(material.metallicFactor or 0))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(float(material.roughnessFactor or .7))
    reader=UsdShade.Shader.Define(stage,'/Tomato/Material/UV'); reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    tex=UsdShade.Shader.Define(stage,'/Tomato/Material/Color'); tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('./base_color.png'))
    tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
    tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3); tex.CreateOutput('a',Sdf.ValueTypeNames.Float)
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
    # 叶片轮廓由纹理 alpha 决定：若省略裁切，透明背景会遮挡深度相机。
    shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(),'a')
    shader.CreateInput('opacityThreshold',Sdf.ValueTypeNames.Float).Set(float(material.alphaCutoff or .5))
    shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    return str(path)
