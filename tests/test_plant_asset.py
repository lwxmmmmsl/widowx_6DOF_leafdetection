"""Validate USD topology/UV/alpha separately from GPU rendering."""
import numpy as np
import pytest
Usd=pytest.importorskip('pxr.Usd')
from pxr import UsdGeom
from widowx_codex.plant import convert_glb
from widowx_codex.common import ROOT

def test_glb_usd_preserves_geometry_uv_alpha(tmp_path):
    import trimesh
    original=trimesh.load(ROOT/'assets/tomato_plant.glb').to_geometry()
    path=convert_glb(ROOT/'assets/tomato_plant.glb',tmp_path)
    stage=Usd.Stage.Open(path); mesh=UsdGeom.Mesh(stage.GetPrimAtPath('/Tomato/Leaves'))
    np.testing.assert_allclose(mesh.GetPointsAttr().Get(),original.vertices,rtol=1e-6,atol=1e-5)
    np.testing.assert_array_equal(mesh.GetFaceVertexIndicesAttr().Get(),original.faces.flatten())
    np.testing.assert_allclose(UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').Get(),original.visual.uv,atol=1e-7)
    assert mesh.GetDoubleSidedAttr().Get()
    shader=stage.GetPrimAtPath('/Tomato/Material/Surface')
    assert shader.GetAttribute('inputs:opacityThreshold').Get()==pytest.approx(original.visual.material.alphaCutoff)
    assert shader.GetAttribute('inputs:opacity').HasAuthoredConnections()
