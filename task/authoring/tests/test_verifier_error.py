from pathlib import Path
import sys
import json
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tests'))
from verifier.mesh_checks import validate_mesh,load_config
from verifier.error import energy_error,exact_gradient,green_energy
from verifier.fem_ref import Problem
ROOT=Path(__file__).resolve().parents[2]


def mesh():
    data=json.loads((ROOT/'environment/data/initial_mesh.json').read_text(encoding='utf-8'))
    return validate_mesh(**data)


def test_zero_fem_norm_and_convergence():
    m=mesh();u=np.zeros(len(m.vertices))
    a=energy_error(m,u)
    b=energy_error(m,u,order=20,max_cell_diameter=.0625,corner_panels=6)
    assert a.relative==pytest.approx(1.)
    assert abs(a.reference_squared-b.reference_squared)<1e-10
    assert abs(b.reference_squared-1.632536170000)<1e-10


def test_green_identity():
    m=mesh()
    direct=energy_error(m,np.zeros(len(m.vertices)),order=20).reference_squared
    assert abs(direct-green_energy())<1e-10


def test_gradient_difference():
    p=Problem.from_config();x=np.array([[-.4,.6],[-.7,-.3],[.2,.5]])
    h=1e-6
    finite=np.stack([(p.g_D(x+h*d)-p.g_D(x-h*d))/(2*h) for d in np.eye(2)],axis=-1)
    np.testing.assert_allclose(exact_gradient(x,p),finite,rtol=2e-8,atol=1e-9)
    with pytest.raises(ValueError,match='origin'): exact_gradient([[0.,0.]])


def test_singular_norm_green():
    config=load_config();config['problem']['A']=0.
    direct=energy_error(mesh(),np.zeros(8),config,order=20).reference_squared
    assert abs(direct-green_energy(config))<1e-11


@pytest.mark.parametrize('options',[{'order':0},{'corner_panels':0},{'batch_size':False},{'max_cell_diameter':-1}])
def test_bad_options(options):
    with pytest.raises(ValueError): energy_error(mesh(),np.zeros(8),**options)


def test_requires_validated_mesh():
    with pytest.raises(TypeError): energy_error(dict(vertices=[],triangles=[]),np.zeros(8))


@pytest.mark.parametrize('values',[np.zeros(7),np.full(8,np.nan),np.full(8,np.inf),
                                   np.array(['0']*8),np.zeros((8,1))])
def test_bad_nodal_values(values):
    with pytest.raises(ValueError,match='nodal_values'): energy_error(mesh(),values)


def test_matches_supplied_fem_error_on_oracle():
    # Authoring-only cross-check (TODO 2.8): the verifier's error term, not
    # just the reference norm, must match the supplied fem.error on the oracle.
    path=ROOT/'authoring/evidence/calibration/submission.json'
    if not path.exists(): pytest.skip('authoring evidence not available')
    sys.path.insert(0,str(ROOT/'environment/data'))
    fem_error=pytest.importorskip('fem.error')
    from fem.mesh import Mesh
    from verifier.mesh_checks import load_submission
    from verifier.fem_ref import solve
    _,m=load_submission(path,max_vertices=11736)
    u=solve(m)
    ours=energy_error(m,u)
    theirs=fem_error.energy_error(Mesh(m.vertices,m.triangles),u)
    assert ours.relative==pytest.approx(theirs.relative,rel=1e-10,abs=0)
    assert ours.relative==pytest.approx(.0499458967,abs=1e-9)
    assert ours.reference_squared==pytest.approx(theirs.reference_norm**2,rel=1e-10)
