"""Independent verifier tests; do not import the supplied FEM package."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tests'))
from verifier.mesh_checks import (validate_mesh,validate_submission,load_submission,
                                 load_config,MeshValidationError,_geometry_checks,_orientation)

V=[[-1,-1],[0,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]]
T=[[1,3,0],[2,0,3],[3,6,2],[5,2,6],[4,7,3],[6,3,7]]


def submission():
    return dict(vertices=copy.deepcopy(V),triangles=copy.deepcopy(T),dof=0,
                reported_error=.9,history=[dict(iter=0,dof=0,estimator=1.)])


def test_initial_and_immutability():
    m=validate_mesh(V,T)
    assert m.area==3 and m.dof==0
    assert m.neumann_edge_mask.sum()==1
    assert m.boundary_edge_mask.sum()==8
    assert m.dirichlet_vertex_mask.all()
    assert m.edges.shape==(13,2)
    with pytest.raises(ValueError): m.vertices.setflags(write=True)


def test_neumann_midpoint_free():
    v=V+[[0,-.5]]
    t=[[8,3,0],[1,8,0]]+T[1:]
    m=validate_mesh(v,t)
    assert m.dof==1
    assert m.neumann_edge_mask.sum()==2
    assert m.dirichlet_vertex_mask[1] and m.dirichlet_vertex_mask[3]


# Each case names the check that must reject it (match=), so a case cannot
# pass because an unrelated check fired first.
INVALID = {
    'clockwise': ('CCW orientation', lambda v,t: t[0].__setitem__(slice(1,None), t[0][1:][::-1])),
    # Collinear triangle on the bottom side; every vertex stays used.
    'degenerate': ('positive area', lambda v,t: (v.append([-.5,-1]), t.append([0,8,1]))),
    'repeated': ('repeats a vertex', lambda v,t: t[0].__setitem__(1, t[0][0])),
    'float_index': ('integers', lambda v,t: t[0].__setitem__(0, 1.)),
    'bool_index': ('integers', lambda v,t: t[0].__setitem__(0, True)),
    'negative': ('out of range', lambda v,t: t[0].__setitem__(0, -1)),
    'out_of_range': ('out of range', lambda v,t: t[0].__setitem__(0, 999)),
    'nan': ('finite', lambda v,t: v[0].__setitem__(0, float('nan'))),
    'inf': ('finite', lambda v,t: v[0].__setitem__(0, float('inf'))),
    'bool_vertex': ('real numbers', lambda v,t: v[1].__setitem__(0, False)),
    'duplicate_vertex': ('duplicate vertices', lambda v,t: v.__setitem__(0, v[1].copy())),
    'duplicate_triangle': ('duplicate triangles', lambda v,t: t.append(t[0].copy())),
    'unused': ('unused vertices', lambda v,t: v.append([-.2,.2])),
    'cutout': ('cut-out', lambda v,t: v.__setitem__(0, [.5,-.5])),
    'outside': ('outside', lambda v,t: v.__setitem__(0, [-2,-1])),
    # Removing [2,0,3] leaves all vertices used: a genuine hole.
    'hole': ('total area', lambda v,t: t.pop(1)),
    'missing_corner': ('missing domain corner', lambda v,t: v.__setitem__(0, [-.9,-1])),
    # A hanging node leaves a once-used interior edge, caught topologically
    # before the geometric check (tested directly below).
    'hanging': ('not on domain boundary',
                lambda v,t: (v.append([-.5,-.5]), t.__setitem__(slice(0,1), [[8,1,3],[8,0,1]]))),
}


@pytest.mark.parametrize('kind', INVALID)
def test_invalid_mesh(kind):
    v,t=copy.deepcopy(V),copy.deepcopy(T)
    message, change = INVALID[kind]
    change(v,t)
    with pytest.raises(MeshValidationError, match=message): validate_mesh(v,t)


def test_size_limit_before_geometry():
    v = V + [[-.9+.001*k, .9] for k in range(10)]
    with pytest.raises(MeshValidationError, match='too many vertices'):
        validate_mesh(v, T, max_vertices=12)
    # 18 triangles on 8 vertices exceeds 2*8 (valid meshes have T < 2V).
    with pytest.raises(MeshValidationError, match='too many triangles'):
        validate_mesh(V, T*3, max_vertices=8)
    assert validate_mesh(V, T, max_vertices=8).dof == 0


def test_geometric_hanging_node():
    with pytest.raises(MeshValidationError,match='hanging'):
        _geometry_checks(np.array([[0.,0.],[1.,0.],[.5,0.]]),np.array([[0,1]]))


def test_geometric_crossing():
    with pytest.raises(MeshValidationError,match='crossing'):
        _geometry_checks(np.array([[0.,0.],[1.,1.],[0.,1.],[1.,0.]]),np.array([[0,1],[2,3]]))


def _brute_force_crossing(v, edges):
    a, b = v[edges[:,0]], v[edges[:,1]]
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            if len({*edges[i], *edges[j]}) < 4:
                continue
            if (_orientation(a[i],b[i],a[j])*_orientation(a[i],b[i],b[j]) < 0 and
                    _orientation(a[j],b[j],a[i])*_orientation(a[j],b[j],b[i]) < 0):
                return True
    return False


def test_crossing_search_matches_brute_force():
    # Mix tiny and long segments: the pruned search must still find crossings
    # between a long edge and a short one far from the long edge's midpoint.
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(4, 14)
        v = rng.uniform(-1, 1, (2*n, 2))
        v[:n] *= rng.choice([1e-3, 1e-1, 1.], size=(n, 1))
        edges = np.arange(2*n).reshape(-1, 2)
        try:
            _geometry_checks(v, edges)
            found = False
        except MeshValidationError as exc:
            found = 'crossing' in str(exc)
        assert found == _brute_force_crossing(v, edges)


def test_near_collinear_exact_predicate():
    assert _orientation(np.array([0.,0.]),np.array([1.,1.]),np.array([.5,np.nextafter(.5,1.)]))==1
    assert _orientation(np.array([0.,0.]),np.array([1.,1.]),np.array([.5,.5]))==0


@pytest.mark.parametrize('change',[{'dof':True},{'reported_error':float('nan')},
    {'history':[{}]},{'history':[{'iter':0,'dof':0,'estimator':-1}]},
    {'history':None},{'triangles':None},{'extra':float('inf')}])
def test_schema(change):
    s=submission();s.update(change)
    with pytest.raises(MeshValidationError): validate_submission(s)


def test_extra_keys_and_claims_are_not_certified():
    s=submission();s.update(extra='allowed',dof=5,reported_error=0.)
    assert validate_submission(s).dof==0


def test_json_duplicate_and_overflow(tmp_path):
    p=tmp_path/'s.json'
    for text in ['{"dof":0,"dof":1}', '{"x":NaN}',json.dumps(submission()).replace('0.9','1e999')]:
        p.write_text(text,encoding='utf-8')
        with pytest.raises(MeshValidationError): load_submission(p)


def test_config():
    c=load_config()
    assert c['dof_budget']==10000 and c['max_iter']==54
    assert c['problem']=={'A':.5,'alpha':1000.,'x0':[-.43,.61]}


@pytest.mark.parametrize('problem',[None,{},{'A':.5,'alpha':1000.},
    {'A':.5,'alpha':0.,'x0':[-.43,.61]},{'A':True,'alpha':1000.,'x0':[-.43,.61]},
    {'A':.5,'alpha':1000.,'x0':[.5,-.5]},{'A':.5,'alpha':1000.,'x0':[-.43]},
    {'A':.5,'alpha':1000.,'x0':[-1.,.5]}])
def test_config_problem_block(tmp_path,problem):
    config=json.loads((Path(load_config.__defaults__[0])).read_text(encoding='utf-8'))
    if problem is None: del config['problem']
    else: config['problem']=problem
    path=tmp_path/'config.json'
    path.write_text(json.dumps(config),encoding='utf-8')
    with pytest.raises(MeshValidationError,match='problem'): load_config(path)


def test_oracle():
    # Authoring checkout only: authoring/ is not present in the verifier image.
    path=Path(__file__).resolve().parents[2]/'authoring/evidence/calibration/submission.json'
    if not path.exists(): pytest.skip('authoring evidence not available')
    data,m=load_submission(path,max_vertices=40000)
    assert m.dof==5383==data['dof']
    assert m.area==pytest.approx(3)
