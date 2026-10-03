import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tests'))
from verifier.mesh_checks import validate_mesh,load_config
from verifier.fem_ref import Problem,assemble,solve,element_gradients

V=[[-1,-1],[0,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]]
T=[[1,3,0],[2,0,3],[3,6,2],[5,2,6],[4,7,3],[6,3,7]]


def mesh(refine=False):
    if not refine: return validate_mesh(V,T)
    v=list(V); mid={}; triangles=[]
    for a,b,c in T:
        def midpoint(x,y):
            e=tuple(sorted((x,y)))
            if e not in mid:
                mid[e]=len(v);v.append(((np.array(V[x])+V[y])/2).tolist())
            return mid[e]
        ab,bc,ca=midpoint(a,b),midpoint(b,c),midpoint(c,a)
        triangles.extend([(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)])
    return validate_mesh(v,triangles)


def test_analytic_stiffness_load():
    m=mesh()
    s=assemble(m,f=lambda p:1,g_N=lambda p:0,max_cell_diameter=2)
    expected=np.zeros_like(s.load)
    for t,area in zip(m.triangles,m.areas): expected[t]+=area/3
    np.testing.assert_allclose(s.load,expected,atol=1e-14)
    np.testing.assert_allclose(s.stiffness@np.ones(len(V)),0,atol=1e-14)
    np.testing.assert_allclose(s.stiffness.toarray(),s.stiffness.toarray().T)
    # First triangle: vertices (0,-1),(0,0),(-1,-1).
    np.testing.assert_allclose(element_gradients(m)[0],[[1,-1],[0,1],[-1,0]])
    g=element_gradients(m)[0]
    np.testing.assert_allclose(.5*g@g.T,[[1,-.5,-.5],[-.5,.5,0],[-.5,0,.5]])


@pytest.mark.parametrize('batch',[1,128])
def test_mixed_affine_patch(batch):
    m=mesh(True)
    exact=lambda p:2+3*p[...,0]-4*p[...,1]
    u=solve(m,f=lambda p:0,g_D=exact,g_N=lambda p:3,max_cell_diameter=2,batch_size=batch)
    np.testing.assert_allclose(u,exact(m.vertices),atol=2e-14)
    np.testing.assert_array_equal(u[m.dirichlet_vertex_mask],exact(m.vertices[m.dirichlet_vertex_mask]))


def test_linear_neumann_load():
    m=mesh()
    s=assemble(m,f=lambda p:0,g_N=lambda p:2+p[...,1],max_cell_diameter=2)
    expected=np.zeros(len(V));expected[1]=2/3;expected[3]=5/6
    np.testing.assert_allclose(s.load,expected,atol=1e-14)


def test_all_dirichlet():
    m=mesh()
    u=solve(m,max_cell_diameter=2)
    np.testing.assert_array_equal(u,Problem.from_config().g_D(m.vertices))


def test_gaussian_load_convergence():
    m=mesh(True)
    a=assemble(m)
    b=assemble(m,volume_order=12,max_cell_diameter=.015)
    np.testing.assert_allclose(a.load,b.load,atol=2e-10,rtol=2e-9)
    assert np.linalg.norm(a.load)>.01


def test_config_flux_and_source():
    config=load_config();config['problem']=dict(A=.7,alpha=3.,x0=[-.3,-.4])
    p=Problem.from_config(config)
    assert p.f(p.x0)==pytest.approx(4*3*.7)
    points=np.array([[0.,-.5],[0.,-.2]])
    h=1e-6; offset=np.array([h,0.])
    finite=(p.peak(points+offset)-p.peak(points-offset))/(2*h)
    np.testing.assert_allclose(p.g_N(points),finite,rtol=1e-9)
    assert np.linalg.norm(p.g_N(points))>.1


@pytest.mark.parametrize('block',[dict(A=True,alpha=1,x0=[-.3,.4]),dict(A=1,alpha=0,x0=[-.3,.4]),dict(A=1,alpha=1,x0=[.3,-.4])])
def test_invalid_parameters(block):
    with pytest.raises(ValueError): Problem(**block)


def uniform_mesh(levels):
    """Independent red refinement of the base mesh (no fem package)."""
    v=[list(map(float,p)) for p in V]; t=[tuple(x) for x in T]
    for _ in range(levels):
        mid={}; new=[]
        def midpoint(x,y):
            e=(min(x,y),max(x,y))
            if e not in mid:
                mid[e]=len(v); v.append([(v[x][0]+v[y][0])/2,(v[x][1]+v[y][1])/2])
            return mid[e]
        for a,b,c in t:
            ab,bc,ca=midpoint(a,b),midpoint(b,c),midpoint(c,a)
            new.extend([(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)])
        t=new
    return validate_mesh(v,t)


def test_smooth_mixed_convergence():
    # u = sin(2x)cos(y) + x with NONZERO Neumann flux du/dx on x=0: a wrong
    # scale in the volume or Neumann load breaks the O(h) energy rate.
    ue=lambda p: np.sin(2*p[...,0])*np.cos(p[...,1])+p[...,0]
    ux=lambda p: 2*np.cos(2*p[...,0])*np.cos(p[...,1])+1
    uy=lambda p: -np.sin(2*p[...,0])*np.sin(p[...,1])
    fe=lambda p: 5*np.sin(2*p[...,0])*np.cos(p[...,1])
    z,w=np.polynomial.legendre.leggauss(6); z,w=(z+1)/2,w/2
    R,S=np.meshgrid(z,z,indexing='ij')
    weights=(w[:,None]*w[None,:]*(1-R)).ravel()
    bary=np.stack(((1-R)*(1-S),R,(1-R)*S),axis=-1).reshape(-1,3)
    errors=[]
    for level in (2,3,4):
        m=uniform_mesh(level)
        u=solve(m,f=fe,g_D=ue,g_N=ux,max_cell_diameter=2)
        gh=np.einsum('ti,tik->tk',u[m.triangles],element_gradients(m))
        pts=np.einsum('qi,tij->tqj',bary,m.vertices[m.triangles])
        e2=2*m.areas[:,None]*weights*((ux(pts)-gh[:,None,0])**2+(uy(pts)-gh[:,None,1])**2)
        errors.append(np.sqrt(e2.sum()))
    ratios=np.array(errors[:-1])/np.array(errors[1:])
    assert np.all((ratios>1.9)&(ratios<2.1)),(errors,ratios)


@pytest.mark.parametrize('options',[dict(volume_order=0),dict(edge_order=0),dict(batch_size=0),
    dict(volume_order=True),dict(batch_size=1.5),dict(max_cell_diameter=0.),
    dict(max_cell_diameter=-1.),dict(max_cell_diameter=np.nan),dict(max_cell_diameter=np.inf)])
def test_invalid_options(options):
    with pytest.raises(ValueError): assemble(mesh(),**options)


def test_requires_validated_mesh():
    with pytest.raises(TypeError): assemble(dict(vertices=V,triangles=T))


def test_matches_supplied_fem_on_oracle_mesh():
    # Authoring-only cross-check (TODO 2.8): the independent verifier solve
    # must reproduce the supplied fem package's solution on the oracle mesh.
    root=Path(__file__).resolve().parents[2]
    path=root/'authoring/evidence/calibration/submission.json'
    if not path.exists(): pytest.skip('authoring evidence not available')
    sys.path.insert(0,str(root/'environment/data'))
    fem_io=pytest.importorskip('fem.io')
    from fem.assembly import assemble_system
    from fem.solver import solve as fem_solve
    from verifier.mesh_checks import load_submission
    _,m=load_submission(path,max_vertices=40000)
    reference=fem_solve(assemble_system(fem_io.load_mesh(path),load_refinements=4))
    u=solve(m)
    assert m.dof==5383
    np.testing.assert_allclose(u,reference,rtol=0,atol=1e-12)
