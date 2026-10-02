import numpy as np
import pytest
from scipy.sparse.linalg import spsolve
from fem.mesh import Mesh
from fem import problem
from fem.assembly import (element_geometry, boundary_masks, assemble_stiffness,
    assemble_load, eliminate_dirichlet, assemble_system)
from fem.basis import physical_gradients
from fem.quadrature import triangle_rule


def lmesh(level=0):
    mesh = problem.initial_mesh()
    for _ in range(level):
        mesh = mesh.uniform_refine()
    return mesh


def test_reference_triangle():
    m = Mesh([[0,0],[1,0],[0,1]],[[0,1,2]])
    np.testing.assert_allclose(assemble_stiffness(m).toarray(),
        [[1,-.5,-.5],[-.5,.5,0],[-.5,0,.5]])
    np.testing.assert_allclose(assemble_load(m,lambda p: 1,None),np.full(3,1/6))
    # integral x*phi_i = (1/24,1/12,1/24)
    np.testing.assert_allclose(assemble_load(m,lambda p:p[...,0],None),[1/24,1/12,1/24])


@pytest.mark.parametrize('batch',[1,17,4096])
def test_skew_geometry_and_scatter(batch):
    m = Mesh([[0,0],[2,.3],[2.6,1.7],[.6,1.4]],[[0,1,2],[0,2,3]])
    _,_,det,grad = element_geometry(m)
    expected = np.zeros((4,4))
    for ids,area,g in zip(m.triangles,m.areas,grad):
        np.testing.assert_allclose(g,physical_gradients(m.vertices[ids]))
        expected[np.ix_(ids,ids)] += area*g@g.T
    K = assemble_stiffness(m,batch_size=batch)
    np.testing.assert_allclose(K.toarray(),expected,atol=1e-15)
    np.testing.assert_allclose(K@np.ones(4),0,atol=1e-15)
    assert K.format == 'csr'
    np.testing.assert_allclose(det,2*m.areas)


@pytest.mark.parametrize('level',[0,1,2])
def test_affine_mixed_patch(level):
    m = lmesh(level)
    D,N = boundary_masks(m)
    exact = lambda p: 2+3*p[...,0]-4*p[...,1]
    system = assemble_system(m,lambda p:0,exact,lambda p:3,batch_size=7)
    u = system.expand(spsolve(system.matrix,system.rhs))
    np.testing.assert_allclose(u,exact(m.vertices),atol=2e-14)
    assert len(system.free_vertices) == np.count_nonzero(~D)
    assert np.all(np.linalg.eigvalsh(system.matrix.toarray()) > 0) if len(system.rhs) else True
    for c in [[0,0],[0,-1]]:
        assert D[np.all(m.vertices == c,axis=1)].all()
    assert not D[(m.vertices[:,0]==0)&(m.vertices[:,1]>-1)&(m.vertices[:,1]<0)].any()


def test_neumann_endpoints_and_linear_flux():
    m = lmesh()
    _,N = boundary_masks(m)
    b = assemble_load(m,lambda p:0,lambda p:2+p[...,1],neumann_edge_mask=N,edge_npoints=2)
    expected = np.zeros(m.n_vertices)
    expected[1],expected[3] = 2/3,5/6
    np.testing.assert_allclose(b,expected,atol=1e-15)


@pytest.mark.parametrize('refinements',[0,1,2,3])
def test_composite_polynomial_and_batches(refinements):
    m = lmesh(1)
    f = lambda p: p[...,0]**2+2*p[...,1]**2
    a = assemble_load(m,f,None,triangle_degree=4,load_refinements=refinements,batch_size=7)
    b = assemble_load(m,f,None,triangle_degree=4)
    np.testing.assert_allclose(a,b,atol=1e-14)
    assert abs(a.sum()-3) < 2e-14


def test_elimination_no_mutation():
    m = lmesh(1)
    K = assemble_stiffness(m)
    b = np.arange(m.n_vertices,dtype=float)
    K0,b0 = K.copy(),b.copy()
    D,_ = boundary_masks(m)
    s = eliminate_dirichlet(m,K,b,lambda p:2)
    F = np.flatnonzero(~D)
    np.testing.assert_allclose(s.rhs,b[F]-K[F][:,D]@np.full(D.sum(),2))
    np.testing.assert_array_equal(K.toarray(),K0.toarray())
    np.testing.assert_array_equal(b,b0)


def test_all_dirichlet_empty_system():
    m = Mesh([[0,0],[1,0],[0,1]],[[0,1,2]])
    s = eliminate_dirichlet(m,assemble_stiffness(m),np.zeros(3),lambda p:7,
                           dirichlet_vertex_mask=np.ones(3,bool))
    assert s.matrix.shape == (0,0)
    np.testing.assert_array_equal(s.expand([]),[7,7,7])


def test_default_problem_runs():
    m = lmesh(1)
    s = assemble_system(m,load_refinements=2)
    assert np.isfinite(s.expand(spsolve(s.matrix,s.rhs))).all()


@pytest.mark.parametrize('kind',['interior_N','interior_D','nan','shape','batch','level'])
def test_invalid_inputs(kind):
    m = lmesh(1)
    with pytest.raises((ValueError,TypeError)):
        if kind == 'interior_N':
            assemble_load(m,lambda p:0,neumann_edge_mask=np.ones(m.n_edges,bool))
        elif kind == 'interior_D':
            eliminate_dirichlet(m,assemble_stiffness(m),np.zeros(m.n_vertices),
                                dirichlet_vertex_mask=np.ones(m.n_vertices,bool))
        elif kind == 'nan':
            assemble_load(m,lambda p:np.nan,None)
        elif kind == 'shape':
            assemble_load(m,lambda p:np.zeros((3,3,3)),None)
        elif kind == 'batch':
            assemble_stiffness(m,batch_size=0)
        else:
            assemble_load(m,load_refinements=-1)


def test_smooth_mixed_convergence():
    # u = sin(2x)cos(y) + x with a NONZERO Neumann flux on x=0 (n=(+1,0)),
    # so a sign or scaling error in the Neumann load breaks the O(h) rate.
    u = lambda p: np.sin(2*p[...,0])*np.cos(p[...,1]) + p[...,0]
    ux = lambda p: 2*np.cos(2*p[...,0])*np.cos(p[...,1]) + 1
    uy = lambda p: -np.sin(2*p[...,0])*np.sin(p[...,1])
    f = lambda p: 5*np.sin(2*p[...,0])*np.cos(p[...,1])
    xy, w = triangle_rule(6)
    errors = []
    for level in (2,3,4):
        m = lmesh(level)
        s = assemble_system(m,f,u,ux)
        uh = s.expand(spsolve(s.matrix.tocsc(),s.rhs))
        origins,J,det,grad = element_geometry(m)
        gh = np.einsum('ti,tik->tk',uh[m.triangles],grad)
        points = origins[:,None] + np.einsum('qi,tji->tqj',xy,J)
        e2 = det[:,None]*w*((ux(points)-gh[:,None,0])**2 + (uy(points)-gh[:,None,1])**2)
        errors.append(np.sqrt(e2.sum()))
    ratios = np.array(errors[:-1])/np.array(errors[1:])
    assert np.all((ratios > 1.9) & (ratios < 2.1)), (errors, ratios)
    assert errors[-1] < 0.1


def test_peak_load_needs_subdivision_on_coarse_mesh():
    # The Gaussian is ~exp(-150) on the boundary, so int f = int f*x = 0
    # to far below roundoff. P1 reproduces 1 and x, hence an accurate load
    # satisfies sum(b) = 0 and b @ vertices = 0.
    m = lmesh(2)
    coarse = assemble_load(m,problem.f,None)
    assert abs(coarse.sum()) > 1      # default rule misses the peak (documented trap)
    fine = assemble_load(m,problem.f,None,load_refinements=4)
    finer = assemble_load(m,problem.f,None,load_refinements=5)
    assert abs(fine.sum()) < 1e-12
    np.testing.assert_allclose(fine @ m.vertices, 0, atol=1e-12)
    np.testing.assert_allclose(fine, finer, rtol=0, atol=1e-12)
