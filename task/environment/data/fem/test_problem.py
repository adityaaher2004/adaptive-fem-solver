import numpy as np
import pytest
from fem import problem as p


def test_values_and_shapes():
    assert p.u_s([0,0]) == 0
    assert p.u_s([1,0]) == 0
    assert p.u_s([0,-1]) == pytest.approx(1)
    assert p.DEFAULT_PROBLEM.u_p(p.X0) == .5
    assert p.f(p.X0) == 2000
    for shape in [(2,), (3,2), (2,3,2), (0,2)]:
        x = np.full(shape, -.3)
        assert p.u(x).shape == shape[:-1]
        assert p.grad_u(x).shape == shape
    with pytest.raises(ValueError, match='undefined'):
        p.grad_u([[0,0],[-.1,.1]])


@pytest.mark.parametrize('x', [[-.4,.6],[-.43,.61],[-.1,-.2],[.3,.4],[-.7,0]])
def test_independent_numerical_derivatives(x):
    x = np.array(x)
    h = 2e-6
    eye = np.eye(2)
    grad = [(p.u(x+h*d)-p.u(x-h*d))/(2*h) for d in eye]
    np.testing.assert_allclose(p.grad_u(x), grad, rtol=2e-7, atol=2e-8)
    lap = sum((-p.grad_u(x+2*h*d)[k]+8*p.grad_u(x+h*d)[k]
               -8*p.grad_u(x-h*d)[k]+p.grad_u(x-2*h*d)[k])/(12*h)
              for k,d in enumerate(eye))
    assert -lap == pytest.approx(p.f(x), rel=2e-7, abs=2e-7)


def test_boundary_and_branch():
    points = np.array(p.DOMAIN_CORNERS)
    assert p.is_dirichlet(points).all()
    np.testing.assert_array_equal(p.g_D(points), p.u(points))
    n = [[0,-.25],[0,-.75]]
    assert p.is_neumann(n).all()
    np.testing.assert_array_equal(p.grad_u_s(n)[:,0], 0)
    # Non-negligible flux variant verifies sign and derivative, unlike defaults.
    q = p.PoissonProblem(alpha=3, x0=(-.3,-.4))
    np.testing.assert_allclose(q.g_N(n), q.grad_u(n)[:,0], rtol=1e-15)
    assert (q.g_N(n) < 0).all()
    np.testing.assert_allclose(p.u([[-.5,1e-12],[-.5,-1e-12]])[0],
                               p.u([[-.5,1e-12],[-.5,-1e-12]])[1], atol=1e-11)
    for x in [[0,0],[0,-1],[.2,.2]]:
        with pytest.raises(ValueError):
            p.g_N(x)
    with pytest.raises(ValueError):
        p.g_D([0,-.5])


def test_mesh_boundary_refinement():
    mesh = p.initial_mesh()
    for level in range(4):
        d, n = p.boundary_masks(mesh)
        assert mesh.area == 3
        assert n.sum() == 2**level
        assert (mesh.boundary_vertex_mask & ~d).sum() == 2**level-1
        lengths = np.linalg.norm(np.diff(mesh.vertices[mesh.edges[n]],axis=1),axis=2)
        assert lengths.sum() == 1
        assert p.in_domain(mesh.vertices).all()
        mesh = mesh.uniform_refine()


def test_singular_energy_identity():
    # Polar Gauss integration after r=R*t**3 removes the singular power.
    z,w = np.polynomial.legendre.leggauss(20)
    t, wt = (z+1)/2,w/2
    theta, wa = (z+1)*3*np.pi/4,w*3*np.pi/4
    R = .7
    r = R*t**3
    xy = r[:,None,None]*np.stack((np.cos(theta),np.sin(theta)),axis=-1)
    density = np.sum(p.grad_u_s(xy)**2,axis=-1)
    integral = np.sum(density*(r*3*R*t*t*wt)[:,None]*wa)
    assert integral == pytest.approx(np.pi/4*R**(2/3), rel=2e-14)


@pytest.mark.parametrize('x', [[.2,-.2],[2,0],[0,np.nan],[1],[[1,2,3]]])
def test_invalid_points(x):
    with pytest.raises(ValueError):
        p.u(x)


@pytest.mark.parametrize('kwargs', [dict(alpha=0),dict(alpha=-1),dict(A=np.inf),
                                    dict(x0=(.2,-.3)),dict(x0=(0,0))])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        p.PoissonProblem(**kwargs)


def test_initial_mesh_json_matches_source():
    # environment/data/initial_mesh.json is generated from initial_mesh() by
    # authoring/provenance/make_initial_mesh.py; this guards against drift.
    from pathlib import Path
    from fem.io import load_mesh
    stored = load_mesh(Path(p.__file__).resolve().parents[1] / 'initial_mesh.json')
    mesh = p.initial_mesh()
    np.testing.assert_array_equal(stored.vertices, mesh.vertices)
    np.testing.assert_array_equal(stored.triangles, mesh.triangles)


def test_initial_mesh_nvb_labelling():
    # Refinement edge = local vertices 1-2 (opposite newest vertex 0). It must
    # be each triangle's longest edge and be shared by both incident triangles.
    mesh = p.initial_mesh()
    v, t = mesh.vertices, mesh.triangles
    lengths = np.linalg.norm(v[t[:, [1, 2, 0]]] - v[t[:, [2, 0, 1]]], axis=2)
    assert np.all(lengths[:, 0] > lengths[:, 1:].max(axis=1))
    refinement = mesh.triangle_edges[:, 1]
    for k, e in enumerate(refinement):
        pair = mesh.edge_triangles[e]
        assert (pair >= 0).all(), 'refinement edges are interior diagonals'
        assert refinement[pair[pair != k][0]] == e
