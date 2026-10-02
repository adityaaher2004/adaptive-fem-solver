import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem.assembly import DirichletSystem, assemble_system, boundary_masks
from fem import problem
from fem.mesh import Mesh
from fem.solver import solve


@pytest.mark.parametrize('level', [1, 2])
def test_mixed_affine_solution_and_preservation(level):
    mesh = problem.initial_mesh()
    for _ in range(level):
        mesh = mesh.uniform_refine()
    exact = lambda p: 2 + 3*p[..., 0] - 4*p[..., 1]
    system = assemble_system(mesh, lambda p: 0, exact, lambda p: 3)
    matrix, rhs, lifting = system.matrix.copy(), system.rhs.copy(), system.lifting.copy()
    u = solve(system)
    assert u.shape == (mesh.n_vertices,)
    np.testing.assert_allclose(u, exact(mesh.vertices), atol=3e-14)
    D, _ = boundary_masks(mesh)
    np.testing.assert_array_equal(u[D], exact(mesh.vertices[D]))
    np.testing.assert_allclose(system.matrix @ u[system.free_vertices], rhs, atol=3e-14)
    np.testing.assert_array_equal(system.matrix.toarray(), matrix.toarray())
    np.testing.assert_array_equal(system.rhs, rhs)
    np.testing.assert_array_equal(system.lifting, lifting)
    assert not np.shares_memory(u, system.lifting)


def test_all_dirichlet(monkeypatch):
    mesh = Mesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]])
    system = assemble_system(mesh, lambda p: 0, lambda p: 7, None,
                             dirichlet_vertex_mask=np.ones(3, dtype=bool))
    def unexpected(*args, **kwargs):
        raise AssertionError('empty system must not call spsolve')
    monkeypatch.setattr('fem.solver.spsolve', unexpected)
    np.testing.assert_array_equal(solve(system), [7, 7, 7])


def test_singular_system():
    system = DirichletSystem(csr_matrix((1, 1)), np.ones(1), np.array([0]), np.zeros(1))
    with pytest.raises(np.linalg.LinAlgError, match='singular'):
        solve(system)


def test_one_free_vertex():
    system = DirichletSystem(csr_matrix([[2.]]), np.array([6.]),
                             np.array([1]), np.array([5., 0., 7.]))
    np.testing.assert_array_equal(solve(system), [5., 3., 7.])


@pytest.mark.parametrize('bad', [None, (csr_matrix([[2.]]), np.ones(1)), np.eye(1)])
def test_rejects_non_system(bad):
    with pytest.raises(TypeError):
        solve(bad)


@pytest.mark.parametrize('field', ['matrix', 'rhs'])
@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_rejects_nonfinite(field, value):
    matrix = csr_matrix([[2., 0.], [0., 3.]])
    rhs = np.ones(2)
    if field == 'matrix':
        matrix.data[0] = value
    else:
        rhs[1] = value
    system = DirichletSystem(matrix, rhs, np.arange(2), np.zeros(2))
    with pytest.raises(ValueError, match='finite'):
        solve(system)
