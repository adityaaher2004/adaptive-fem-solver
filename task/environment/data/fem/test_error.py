import numpy as np
import pytest

from fem import problem
from fem.assembly import assemble_system
from fem.error import energy_error
from fem.example_uniform import main as example_main
from fem.solver import solve

SINGULAR = problem.PoissonProblem(A=0)


def lmesh(level):
    mesh = problem.initial_mesh()
    for _ in range(level):
        mesh = mesh.uniform_refine()
    return mesh


def green_energy(pb, n=100):
    """||grad u||^2 = oint u du/dn for harmonic u (independent 1D reference).

    u_s = 0 on the theta=0 edge and du_s/dn = 0 on Gamma_N, so only the four
    outer sides contribute; u_s is smooth there, so Gauss-Legendre is exact
    to roundoff.
    """
    z, w = np.polynomial.legendre.leggauss(n)
    total = 0.
    for a, b, normal in [((-1, -1), (0, -1), (0, -1)), ((-1, 1), (-1, -1), (-1, 0)),
                         ((1, 1), (-1, 1), (0, 1)), ((1, 0), (1, 1), (1, 0))]:
        a, b, normal = map(np.array, (a, b, normal))
        for lo, hi in [(0, .5), (.5, 1)]:
            t = lo + (hi-lo)*(z+1)/2
            points = a + t[:, None]*(b-a)
            total += np.linalg.norm(b-a)*(hi-lo)/2*np.sum(
                w*pb.u(points)*(pb.grad_u(points) @ normal))
    return total


def test_reference_norm_matches_green_identity():
    # Without the t**3 corner map this integrand is O(t**(-1/3)) and Gauss
    # quadrature would miss the 1e-9 tolerance by several orders.
    mesh = lmesh(2)
    uh = solve(assemble_system(mesh, SINGULAR.f, SINGULAR.g_D, SINGULAR.g_N))
    result = energy_error(mesh, uh, SINGULAR.grad_u)
    assert result.reference_norm**2 == pytest.approx(green_energy(SINGULAR), rel=1e-9)
    assert result.relative == pytest.approx(result.absolute/result.reference_norm, rel=1e-15)
    assert 0 < result.relative < 1


def test_linear_solution_has_zero_error():
    # The interpolant of a linear u is exact, including on the corner cells
    # (origin is a vertex), so the error vanishes and ||grad u||^2 = |c|^2 * 3.
    mesh = lmesh(1)
    c = np.array([3., -4.])
    exact = lambda p: np.broadcast_to(c, np.shape(p)).copy()
    nodal = 2 + mesh.vertices @ c
    result = energy_error(mesh, nodal, exact)
    assert result.absolute < 1e-12
    assert result.reference_norm**2 == pytest.approx(75., rel=1e-13)


@pytest.mark.parametrize('kwargs', [
    dict(order=0), dict(order=True), dict(order=1.5),
    dict(batch_size=0), dict(batch_size=np.bool_(True)),
    dict(max_cell_diameter=0.), dict(max_cell_diameter=-1.),
    dict(max_cell_diameter=np.nan), dict(max_cell_diameter=np.inf),
])
def test_invalid_options(kwargs):
    mesh = lmesh(1)
    with pytest.raises(ValueError):
        energy_error(mesh, np.zeros(mesh.n_vertices), **kwargs)


@pytest.mark.parametrize('nodal', ['short', 'nan', 'inf'])
def test_invalid_nodal_values(nodal):
    mesh = lmesh(1)
    values = np.zeros(mesh.n_vertices)
    if nodal == 'short':
        values = values[:-1]
    else:
        values[3] = np.nan if nodal == 'nan' else np.inf
    with pytest.raises(ValueError, match='nodal_values'):
        energy_error(mesh, values)


@pytest.mark.parametrize('bad', [lambda p: np.zeros(p.shape[:-1]),
                                 lambda p: np.full(p.shape, np.nan)])
def test_invalid_exact_gradient(bad):
    mesh = lmesh(1)
    with pytest.raises(ValueError, match='exact_gradient'):
        energy_error(mesh, np.zeros(mesh.n_vertices), bad)


def test_example_uniform_smoke(capsys):
    assert example_main(['--levels', '3', '--singular-only']) == 0
    out = capsys.readouterr().out
    rows = [line.split() for line in out.splitlines() if line[:2].strip().isdigit()]
    assert [int(r[0]) for r in rows] == [0, 1, 2, 3]
    assert [int(r[3]) for r in rows] == [0, 6, 36, 168]
    errors = [float(r[4]) for r in rows]
    assert all(a > b for a, b in zip(errors, errors[1:]))
    # Pre-asymptotic but already near the r^(1/3) prediction N^(-1/6).
    assert float(rows[-1][5]) == pytest.approx(1/6, abs=0.02)
