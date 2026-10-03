"""Tests for the reference AFEM (run from the task root: pytest solution)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import afem  # noqa: E402  (afem puts the fem package on sys.path)
from fem import problem  # noqa: E402
from fem.basis import physical_gradients  # noqa: E402
from fem.io import load_mesh  # noqa: E402
from fem.quadrature import edge_rule  # noqa: E402


def lmesh(level=0):
    mesh = problem.initial_mesh()
    for _ in range(level):
        mesh = mesh.uniform_refine()
    return mesh


def hanging_nodes(mesh):
    """Independent geometric check: count vertices strictly inside an edge."""
    v, count = mesh.vertices, 0
    for a, b in mesh.edges:
        d = v[b] - v[a]
        o = v - v[a]
        cross = o[:, 0]*d[1] - o[:, 1]*d[0]
        proj = o @ d / (d @ d)
        count += np.sum((np.abs(cross) <= 1e-14*(d @ d)) & (proj > 1e-12) & (proj < 1-1e-12))
    return int(count)


def min_angle(mesh):
    p = mesh.vertices[mesh.triangles]
    h = np.sort(np.linalg.norm(p[:, [1, 2, 0]] - p, axis=2), axis=1)
    return np.degrees(np.arcsin(np.clip(2*mesh.areas/(h[:, 2]*h[:, 1]), 0, 1))).min()


# ---------------------------------------------------------------- Dörfler

def test_dorfler_minimal_prefix():
    eta = np.array([1., 4., 2., 3.])          # total 10
    assert sorted(afem.dorfler(eta, .4)) == [1]        # 4 >= 4
    assert sorted(afem.dorfler(eta, .41)) == [1, 3]    # 4 < 4.1 <= 7
    assert sorted(afem.dorfler(eta, 1.)) == [0, 1, 2, 3]
    assert len(afem.dorfler(np.zeros(5), .5)) == 0


@pytest.mark.parametrize('eta, theta', [([1., -1.], .5), ([np.nan], .5),
                                        ([1.], 0.), ([1.], 1.5), ([[1.]], .5)])
def test_dorfler_invalid(eta, theta):
    with pytest.raises(ValueError):
        afem.dorfler(np.array(eta), theta)


# ---------------------------------------------------------------- NVB

def test_nvb_closure_on_initial_mesh():
    # Marking one triangle bisects its shared diagonal, so its partner splits too.
    mesh = afem.newest_vertex_bisection(lmesh(), [0])
    assert (mesh.n_triangles, mesh.n_vertices) == (8, 9)
    np.testing.assert_array_equal(mesh.vertices[-1], [-.5, -.5])
    assert hanging_nodes(mesh) == 0


def test_nvb_empty_marking_is_identity():
    mesh = lmesh()
    same = afem.newest_vertex_bisection(mesh, np.empty(0, dtype=int))
    np.testing.assert_array_equal(same.triangles, mesh.triangles)


@pytest.mark.parametrize('marked', [[-1], [6], [0.5], [[0]]])
def test_nvb_invalid_marking(marked):
    with pytest.raises(ValueError):
        afem.newest_vertex_bisection(lmesh(), np.array(marked))


def test_nvb_random_rounds_conforming_and_shape_regular():
    rng = np.random.default_rng(0)
    mesh = lmesh()
    for _ in range(12):
        marked = rng.choice(mesh.n_triangles, size=max(1, mesh.n_triangles//5), replace=False)
        child = afem.newest_vertex_bisection(mesh, marked)
        assert child.area == pytest.approx(3., abs=1e-12)
        assert hanging_nodes(child) == 0
        new = set(map(tuple, child.triangles))
        assert all(tuple(mesh.triangles[k]) not in new for k in marked)
        # NVB from right isosceles triangles only produces similar triangles.
        assert min_angle(child) == pytest.approx(45.)
        problem.boundary_masks(child)  # corners kept, boundary on domain sides
        mesh = child


# ---------------------------------------------------------------- estimator

def reference_indicator(mesh, u, g_N):
    """Loop-based reference for the jump and Neumann terms (no element term)."""
    eta = np.zeros(mesh.n_triangles)
    grads = [u[t] @ physical_gradients(mesh.vertices[t]) for t in mesh.triangles]
    _, neumann = problem.boundary_masks(mesh)
    t_edge, w_edge = edge_rule(8)
    for e, (a, b) in enumerate(mesh.edges):
        pa, pb = mesh.vertices[a], mesh.vertices[b]
        length = np.linalg.norm(pb - pa)
        normal = np.array([pb[1]-pa[1], pa[0]-pb[0]])/length
        k0, k1 = mesh.edge_triangles[e]
        if k1 >= 0:
            jump = (grads[k0] - grads[k1]) @ normal
            eta[k0] += length**2*jump**2/2
            eta[k1] += length**2*jump**2/2
        elif neumann[e]:
            outward = np.array([1., 0.])          # Gamma_N lies on x = 0
            points = pa + t_edge[:, None]*(pb - pa)
            diff = g_N(points) - grads[k0] @ outward
            eta[k0] += length**2*np.sum(w_edge*diff**2)
    return eta


def test_estimator_matches_loop_reference():
    mesh = lmesh(2)
    u = np.random.default_rng(1).normal(size=mesh.n_vertices)
    g_N = lambda p: 1 + p[..., 1]**2
    eta = afem.residual_estimator(mesh, u, np.zeros(mesh.n_triangles), g_N)
    np.testing.assert_allclose(eta, reference_indicator(mesh, u, g_N), rtol=1e-13, atol=1e-15)


def test_estimator_vanishes_for_consistent_linear_solution():
    # u = 2 + 3x - 4y: no jumps; on Gamma_N the outward flux is +3, so a
    # wrongly oriented normal would leave a residual of 6.
    mesh = lmesh(2)
    u = 2 + mesh.vertices @ np.array([3., -4.])
    eta = afem.residual_estimator(mesh, u, np.zeros(mesh.n_triangles), lambda p: np.full(p.shape[:-1], 3.))
    assert np.abs(eta).max() < 1e-24


def test_element_term_and_peak_quadrature():
    mesh = lmesh()
    # Constant f = 2: h_T^2 * int f^2 = 2 * 4 * 0.5 = 4 on every unit right triangle.
    _, element = afem.volume_terms(mesh, lambda p: np.full(p.shape[:-1], 2.))
    np.testing.assert_allclose(element, 4., rtol=1e-12)  # 15-digit weights, many subcells
    # The narrow peak is resolved on the coarse mesh: halving the integration
    # cell size leaves the load unchanged, and sum(load) = int f = 0.
    a, ea = afem.volume_terms(mesh, problem.f, .04)
    b, eb = afem.volume_terms(mesh, problem.f, .02)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)
    np.testing.assert_allclose(ea, eb, rtol=1e-8, atol=1e-12*eb.max())
    assert abs(a.sum()) < 1e-10 and ea.max() > 1


# ---------------------------------------------------------------- stopping

def quiet_run(**kwargs):
    return afem.run(lmesh(), **kwargs)


def test_stop_on_tolerance(capsys):
    _, _, history = quiet_run(tol=.95, budget=10**6, N_max=50)
    assert history[-1]['stop_reason'] == 'tol'
    assert history[-1]['reported_error'] <= .95
    assert all(h['reported_error'] > .95 for h in history[:-1])


def test_stop_on_budget(capsys):
    mesh, _, history = quiet_run(tol=1e-6, budget=20, N_max=200)
    assert history[-1]['stop_reason'] == 'budget'
    assert all(h['dof'] <= 20 for h in history)
    assert np.count_nonzero(~problem.boundary_masks(mesh)[0]) == history[-1]['dof']


def test_stop_on_n_max(capsys):
    _, _, history = quiet_run(tol=1e-6, budget=10**6, N_max=3)
    assert history[-1]['stop_reason'] == 'N_max'
    assert [h['iter'] for h in history] == [0, 1, 2]


def test_initial_mesh_over_budget_rejected():
    with pytest.raises(ValueError, match='budget'):
        afem.run(lmesh(2), budget=1)


# ---------------------------------------------------------------- end to end

def test_main_writes_valid_submission(tmp_path, capsys):
    out = tmp_path / 'out' / 'submission.json'
    assert afem.main(['--tol', '.5', '--budget', '2000', '--N-max', '40', '--output', str(out)]) == 0
    data = json.loads(out.read_text(encoding='utf-8'))
    mesh = load_mesh(out)
    assert data['dof'] == np.count_nonzero(~problem.boundary_masks(mesh)[0]) <= 2000
    assert data['reported_error'] <= .5
    assert len(data['history']) <= 40
    assert data['history'][-1]['stop_reason'] == 'tol'
    assert all({'iter', 'dof', 'estimator'} <= set(h) for h in data['history'])
    assert hanging_nodes(mesh) == 0


# ---------------------------------------------------------------- task limits

def test_task_limits_consistent():
    """selected_config.json, specs.txt, tests/config.json, solve.sh and the
    problem.py constants must agree (authoring checkout only)."""
    import re
    task = Path(__file__).resolve().parents[1]
    paths = dict(selected=task/'authoring/provenance/selected_config.json',
                 sealed=task/'tests/config.json', specs=task/'environment/data/specs.txt',
                 solve=task/'solution/solve.sh')
    if not all(p.exists() for p in paths.values()):
        pytest.skip('not an authoring checkout')
    selected = json.loads(paths['selected'].read_text(encoding='utf-8'))
    sealed = json.loads(paths['sealed'].read_text(encoding='utf-8'))
    specs = paths['specs'].read_text(encoding='utf-8')
    solve_sh = paths['solve'].read_text(encoding='utf-8')

    def spec_value(label):
        return float(re.search(rf'^{label}:\s*(\S+)', specs, re.M).group(1))

    def sh_value(name):
        return float(re.search(rf'^{name}=(\S+)$', solve_sh, re.M).group(1))

    limits = (selected['tol'], selected['dof_budget'], selected['N_max'])
    assert (sealed['error_tol'], sealed['dof_budget'], sealed['max_iter']) == limits
    assert (spec_value('Error tolerance'), spec_value('DOF budget'), spec_value('N_max iter')) == limits
    assert (sh_value('TOL'), sh_value('BUDGET'), sh_value('N_MAX')) == limits

    peak = (selected['A'], selected['alpha'], selected['x0'])
    assert (sealed['problem']['A'], sealed['problem']['alpha'], sealed['problem']['x0']) == peak
    assert (problem.A, problem.ALPHA, list(problem.X0)) == peak
    match = re.search(r'A = (\S+),\s*alpha = (\S+),\s*x0 = \((\S+), (\S+)\)', specs)
    assert (float(match[1]), float(match[2]), [float(match[3]), float(match[4])]) == peak
    assert 'TBD' not in specs and 'provisional' not in specs.lower()
