"""Residual estimator used by the peak-blind calibration baseline.

Moved from solution/test_afem.py when the reference solver switched to
exact-error marking; the estimator now lives in authoring/provenance/calibrate.py.
"""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('calibrate', ROOT/'authoring/provenance/calibrate.py')
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)          # puts environment/data on sys.path
from fem import problem  # noqa: E402
from fem.basis import physical_gradients  # noqa: E402
from fem.quadrature import edge_rule  # noqa: E402


def lmesh(level=0):
    mesh = problem.initial_mesh()
    for _ in range(level):
        mesh = mesh.uniform_refine()
    return mesh


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
    eta = calibrate.residual_estimator(mesh, u, np.zeros(mesh.n_triangles), g_N)
    np.testing.assert_allclose(eta, reference_indicator(mesh, u, g_N), rtol=1e-13, atol=1e-15)


def test_estimator_vanishes_for_consistent_linear_solution():
    # u = 2 + 3x - 4y: no jumps; on Gamma_N the outward flux is +3, so a
    # wrongly oriented normal would leave a residual of 6.
    mesh = lmesh(2)
    u = 2 + mesh.vertices @ np.array([3., -4.])
    eta = calibrate.residual_estimator(mesh, u, np.zeros(mesh.n_triangles), lambda p: np.full(p.shape[:-1], 3.))
    assert np.abs(eta).max() < 1e-24
