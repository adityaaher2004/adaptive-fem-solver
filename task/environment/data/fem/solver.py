"""Sparse direct solve and full nodal reconstruction for P1 systems.

Usage::

    system = assemble_system(mesh)
    u_h = solve(system)
"""
import warnings

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .assembly import DirichletSystem

__all__ = ['solve']


def solve(system: DirichletSystem) -> np.ndarray:
    """Return a fresh full nodal vector in the mesh's vertex ordering.

    Solve the Dirichlet-eliminated system with SciPy's sparse direct solver,
    then restore prescribed values using ``system.expand``. An empty reduced
    system returns the prescribed values without calling the sparse solver.
    Inputs are not modified. Reported singularity or nonfinite solutions raise
    ``numpy.linalg.LinAlgError``; no gauge fixing for pure Neumann problems is
    performed. This wrapper does not estimate conditioning or FEM error.
    """
    if not isinstance(system, DirichletSystem):
        raise TypeError(
            'system must be a DirichletSystem from assemble_system')
    n = system.rhs.size
    if system.rhs.shape != (n,) or system.matrix.shape != (n, n):
        raise ValueError('reduced matrix and rhs shapes must agree')
    if not np.isfinite(system.matrix.data).all() or not np.isfinite(system.rhs).all():
        raise ValueError('reduced matrix and rhs must be finite')
    if n == 0:
        return system.expand(np.empty(0))
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        try:
            free_values = spsolve(system.matrix, system.rhs)
        except MatrixRankWarning as exc:
            raise np.linalg.LinAlgError(
                'reduced FEM system is singular') from exc
    if not np.isfinite(free_values).all():
        raise np.linalg.LinAlgError(
            'sparse solve returned nonfinite nodal values')
    return system.expand(free_values)
