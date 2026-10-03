"""Manufactured mixed-boundary Poisson problem on a lower-right-cutout L.

Omega = (-1,1)^2 minus ([0,1) x (-1,0]). Angles run counterclockwise
from the positive x axis to the negative y axis, in [0,3*pi/2].
All point inputs have shape (...,2); scalar fields return shape (...,).
The origin has u_s=0 but NO finite gradient: grad_u raises there.

The weak load is integral(f*v) + integral_Gamma_N(g_N*v). Dirichlet
values are interpolated at boundary nodes, including both N/D junctions.
Relative energy error uses the FULL grad_u, including its cross term.
Exact fields are exposed intentionally; hide them in the verifier if the
exam must assess residual-estimator design rather than mesh quality alone.

With alpha=1000, load/error integration must resolve the peak
(scale alpha**-0.5); fixed low-order quadrature on coarse cells can miss it.
Singular energy integration requires corner-aware quadrature; this module
supplies fields, not an error integrator or a certified mesh verifier.
"""
from dataclasses import dataclass
import numpy as np

A = 0.5
ALPHA = 1000.0
X0 = (-0.43, 0.61)
DOMAIN_CORNERS = ((-1., -1.), (0., -1.), (0., 0.),
                  (1., 0.), (1., 1.), (-1., 1.))
__all__ = ['A', 'ALPHA', 'X0', 'DOMAIN_CORNERS', 'PoissonProblem',
           'DEFAULT_PROBLEM', 'u', 'grad_u', 'f', 'g_D', 'g_N',
           'u_s', 'grad_u_s', 'in_domain', 'is_neumann', 'is_dirichlet',
           'boundary_masks', 'initial_mesh']


def _points(points):
    raw = np.asarray(points)
    if raw.dtype.kind not in 'iuf':
        raise TypeError('points must be real numbers')
    p = np.asarray(raw, dtype=float)
    if p.ndim < 1 or p.shape[-1] != 2 or not np.isfinite(p).all():
        raise ValueError('points must be finite with shape (...,2)')
    return p


def in_domain(points, *, closure=True):
    """Exact geometric predicate; no tolerance that could merge refined nodes."""
    p = _points(points)
    x, y = p[..., 0], p[..., 1]
    if closure:
        return (np.abs(p) <= 1).all(axis=-1) & ((x <= 0) | (y >= 0))
    return (np.abs(p) < 1).all(axis=-1) & ((x < 0) | (y > 0))


def _domain_points(points):
    p = _points(points)
    if not np.all(in_domain(p)):
        raise ValueError('point outside the closed L-shaped domain')
    return p


def is_neumann(points):
    """Open Neumann segment; its two endpoints are Dirichlet nodes."""
    p = _points(points)
    return (p[..., 0] == 0) & (p[..., 1] > -1) & (p[..., 1] < 0)


def is_dirichlet(points):
    p = _points(points)
    return in_domain(p) & ~in_domain(p, closure=False) & ~is_neumann(p)


def _polar(points):
    p = _domain_points(points)
    r = np.hypot(p[..., 0], p[..., 1])
    theta = np.mod(np.arctan2(p[..., 1], p[..., 0]), 2*np.pi)
    return p, r, theta


def u_s(points):
    """Harmonic singular component r**(1/3)*sin(theta/3), continuous at 0."""
    _, r, theta = _polar(points)
    return np.cbrt(r)*np.sin(theta/3)


def grad_u_s(points):
    """Cartesian singular gradient; reject any batch containing the origin."""
    p, r, theta = _polar(points)
    if np.any(r == 0):
        raise ValueError('gradient is undefined at the re-entrant origin')
    direction = np.stack((-np.sin(2*theta/3), np.cos(2*theta/3)), axis=-1)
    # Enforce the analytic zero on the Neumann ray without sin(pi) roundoff.
    direction[..., 0] = np.where((p[..., 0] == 0) & (
        p[..., 1] < 0), 0., direction[..., 0])
    return direction * (r**(-2/3)/3)[..., None]


@dataclass(frozen=True)
class PoissonProblem:
    """Configurable Gaussian plus the fixed mixed-corner singularity."""
    A: float = A
    alpha: float = ALPHA
    x0: tuple = X0

    def __post_init__(self):
        for name in ('A', 'alpha'):
            value = np.asarray(getattr(self, name))
            if value.shape != () or value.dtype.kind not in 'iuf':
                raise TypeError(f'{name} must be a real scalar')
            if not np.isfinite(value) or (name == 'alpha' and value <= 0):
                raise ValueError(
                    'A must be finite and alpha finite and positive')
            object.__setattr__(self, name, float(value))
        center = _points(self.x0)
        if center.shape != (2,) or not in_domain(center, closure=False):
            raise ValueError('x0 must be a single interior point')
        object.__setattr__(self, 'x0', tuple(center))

    def u_p(self, points):
        z = _domain_points(points) - self.x0
        return self.A*np.exp(-self.alpha*np.sum(z*z, axis=-1))

    def u(self, points):
        return u_s(points) + self.u_p(points)

    def grad_u(self, points):
        p = _domain_points(points)
        return grad_u_s(p) - 2*self.alpha*(p-self.x0)*self.u_p(p)[..., None]

    def f(self, points):
        p = _domain_points(points)
        q = self.alpha*np.sum((p-self.x0)**2, axis=-1)
        return 4*self.alpha*(1-q)*self.u_p(p)

    def g_D(self, points):
        """Dirichlet trace; reject points not on Gamma_D."""
        p = _domain_points(points)
        if not np.all(is_dirichlet(p)):
            raise ValueError('g_D requires Dirichlet boundary points')
        return self.u(p)

    def g_N(self, points):
        """Outward flux on the open Gamma_N (normal +x).

        Evaluates only the Gaussian contribution, avoiding singular gradient
        cancellation and preserving the extremely small default flux.
        """
        p = _domain_points(points)
        if not np.all(is_neumann(p)):
            raise ValueError('g_N requires open Neumann boundary points')
        return 2*self.alpha*self.x0[0]*self.u_p(p)


def boundary_masks(mesh):
    """Return (D vertex mask, N edge mask), indexed over ALL vertices/edges.

    Assumes a conforming domain mesh. Checks corners and boundary segments,
    but does not certify interior coverage/overlap. Exact coordinates are
    intentional: uniform midpoint refinement preserves the axis-aligned sides.
    Neumann edges are classified by endpoints on the CLOSED segment, so an
    entire edge with two Dirichlet endpoints still receives a Neumann load.
    """
    p = _domain_points(mesh.vertices)
    for corner in DOMAIN_CORNERS:
        if not np.any(np.all(p == corner, axis=1)):
            raise ValueError('mesh is missing a required domain corner')
    endpoints = p[mesh.edges]
    corners = np.array(DOMAIN_CORNERS)
    valid = np.zeros(mesh.n_edges, dtype=bool)
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        valid |= ((endpoints >= lo) & (endpoints <= hi)).all(axis=(1, 2))
    if not np.all(valid[mesh.boundary_edge_mask]):
        raise ValueError('mesh boundary edge does not lie on a domain side')
    on_n = ((endpoints[..., 0] == 0) & (endpoints[..., 1] >= -1)
            & (endpoints[..., 1] <= 0)).all(axis=1)
    return (mesh.boundary_vertex_mask & is_dirichlet(p),
            mesh.boundary_edge_mask & on_n)


def initial_mesh():
    """Six CCW triangles, three unit squares, lower-right quadrant removed.

    Newest-vertex-bisection convention: local vertex 0 of each triangle is
    its newest vertex, and the refinement edge is the opposite edge (local
    vertices 1-2). Here vertex 0 is the right-angle corner, so every
    refinement edge is its square's diagonal, shared by both halves of the
    square (a compatible initial labelling). This is the same mesh as
    initial_mesh.json, which authoring/provenance/make_initial_mesh.py
    generates from this function.
    """
    from .mesh import Mesh
    return Mesh([[-1, -1], [0, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]],
                [[1, 3, 0], [2, 0, 3], [3, 6, 2], [5, 2, 6], [4, 7, 3], [6, 3, 7]])


DEFAULT_PROBLEM = PoissonProblem()
u = DEFAULT_PROBLEM.u
grad_u = DEFAULT_PROBLEM.grad_u
f = DEFAULT_PROBLEM.f
g_D = DEFAULT_PROBLEM.g_D
g_N = DEFAULT_PROBLEM.g_N
