"""P1 Lagrange basis and affine geometry for one 2D triangle.

Both orientations are supported. Use abs(det(J)) for area integration;
never replace J by an orientation-normalized matrix when transforming
points or gradients. Geometry is for one triangle, not a mesh batch.
"""
import numpy as np

__all__ = [
    "shape_functions", "reference_gradients", "jacobian",
    "jacobian_determinant", "triangle_area", "reference_to_physical",
    "physical_to_reference", "physical_gradients",
]


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers")
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numbers")
    return array


def _points(value):
    points = _real_array(value, "points")
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError("points must have shape (..., 2)")
    return points


def _geometry(vertices):
    vertices = _real_array(vertices, "vertices")
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2)")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        J = (vertices[1:] - vertices[0]).T
        det = J[0, 0]*J[1, 1] - J[0, 1]*J[1, 0]
    if not np.isfinite(J).all() or not np.isfinite(det) or det == 0:
        raise ValueError(
            "triangle is degenerate or its Jacobian is not representable")
    return vertices, J, float(det)


def shape_functions(points):
    """Evaluate (1-xi-eta, xi, eta); input (...,2), output (...,3).

    A single point (2,) returns (3,). Empty point batches are supported.
    Points outside the reference triangle are allowed for extrapolation;
    this function is not a point-in-triangle predicate.
    """
    points = _points(points)
    return np.concatenate((1 - points.sum(axis=-1, keepdims=True), points), axis=-1)


def reference_gradients():
    """Return a fresh (3,2) array of constant reference basis gradients."""
    return np.array([[-1., -1.], [1., 0.], [0., 1.]])


def jacobian(vertices):
    """Return J (2,2), with columns V[1]-V[0] and V[2]-V[0].

    All geometry helpers reject nonfinite coordinates, invalid shapes, and
    zero/nonfinite computed determinants. No mesh-quality cutoff is imposed:
    callers must assess conditioning of very thin triangles separately.
    """
    return _geometry(vertices)[1]


def jacobian_determinant(vertices):
    """Return signed det(J), positive for counterclockwise vertex order."""
    return _geometry(vertices)[2]


def triangle_area(vertices):
    """Return unsigned area abs(det(J))/2."""
    return abs(jacobian_determinant(vertices)) / 2


def reference_to_physical(points, vertices):
    """Map reference points (...,2) to physical points of identical shape."""
    points = _points(points)
    vertices, J, _ = _geometry(vertices)
    return vertices[0] + points @ J.T


def physical_to_reference(points, vertices):
    """Invert the affine map for points (...,2), preserving shape.

    Uses a linear solve rather than explicitly forming the inverse.
    Outside points are allowed; no containment check or clipping is done.
    """
    points = _points(points)
    vertices, J, _ = _geometry(vertices)
    rhs = (points - vertices[0]).reshape(-1, 2).T
    return np.linalg.solve(J, rhs).T.reshape(points.shape)


def physical_gradients(vertices):
    """Return constant physical gradients (3,2), one row per local basis.

    Solves J.T @ grad_x.T = grad_reference.T, avoiding explicit inversion.
    """
    _, J, _ = _geometry(vertices)
    return np.linalg.solve(J.T, reference_gradients().T).T
