"""Quadrature for affine P1 triangular finite elements (NumPy only).

triangle_rule(degree) returns (xy, w) on {(x,y): x,y >= 0, x+y <= 1}.
Weights sum to 1/2, NOT 1. On a physical triangle multiply by abs(det J).
edge_rule(npoints) returns (t, w) on [0,1]; weights sum to 1. On a
physical edge multiply by its length. No physical scaling is done here.

Dunavant degrees 1--8 have 1,3,4,6,7,12,13,16 points respectively.

"""
from functools import lru_cache
from itertools import permutations
from numbers import Integral

import numpy as np
from numpy.polynomial.legendre import leggauss

__all__ = ["triangle_rule", "edge_rule"]

# (barycentric representative, weight per point), weights normalized to 1.
# Expand distinct permutations, then scale by reference area 1/2.
_RULES = {
    1: [((1/3, 1/3, 1/3), 1.0)],
    2: [((2/3, 1/6, 1/6), 1/3)],
    3: [((1/3, 1/3, 1/3), -0.5625),
        ((0.6, 0.2, 0.2), 0.520833333333333333)],
    4: [((0.108103018168070, 0.445948490915965, 0.445948490915965), 0.223381589678011),
        ((0.816847572980459, 0.091576213509771, 0.091576213509771), 0.109951743655322)],
    5: [((1/3, 1/3, 1/3), 0.225),
        ((0.059715871789770, 0.470142064105115, 0.470142064105115), 0.132394152788506),
        ((0.797426985353087, 0.101286507323456, 0.101286507323456), 0.125939180544827)],
    6: [((0.501426509658179, 0.249286745170910, 0.249286745170910), 0.116786275726379),
        ((0.873821971016996, 0.063089014491502, 0.063089014491502), 0.050844906370207),
        ((0.053145049844817, 0.310352451033784, 0.636502499121399), 0.082851075618374)],
    7: [((1/3, 1/3, 1/3), -0.149570044467682),
        ((0.479308067841920, 0.260345966079040, 0.260345966079040), 0.175615257433208),
        ((0.869739794195568, 0.065130102902216, 0.065130102902216), 0.053347235608838),
        ((0.048690315425316, 0.312865496004874, 0.638444188569810), 0.077113760890257)],
    8: [((1/3, 1/3, 1/3), 0.144315607677787),
        ((0.081414823414554, 0.459292588292723, 0.459292588292723), 0.095091634267285),
        ((0.658861384496480, 0.170569307751760, 0.170569307751760), 0.103217370534718),
        ((0.898905543365938, 0.050547228317031, 0.050547228317031), 0.032458497623198),
        ((0.008394777409958, 0.263112829634638, 0.728492392955404), 0.027230314174435)],
}


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer (not a boolean)")
    return int(value)


@lru_cache(maxsize=8)
def _triangle_cached(degree):
    points, weights = [], []
    for barycentric, weight in _RULES[degree]:
        for lam in sorted(set(permutations(barycentric))):
            points.append(lam[1:])  # lambda_0 = 1-x-y
            weights.append(weight / 2)
    return np.array(points, dtype=np.float64), np.array(weights, dtype=np.float64)


def triangle_rule(degree=2):
    """Return float64 points (n,2), weights (n,) exact to total degree.

    degree must be an integer in [1,8]; unsupported degrees raise ValueError
    rather than silently underintegrating. Returned arrays are independent,
    writable copies, so callers cannot corrupt cached tables.
    """
    degree = _integer(degree, "degree")
    if degree not in _RULES:
        raise ValueError("triangle degree must be between 1 and 8 inclusive")
    points, weights = _triangle_cached(degree)
    return points.copy(), weights.copy()


@lru_cache(maxsize=32)
def _edge_cached(npoints):
    points, weights = leggauss(npoints)
    return (points + 1) / 2, weights / 2


def edge_rule(npoints=2):
    """Return float64 nodes (n,), weights (n,) for integration on [0,1].

    npoints is a positive integer, NOT an exactness degree. The n-point rule
    integrates polynomials through degree 2*n-1 to floating-point accuracy.
    To integrate degree p >= 0, use npoints=(p+2)//2. Returns fresh arrays.
    """
    npoints = _integer(npoints, "npoints")
    if npoints < 1:
        raise ValueError("edge npoints must be positive")
    points, weights = _edge_cached(npoints)
    return points.copy(), weights.copy()
