# Analytic moments are the oracle; no quadrature tables are used as truth
from itertools import permutations
from math import factorial

import numpy as np
import pytest
from fem.quadrature import edge_rule, triangle_rule


@pytest.mark.parametrize("degree", range(1, 9))
def test_triangle_all_barycentric_moments(degree):
    xy, w = triangle_rule(degree)
    lam = np.column_stack((1 - xy.sum(axis=1), xy))
    # Integral lambda0^a lambda1^b lambda2^c = a! b! c!/(a+b+c+2)!
    for a in range(degree + 1):
        for b in range(degree + 1 - a):
            for c in range(degree + 1 - a - b):
                exact = factorial(a)*factorial(b)*factorial(c)/factorial(a+b+c+2)
                actual = w @ (lam[:, 0]**a * lam[:, 1]**b * lam[:, 2]**c)
                assert actual == pytest.approx(exact, rel=8e-14, abs=2e-16), (degree,a,b,c)


@pytest.mark.parametrize("degree,count", enumerate([1,3,4,6,7,12,13,16], start=1))
def test_triangle_structure(degree, count):
    xy, w = triangle_rule(degree)
    assert xy.shape == (count, 2) and w.shape == (count,)
    assert xy.dtype == w.dtype == np.float64
    assert np.isfinite(xy).all() and np.isfinite(w).all()
    assert (xy > 0).all() and (xy.sum(axis=1) < 1).all()
    assert w.sum() == pytest.approx(0.5, abs=3e-15)
    assert np.count_nonzero(w < 0) == (1 if degree in (3,7) else 0)


@pytest.mark.parametrize("n", range(1, 17))
def test_edge_moments_and_structure(n):
    t, w = edge_rule(n)
    assert t.shape == w.shape == (n,)
    assert t.dtype == w.dtype == np.float64
    assert (t > 0).all() and (t < 1).all() and (w > 0).all()
    assert np.all(np.diff(t) > 0)
    for p in range(2*n):
        assert w @ t**p == pytest.approx(1/(p+1), rel=8e-14, abs=2e-16)


@pytest.mark.parametrize("degree", range(1,9))
def test_physical_triangle_affine_mapping(degree):
    vertices = np.array([[2.,-1.], [5.,0.], [1.,3.]])
    # All vertex orderings, including reversed orientation, must give same moments.
    for order in permutations(range(3)):
        v = vertices[list(order)]
        J = (v[1:] - v[0]).T
        xy, w = triangle_rule(degree)
        physical = v[0] + xy @ J.T
        physical_w = abs(np.linalg.det(J)) * w
        area = 6.5
        assert physical_w.sum() == pytest.approx(area, abs=5e-14)
        np.testing.assert_allclose(physical_w @ physical, area*vertices.mean(axis=0), rtol=1e-14)
        if degree >= 2:
            # Integral x^2 = area/6*(sum x_i^2 + sum_{i<j} x_i*x_j).
            x = vertices[:, 0]
            exact = area/6 * (x@x + x[0]*x[1]+x[0]*x[2]+x[1]*x[2])
            assert physical_w @ physical[:,0]**2 == pytest.approx(exact, rel=2e-14)


@pytest.mark.parametrize("n", range(1,9))
def test_physical_edge(n):
    a, b = np.array([1.,2.]), np.array([4.,6.])
    for start, end in [(a,b), (b,a)]:
        t, w = edge_rule(n)
        xy = start + t[:,None]*(end-start)
        assert 5*w.sum() == pytest.approx(5.)
        # Projection from a is exactly t (or 1-t on reversal).
        s = (xy-a) @ (b-a) / 25
        for p in range(2*n):
            assert (5*w) @ s**p == pytest.approx(5/(p+1), rel=8e-14)


@pytest.mark.parametrize("rule", [triangle_rule, edge_rule])
@pytest.mark.parametrize("bad", [True, np.bool_(False), 2.0, "2", None, [2]])
def test_noninteger_rejected(rule, bad):
    with pytest.raises(TypeError):
        rule(bad)


@pytest.mark.parametrize("bad", [-1, 0, 9, 100])
def test_unsupported_triangle_degree(bad):
    with pytest.raises(ValueError):
        triangle_rule(bad)


@pytest.mark.parametrize("bad", [-1, 0])
def test_invalid_edge_count(bad):
    with pytest.raises(ValueError):
        edge_rule(bad)


@pytest.mark.parametrize("rule", [triangle_rule, edge_rule])
def test_cache_isolation_and_numpy_integer(rule):
    p, w = rule(np.int64(2))
    expected_p, expected_w = p.copy(), w.copy()
    p[:] = -100
    w[:] = -100
    fresh_p, fresh_w = rule(2)
    np.testing.assert_array_equal(fresh_p, expected_p)
    np.testing.assert_array_equal(fresh_w, expected_w)
