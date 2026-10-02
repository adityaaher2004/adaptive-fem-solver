"""P1 identities and analytic element matrices, independent of implementation."""
from itertools import permutations

import numpy as np
import pytest
from fem.basis import (
    shape_functions, reference_gradients, jacobian, jacobian_determinant,
    triangle_area, reference_to_physical, physical_to_reference,
    physical_gradients,
)
from fem.quadrature import triangle_rule

V = np.array([[2., -1.], [5., 0.], [1., 3.]])


def test_nodal_values_and_partition():
    nodes = np.array([[0.,0.], [1.,0.], [0.,1.]])
    np.testing.assert_array_equal(shape_functions(nodes), np.eye(3))
    points = np.array([[[.2,.3], [0.,.4]], [[1.,1.], [-.5,.1]]])
    phi = shape_functions(points)
    assert phi.shape == (2,2,3)
    np.testing.assert_allclose(phi.sum(axis=-1), 1.)
    np.testing.assert_allclose(phi @ nodes, points)
    np.testing.assert_allclose(shape_functions([.2,.3]), [.5,.2,.3])


def test_reference_gradients_and_mutation():
    grad = reference_gradients()
    np.testing.assert_array_equal(grad, [[-1.,-1.],[1.,0.],[0.,1.]])
    np.testing.assert_array_equal(grad.sum(axis=0), [0.,0.])
    grad[:] = 0
    assert reference_gradients()[0,0] == -1


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_geometry_and_affine_reproduction(order):
    v = V[list(order)]
    J = jacobian(v)
    assert abs(jacobian_determinant(v)) == pytest.approx(13.)
    assert triangle_area(v) == pytest.approx(6.5)
    # Independent orientation via the shoelace formula.
    shoelace = sum(v[i,0]*v[(i+1)%3,1] - v[(i+1)%3,0]*v[i,1] for i in range(3))
    assert np.sign(jacobian_determinant(v)) == np.sign(shoelace)
    np.testing.assert_allclose(J, (v[1:] - v[0]).T, rtol=0, atol=0)
    points = np.array([[[0.,0.],[1.,0.]], [[0.,1.],[.2,.3]], [[-.1,1.5],[.7,.1]]])
    mapped = reference_to_physical(points, v)
    np.testing.assert_allclose(mapped, shape_functions(points) @ v, atol=1e-14)
    np.testing.assert_allclose(physical_to_reference(mapped,v), points, atol=1e-14)
    np.testing.assert_allclose(physical_to_reference(v[0],v), [0.,0.], atol=1e-14)
    grad = physical_gradients(v)
    np.testing.assert_allclose(grad.sum(axis=0), 0., atol=1e-15)
    # Exact interpolation of u(x,y)=7+2x-3y must recover its gradient.
    nodal_values = 7 + v @ np.array([2.,-3.])
    np.testing.assert_allclose(nodal_values @ grad, [2.,-3.], atol=1e-14)
    np.testing.assert_allclose(shape_functions(points) @ nodal_values,
                               7 + mapped @ np.array([2.,-3.]), atol=1e-14)
    # Difference quotient independently checks the physical derivative.
    # Everything is affine, so a large step has no truncation error and
    # keeps roundoff (~eps/h) far below the tolerance.
    x = np.array([2.5,.5])
    for axis in range(2):
        step = np.eye(2)[axis]*1e-2
        diff = (shape_functions(physical_to_reference(x+step,v)) -
                shape_functions(physical_to_reference(x-step,v))) / 2e-2
        np.testing.assert_allclose(diff, grad[:,axis], atol=1e-12, rtol=0)


def test_known_jacobian_and_gradients():
    np.testing.assert_array_equal(jacobian(V), [[3.,-1.],[1.,4.]])
    assert jacobian_determinant(V) == 13.
    assert jacobian_determinant(V[[0,2,1]]) == -13.
    np.testing.assert_allclose(physical_gradients(V),
                               np.array([[-3.,-4.],[4.,1.],[-1.,3.]])/13)


@pytest.mark.parametrize("scale", [1e-9, 1., 1e9])
def test_scale_invariance(scale):
    v = scale*V
    assert triangle_area(v) == pytest.approx(6.5*scale**2)
    np.testing.assert_allclose(physical_gradients(v)*scale, physical_gradients(V))


def test_quadrature_mass_and_stiffness():
    xy,w = triangle_rule(2)
    phi = shape_functions(xy)
    mass = phi.T @ ((13*w)[:,None]*phi)
    np.testing.assert_allclose(mass, (6.5/12)*(np.ones((3,3))+np.eye(3)), rtol=1e-14)
    # Unit right triangle: analytic Laplacian element stiffness.
    unit = np.array([[0.,0.],[1.,0.],[0.,1.]])
    grad = physical_gradients(unit)
    stiffness = triangle_area(unit)*(grad @ grad.T)
    np.testing.assert_allclose(stiffness, [[1.,-.5,-.5],[-.5,.5,0.],[-.5,0.,.5]])


def test_empty_batches_and_no_input_mutation():
    points = np.empty((2,0,2))
    assert shape_functions(points).shape == (2,0,3)
    assert reference_to_physical(points,V).shape == points.shape
    assert physical_to_reference(points,V).shape == points.shape
    v = V.copy()
    J = jacobian(v)
    J[:] = 0
    np.testing.assert_array_equal(v,V)


GEOMETRY = [jacobian, jacobian_determinant, triangle_area, physical_gradients,
            lambda v: reference_to_physical([.2,.3],v),
            lambda v: physical_to_reference([.2,.3],v)]


@pytest.mark.parametrize("function", GEOMETRY)
@pytest.mark.parametrize("bad", [np.zeros((3,2)), [[0,0],[1,1],[2,2]],
    [[0,0],[1,0],[1,0]], np.zeros((2,3)), np.zeros((1,3,2)),
    [[0,0],[1,0],[0,np.nan]], [[0,0],[1,0],[0,np.inf]]])
def test_invalid_geometry(function, bad):
    with pytest.raises(ValueError):
        function(bad)


@pytest.mark.parametrize("bad", [1., [1.], [[1.,2.,3.]], [np.nan,0.], [0.,np.inf]])
@pytest.mark.parametrize("function", [shape_functions,
    lambda p: reference_to_physical(p,V), lambda p: physical_to_reference(p,V)])
def test_invalid_points(function, bad):
    with pytest.raises(ValueError):
        function(bad)


@pytest.mark.parametrize("bad", [[1j,0], ["1","0"], [True,False]])
def test_nonreal_points(bad):
    with pytest.raises(TypeError):
        shape_functions(bad)
