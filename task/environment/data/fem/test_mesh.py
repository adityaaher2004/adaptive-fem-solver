from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from fem.mesh import Mesh


def l_shape():
    # Problem domain (-1,1)^2 \ ([0,1) x (-1,0]): lower-right quadrant removed,
    # re-entrant corner at the origin. Three unit squares, two CCW triangles each.
    # [-1,1]^2 minus the open upper-right quadrant: three unit squares.
    v = [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1]]
    t = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6]]
    return Mesh(v,t)


def test_single_triangle_topology():
    m = Mesh([[0,0],[1,0],[0,1]], [[0,1,2]])
    np.testing.assert_array_equal(m.edges, [[0,1],[0,2],[1,2]])
    np.testing.assert_array_equal(m.triangle_edges, [[0,2,1]])
    np.testing.assert_array_equal(m.edge_triangles, [[0,-1],[0,-1],[0,-1]])
    assert m.area == .5
    assert m.boundary_vertex_mask.all() and m.boundary_edge_mask.all()


def check_topology(m):
    for k, triangle in enumerate(m.triangles):
        for local, (i,j) in enumerate([(0,1),(1,2),(2,0)]):
            e = m.triangle_edges[k,local]
            np.testing.assert_array_equal(m.edges[e], sorted([triangle[i],triangle[j]]))
            assert k in m.edge_triangles[e]
    for e, pair in enumerate(m.edge_triangles):
        incident = np.flatnonzero(np.any(m.triangle_edges == e, axis=1))
        np.testing.assert_array_equal(pair[pair >= 0], incident)
    np.testing.assert_array_equal(m.boundary_edges, m.edges[m.boundary_edge_mask])
    np.testing.assert_array_equal(m.boundary_vertices, np.flatnonzero(m.boundary_vertex_mask))
    assert m.n_vertices - m.n_edges + m.n_triangles == 1
    assert (m.areas > 0).all()


def test_l_shape_and_repeated_red_refinement():
    m = l_shape()
    assert (m.n_vertices,m.n_edges,m.n_triangles) == (8,13,6)
    assert len(m.boundary_edges) == 8
    for _ in range(4):
        check_topology(m)
        assert m.area == pytest.approx(3.)
        child = m.uniform_refine()
        assert child.n_vertices == m.n_vertices + m.n_edges
        assert child.n_triangles == 4*m.n_triangles
        assert child.n_edges == 2*m.n_edges + 3*m.n_triangles
        assert len(child.boundary_edges) == 2*len(m.boundary_edges)
        np.testing.assert_array_equal(child.vertices[:m.n_vertices], m.vertices)
        np.testing.assert_allclose(child.areas.reshape(-1,4), np.repeat(m.areas[:,None]/4,4,axis=1))
        # Each old edge has exactly one new midpoint, with correct boundary flag.
        np.testing.assert_array_equal(child.vertices[m.n_vertices:], m.vertices[m.edges].mean(axis=1))
        np.testing.assert_array_equal(child.boundary_vertex_mask[m.n_vertices:], m.boundary_edge_mask)
        # Every old boundary edge splits into exactly its two boundary halves.
        expected = set()
        for e in np.flatnonzero(m.boundary_edge_mask):
            a,b = m.edges[e]
            mid = m.n_vertices + e
            expected.update([tuple(sorted((a,mid))),tuple(sorted((mid,b)))])
        assert set(map(tuple,child.boundary_edges)) == expected
        m = child
    assert m.n_triangles == 6*4**4
    assert m.area == pytest.approx(3.)


def test_geometric_conformity_after_refinement():
    m = l_shape().uniform_refine().uniform_refine()
    # No other vertex lies strictly inside any edge (independent geometric check).
    for a,b in m.edges:
        delta = m.vertices[b]-m.vertices[a]
        offsets = m.vertices-m.vertices[a]
        cross = offsets[:,0]*delta[1] - offsets[:,1]*delta[0]
        projection = offsets @ delta / (delta @ delta)
        assert not np.any((cross == 0) & (projection > 0) & (projection < 1))


def test_snapshot_and_immutability():
    v = np.array([[0.,0.],[1.,0.],[0.,1.]])
    t = np.array([[0,1,2]])
    m = Mesh(v,t)
    v[:] = 99
    t[:] = 0
    assert m.area == .5 and m.triangles[0,1] == 1
    for name in m.__dataclass_fields__:
        array = getattr(m,name)
        with pytest.raises(ValueError):
            array.flat[0] = 0
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        m.vertices = v


@pytest.mark.parametrize('v,t', [
    ([[0,0],[0,1],[1,0]], [[0,1,2]]),  # CW
    ([[0,0],[1,0],[2,0]], [[0,1,2]]),  # collinear
    ([[0,0],[1,0],[0,1]], [[0,0,2]]),
    ([[0,0],[1,0],[0,1]], [[0,1,3]]),
    ([[0,0],[1,0],[0,1]], [[0,1,-1]]),
    ([[0,0],[1,0],[0,1]], [[0,1,2],[1,2,0]]),
    ([[0,0],[1,0],[0,1],[0,1]], [[0,1,2],[0,1,3]]),
    ([[0,0],[1,0],[0,1],[2,2]], [[0,1,2]]),
    ([[0,0],[1,0],[0,np.nan]], [[0,1,2]]),
    ([[0,0],[1,0],[0,np.inf]], [[0,1,2]]),
    ([[0,0],[1,0],[0,1]], np.empty((0,3),dtype=int)),
    ([[0,0],[1,0],[0,1]], [[0,1]]),
    ([[0,0,0],[1,0,0],[0,1,0]], [[0,1,2]]),
    # Same directed shared edge, both triangles CCW but on same side.
    ([[0,0],[1,0],[0,1],[0,2]], [[0,1,2],[0,1,3]]),
    # Three incident triangles.
    ([[0,0],[1,0],[0,1],[0,-1],[0,2]], [[0,1,2],[1,0,3],[0,1,4]]),
])
def test_invalid_mesh(v,t):
    with pytest.raises(ValueError):
        Mesh(v,t)


@pytest.mark.parametrize('t', [[[0.,1.,2.]], [[False,True,True]], [['0','1','2']]])
def test_noninteger_indices(t):
    with pytest.raises(TypeError):
        Mesh([[0,0],[1,0],[0,1]],t)


def test_unsigned_index_overflow_rejected():
    with pytest.raises(ValueError):
        Mesh([[0,0],[1,0],[0,1]],np.array([[0,1,2**64-1]],dtype=np.uint64))
