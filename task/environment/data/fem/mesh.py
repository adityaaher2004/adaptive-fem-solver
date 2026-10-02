"""Triangular mesh topology and uniform red refinement (NumPy only).
"""
from dataclasses import dataclass

import numpy as np

__all__ = ["Mesh"]


def _readonly(array):
    # Immutable bytes backing prevents re-enabling writes and stale topology.
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, init=False, eq=False)
class Mesh:
    """Immutable snapshot of a nonempty triangle mesh.

    vertices: float64 (N,2); triangles: int64 (T,3), CCW, zero-based.
    edges: int64 (E,2), sorted endpoints and lexicographic row order.
    triangle_edges: (T,3), global edge IDs in local cyclic-edge order.
    edge_triangles: (E,2), incident triangle IDs in ascending order; -1
        in the second slot denotes boundary (never use -1 as an index).
    boundary_edge_mask: bool (E,); boundary_vertex_mask: bool (N,).
    boundary_edges: (B,2) endpoint IDs; boundary_vertices: (Nb,) vertex IDs.
    areas: (T,) positive triangle areas.

    All vertices must be used. Exact duplicate coordinates and duplicate
    triangles (regardless of local ordering) are rejected. No aspect-ratio
    cutoff is imposed. Construct a new Mesh to change geometry/connectivity.
    """
    vertices: np.ndarray
    triangles: np.ndarray
    edges: np.ndarray
    triangle_edges: np.ndarray
    edge_triangles: np.ndarray
    boundary_edge_mask: np.ndarray
    boundary_vertex_mask: np.ndarray
    boundary_edges: np.ndarray
    boundary_vertices: np.ndarray
    areas: np.ndarray

    def __init__(self, vertices, triangles):
        raw_v = np.asarray(vertices)
        raw_t = np.asarray(triangles)
        if raw_v.dtype.kind not in "iuf":
            raise TypeError("vertices must contain real numbers")
        if raw_t.dtype.kind not in "iu":
            raise TypeError("triangles must contain integer indices")
        v = np.array(raw_v, dtype=np.float64, copy=True)
        if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
            raise ValueError("vertices must have shape (N,2), N >= 3")
        if raw_t.ndim != 2 or raw_t.shape[1] != 3 or len(raw_t) == 0:
            raise ValueError("triangles must have nonempty shape (T,3)")
        if not np.isfinite(v).all():
            raise ValueError("vertices must be finite")
        if np.any(raw_t < 0) or np.any(raw_t >= len(v)):
            raise ValueError("triangle index outside vertex range")
        t = np.array(raw_t, dtype=np.int64, copy=True)
        sorted_t = np.sort(t, axis=1)
        if np.any(np.diff(sorted_t, axis=1) == 0):
            raise ValueError("triangle has repeated vertex indices")
        if len(np.unique(sorted_t, axis=0)) != len(t):
            raise ValueError("duplicate triangles")
        if len(np.unique(v, axis=0)) != len(v):
            raise ValueError("duplicate vertex coordinates")
        if len(np.unique(t)) != len(v):
            raise ValueError("unused vertices")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            a = v[t[:,1]] - v[t[:,0]]
            b = v[t[:,2]] - v[t[:,0]]
            areas = (a[:,0]*b[:,1] - a[:,1]*b[:,0]) / 2
        if not np.isfinite(areas).all() or np.any(areas <= 0):
            raise ValueError("triangles must have finite positive area and CCW orientation")
        if not np.isfinite(areas.sum()):
            raise ValueError("total area is not representable")

        directed = t[:, [[0,1], [1,2], [2,0]]].reshape(-1,2)
        edges, inverse, counts = np.unique(
            np.sort(directed, axis=1), axis=0, return_inverse=True,
            return_counts=True,
        )
        if np.any(counts > 2):
            raise ValueError("nonmanifold edge: more than two incident triangles")
        # Consistently oriented neighbours traverse a shared edge in opposite
        # directions; the same directed edge twice means overlap or a fold.
        if len(np.unique(directed, axis=0)) != len(directed):
            raise ValueError("inconsistent orientation across a shared edge")
        inverse = inverse.reshape(-1).astype(np.int64)
        edges = edges.astype(np.int64)

        # Stable grouping preserves ascending incident triangle IDs.
        order = np.argsort(inverse, kind="stable")
        owner = order // 3  # directed edge k belongs to triangle k // 3
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        edge_triangles = np.full((len(edges), 2), -1, dtype=np.int64)
        edge_triangles[:, 0] = owner[starts]
        interior = counts == 2
        edge_triangles[interior, 1] = owner[starts[interior] + 1]

        boundary_edge_mask = counts == 1
        boundary_edges = edges[boundary_edge_mask]
        boundary_vertex_mask = np.zeros(len(v), dtype=bool)
        boundary_vertex_mask[boundary_edges.reshape(-1)] = True

        fields = {
            "vertices": v,
            "triangles": t,
            "edges": edges,
            "triangle_edges": inverse.reshape(-1, 3),
            "edge_triangles": edge_triangles,
            "boundary_edge_mask": boundary_edge_mask,
            "boundary_vertex_mask": boundary_vertex_mask,
            "boundary_edges": boundary_edges,
            "boundary_vertices": np.flatnonzero(boundary_vertex_mask).astype(np.int64),
            "areas": areas,
        }
        for name, value in fields.items():
            object.__setattr__(self, name, _readonly(value))

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def n_triangles(self):
        return len(self.triangles)

    @property
    def area(self):
        """Total mesh area (sum of triangle areas)."""
        return float(self.areas.sum())

    def uniform_refine(self):
        """Return a new Mesh with every triangle split into four (red refinement).

        Numbering contract:
          - old vertices keep their IDs; the midpoint of old edge e becomes
            vertex n_vertices + e, computed as (a + b) / 2;
          - the children of old triangle k are triangles 4k .. 4k+3, ordered
            (corner at local vertex 0, 1, 2, then the central triangle).
        All children inherit the parent's CCW orientation.
        """
        a, b = self.edges[:, 0], self.edges[:, 1]
        midpoints = (self.vertices[a] + self.vertices[b]) / 2
        vertices = np.concatenate((self.vertices, midpoints))

        v0, v1, v2 = self.triangles.T
        # Local edges in cyclic order: (0,1), (1,2), (2,0).
        m01, m12, m20 = (self.n_vertices + self.triangle_edges).T
        children = np.stack((
            np.column_stack((v0, m01, m20)),
            np.column_stack((m01, v1, m12)),
            np.column_stack((m20, m12, v2)),
            np.column_stack((m01, m12, m20)),
        ), axis=1).reshape(-1, 3)
        return Mesh(vertices, children)