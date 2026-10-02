"""Batched P1 Poisson assembly on affine CCW triangles.

COO duplicate entries are summed on conversion to CSR. The weak load uses
+ integral_Gamma_N(g_N*v). Callbacks accept (...,2) points and return (...,)
values (a scalar constant is also accepted). No exact solution is used.

Fixed quadrature is not an accuracy certificate for the narrow Gaussian.
load_refinements uniformly subdivides REFERENCE integration cells only;
it changes neither the mesh nor its DOFs. Check quadrature convergence.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from .basis import reference_gradients, shape_functions
from .quadrature import triangle_rule, edge_rule
from . import problem

__all__ = ['element_geometry', 'boundary_masks', 'assemble_stiffness',
           'assemble_load', 'eliminate_dirichlet', 'assemble_system',
           'DirichletSystem']


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer')
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return int(value)


def _values(callback, points):
    raw = np.asarray(callback(points))
    if raw.dtype.kind not in 'iuf':
        raise TypeError('field values must be real')
    values = np.broadcast_to(raw, points.shape[:-1]).astype(float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError('field values must be finite')
    return values


def _mask(value, size, name):
    a = np.asarray(value)
    if a.dtype != bool or a.shape != (size,):
        raise ValueError(f'{name} must be a boolean mask of shape ({size},)')
    return a


def element_geometry(mesh, elements=slice(None)):
    """Return (origins, J, detJ, gradients) for a slice/index array of cells.

    Shapes: (B,2), (B,2,2), (B,), (B,3,2). J stores edge columns;
    gradients store physical row vectors. Use a batch, even for one cell.
    """
    v = mesh.vertices[mesh.triangles[elements]]
    if v.ndim != 3:
        raise ValueError('elements must select a batch, not a scalar cell')
    J = np.swapaxes(v[:,1:] - v[:,:1], 1, 2)
    det = J[:,0,0]*J[:,1,1] - J[:,0,1]*J[:,1,0]
    rhs = np.broadcast_to(reference_gradients().T, (len(v),2,3))
    gradients = np.linalg.solve(J.swapaxes(1,2), rhs).swapaxes(1,2)
    if not np.isfinite(gradients).all():
        raise ValueError('physical gradients are not representable')
    return v[:,0], J, det, gradients


def boundary_masks(mesh):
    """Return D-vertex and N-edge masks for the specified L-shaped domain.

    N is x=0, -1<y<0; both junction vertices are D. Edges are classified
    by their relative interiors. Exact coordinates match midpoint refinement.
    Other domains can supply explicit masks to assemble_system/assemble_load.
    This checks boundary segments, not full domain coverage or conformity.
    """
    p = mesh.vertices
    if not np.all((np.abs(p) <= 1).all(axis=1) & ((p[:,0] <= 0) | (p[:,1] >= 0))):
        raise ValueError('vertices outside the closed L-shaped domain')
    corners = np.array([[-1,-1],[0,-1],[0,0],[1,0],[1,1],[-1,1]])
    for c in corners:
        if not np.any(np.all(p == c, axis=1)):
            raise ValueError('missing required domain corner')
    ep = p[mesh.edges]
    valid = np.zeros(mesh.n_edges, bool)
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        valid |= ((ep >= np.minimum(a,b)) & (ep <= np.maximum(a,b))).all(axis=(1,2))
    if not valid[mesh.boundary_edge_mask].all():
        raise ValueError('boundary edge does not lie on a domain side')
    n_edges = mesh.boundary_edge_mask & ((ep[...,0] == 0) &
               (ep[...,1] >= -1) & (ep[...,1] <= 0)).all(axis=1)
    n_vertices = (p[:,0] == 0) & (p[:,1] > -1) & (p[:,1] < 0)
    return mesh.boundary_vertex_mask & ~n_vertices, n_edges


def assemble_stiffness(mesh, *, batch_size=4096):
    """Assemble integral grad(phi_i).grad(phi_j); return CSR (N,N)."""
    batch_size = _integer(batch_size, 'batch_size')
    entries = np.empty((mesh.n_triangles,3,3))
    for start in range(0, mesh.n_triangles, batch_size):
        s = slice(start, start+batch_size)
        _, _, det, grad = element_geometry(mesh, s)
        entries[s] = (det/2)[:,None,None]*np.einsum('tik,tjk->tij', grad, grad)
    t = mesh.triangles
    rows = np.broadcast_to(t[:,:,None], entries.shape).ravel()
    cols = np.broadcast_to(t[:,None,:], entries.shape).ravel()
    K = coo_matrix((entries.ravel(), (rows, cols)),
                   shape=(mesh.n_vertices,mesh.n_vertices)).tocsr()
    K.eliminate_zeros()
    return K


def _volume_rule(degree, refinements):
    points, weights = triangle_rule(degree)
    cells = np.array([[[0.,0.],[1.,0.],[0.,1.]]])
    for _ in range(refinements):
        a,b,c = cells[:,0],cells[:,1],cells[:,2]
        ab,bc,ca = (a+b)/2,(b+c)/2,(c+a)/2
        cells = np.stack((np.stack((a,ab,ca),1), np.stack((ab,b,bc),1),
                          np.stack((ca,bc,c),1), np.stack((ab,bc,ca),1)),1).reshape(-1,3,2)
    mapped = cells[:,None,0] + np.einsum('qi,tij->tqj', points, cells[:,1:]-cells[:,:1])
    return mapped.reshape(-1,2), np.tile(weights/len(cells), len(cells))


def assemble_load(mesh, f=problem.f, g_N=problem.g_N, *, neumann_edge_mask=None,
                  triangle_degree=8, edge_npoints=4, load_refinements=0,
                  batch_size=4096):
    """Assemble volume and Neumann loads, before Dirichlet elimination.

    Set g_N=None to omit boundary loads. Explicit N masks index ALL edges
    and may select only boundary edges. Integration batches contain at most
    batch_size quadrature points per callback (unless one rule exceeds it).
    """
    batch_size = _integer(batch_size, 'batch_size')
    level = _integer(load_refinements, 'load_refinements', 0)
    q,w = _volume_rule(triangle_degree, level)
    phi = shape_functions(q)
    result = np.zeros(mesh.n_vertices)
    cells_per_batch = max(1, batch_size//len(w))
    for start in range(0,mesh.n_triangles,cells_per_batch):
        s = slice(start,start+cells_per_batch)
        v = mesh.vertices[mesh.triangles[s]]
        points = v[:,None,0] + np.einsum('qi,tij->tqj', q, v[:,1:]-v[:,:1])
        local = 2*mesh.areas[s,None]*np.einsum('tq,q,qi->ti', _values(f,points),w,phi)
        np.add.at(result, mesh.triangles[s].ravel(),local.ravel())
    if g_N is not None:
        if neumann_edge_mask is None:
            _, neumann_edge_mask = boundary_masks(mesh)
        mask = _mask(neumann_edge_mask,mesh.n_edges,'neumann_edge_mask')
        if np.any(mask & ~mesh.boundary_edge_mask):
            raise ValueError('Neumann mask includes interior edges')
        t,w = edge_rule(edge_npoints)
        phi = np.column_stack((1-t,t))
        edges = mesh.edges[mask]
        step = max(1,batch_size//len(w))
        for start in range(0,len(edges),step):
            ids = edges[start:start+step]
            v = mesh.vertices[ids]
            d = v[:,1]-v[:,0]
            points = v[:,None,0]+t[None,:,None]*d[:,None,:]
            local = np.linalg.norm(d,axis=1)[:,None]*np.einsum('eq,q,qi->ei',_values(g_N,points),w,phi)
            np.add.at(result,ids.ravel(),local.ravel())
    return result


@dataclass(frozen=True)
class DirichletSystem:
    """Reduced CSR matrix/rhs plus full-sized lifting and free vertex IDs.

    Solve matrix @ u_free = rhs; expand(u_free) restores nodal ordering.
    len(free_vertices) is the number of unconstrained algebraic DOFs.
    """
    matrix: csr_matrix
    rhs: np.ndarray
    free_vertices: np.ndarray
    lifting: np.ndarray

    def expand(self, free_values):
        values = np.asarray(free_values, dtype=float)
        if values.shape != self.rhs.shape or not np.isfinite(values).all():
            raise ValueError('free_values must be finite with shape matching rhs')
        u = self.lifting.copy()
        u[self.free_vertices] = values
        return u


def eliminate_dirichlet(mesh, matrix, rhs, g_D=problem.g_D, *, dirichlet_vertex_mask=None):
    """Return K_FF, b_F-K_FD*g_D and reconstruction data; inputs unchanged.

    Only selected D vertices are constrained. N-only boundary nodes remain
    free. An empty D mask is allowed, but a pure-Neumann matrix is singular.
    """
    n = mesh.n_vertices
    K = csr_matrix(matrix, dtype=float, copy=True)
    b = np.asarray(rhs, dtype=float)
    if K.shape != (n,n) or b.shape != (n,):
        raise ValueError('matrix/rhs shapes must match mesh vertices')
    if not np.isfinite(K.data).all() or not np.isfinite(b).all():
        raise ValueError('matrix/rhs must be finite')
    if dirichlet_vertex_mask is None:
        dirichlet_vertex_mask, _ = boundary_masks(mesh)
    mask = _mask(dirichlet_vertex_mask,n,'dirichlet_vertex_mask')
    if np.any(mask & ~mesh.boundary_vertex_mask):
        raise ValueError('Dirichlet mask includes interior vertices')
    lifting = np.zeros(n)
    if mask.any():
        lifting[mask] = _values(g_D, mesh.vertices[mask])
    free = np.flatnonzero(~mask)
    return DirichletSystem(K[free][:,free].tocsr(), (b-K@lifting)[free], free, lifting)


def assemble_system(mesh, f=problem.f, g_D=problem.g_D, g_N=problem.g_N, *,
                    dirichlet_vertex_mask=None, neumann_edge_mask=None, **quadrature_options):
    """Assemble and eliminate D data; return a DirichletSystem.

    quadrature_options are assemble_load keywords, including batch_size.
    Default boundary masks describe the exam's lower-right-cutout L-shape.
    """
    K = assemble_stiffness(mesh, batch_size=quadrature_options.get('batch_size',4096))
    b = assemble_load(mesh,f,g_N,neumann_edge_mask=neumann_edge_mask,**quadrature_options)
    return eliminate_dirichlet(mesh,K,b,g_D,dirichlet_vertex_mask=dirichlet_vertex_mask)
