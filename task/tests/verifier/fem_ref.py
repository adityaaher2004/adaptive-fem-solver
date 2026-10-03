"""Independent affine P1 Poisson assembly and sparse direct solve.

Uses only ValidatedMesh and verifier-owned configuration. Volume integration
uses tensor Gauss-Legendre quadrature through a Duffy map, with integration-only
red subdivision. No supplied fem code or Dunavant tables are imported.
"""
from dataclasses import dataclass
from numbers import Integral, Real
import warnings
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from .mesh_checks import ValidatedMesh, load_config

__all__ = ['Problem', 'ReferenceSystem', 'assemble', 'solve', 'element_gradients']


@dataclass(frozen=True)
class Problem:
    A: float
    alpha: float
    x0: tuple

    def __post_init__(self):
        for name in ('A', 'alpha'):
            value=getattr(self,name)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value):
                raise ValueError(f'{name} must be finite and real')
            object.__setattr__(self,name,float(value))
        if self.alpha<=0:
            raise ValueError('alpha must be positive')
        raw=np.asarray(self.x0)
        if raw.shape!=(2,) or raw.dtype.kind not in 'iuf':
            raise ValueError('x0 must contain two real coordinates')
        x,y=map(float,raw)
        if not np.isfinite([x,y]).all() or not (-1<x<1 and -1<y<1 and (x<0 or y>0)):
            raise ValueError('x0 must be inside the L-shaped domain')
        object.__setattr__(self,'x0',(x,y))

    @classmethod
    def from_config(cls, config=None):
        block=(load_config() if config is None else config)['problem']
        return cls(A=block['A'],alpha=block['alpha'],x0=block['x0'])

    def peak(self, points):
        offset=np.asarray(points)-self.x0
        return self.A*np.exp(-self.alpha*np.sum(offset*offset,axis=-1))

    def f(self, points):
        offset=np.asarray(points)-self.x0
        q=self.alpha*np.sum(offset*offset,axis=-1)
        return 4*self.alpha*(1-q)*self.peak(points)

    def g_D(self, points):
        p=np.asarray(points)
        r=np.hypot(p[...,0],p[...,1])
        angle=np.remainder(np.arctan2(p[...,1],p[...,0]),2*np.pi)
        return np.cbrt(r)*np.sin(angle/3)+self.peak(p)

    def g_N(self, points):
        # Gamma_N has outward normal (+1,0); singular flux is exactly zero.
        return 2*self.alpha*self.x0[0]*self.peak(points)


def element_gradients(mesh):
    """Physical P1 basis gradients, shape (triangles,3,2)."""
    p=mesh.vertices[mesh.triangles]
    x,y=p[:,:,0],p[:,:,1]
    return np.stack((y[:,[1,2,0]]-y[:,[2,0,1]],
                     x[:,[2,0,1]]-x[:,[1,2,0]]),axis=2)/(2*mesh.areas[:,None,None])


def _positive_integer(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<1:
        raise ValueError(f'{name} must be a positive integer')


def _values(fn,p):
    raw=np.asarray(fn(p))
    if raw.dtype.kind not in 'iuf':
        raise ValueError('PDE callback must return real values')
    result=np.broadcast_to(raw,p.shape[:-1])
    if not np.isfinite(result).all():
        raise ValueError('PDE callback returned nonfinite values')
    return result


@dataclass(frozen=True)
class ReferenceSystem:
    stiffness: object
    load: np.ndarray
    matrix: object
    rhs: np.ndarray
    free_vertices: np.ndarray
    lifting: np.ndarray
    basis_gradients: np.ndarray

    def expand(self, free_values):
        values=np.asarray(free_values,float)
        if values.shape!=self.rhs.shape or not np.isfinite(values).all():
            raise ValueError('invalid free solution')
        full=self.lifting.copy()
        full[self.free_vertices]=values
        return full


def assemble(mesh, config=None, *, volume_order=8, edge_order=8,
             max_cell_diameter=None, batch_size=128,
             f=None, g_D=None, g_N=None):
    """Return full and Dirichlet-eliminated independent P1 systems.

    Default data come from config['problem'] (tests/config.json when omitted).
    Optional callbacks support independent manufactured patch tests. Callbacks
    take (...,2) points and return broadcastable scalar values. Set g_N to a
    zero callback to omit flux. None always selects the configured data.

    Default integration diameter is 1/sqrt(alpha), resolving the Gaussian.
    Subdivision changes neither FEM basis functions nor DOFs. Convergence must
    be checked by increasing order/decreasing diameter; no certification is
    implied. Boundary masks are consumed exactly as validated, including D
    junction vertices and Neumann edges with D endpoints.
    """
    if not isinstance(mesh,ValidatedMesh):
        raise TypeError('mesh must come from verifier.mesh_checks')
    p=Problem.from_config(config)
    f=p.f if f is None else f
    g_D=p.g_D if g_D is None else g_D
    g_N=p.g_N if g_N is None else g_N
    for name,value in [('volume_order',volume_order),('edge_order',edge_order),('batch_size',batch_size)]:
        _positive_integer(value,name)
    limit=1/np.sqrt(p.alpha) if max_cell_diameter is None else max_cell_diameter
    if isinstance(limit,(bool,np.bool_)) or not np.isfinite(limit) or limit<=0:
        raise ValueError('max_cell_diameter must be positive and finite')
    gradients=element_gradients(mesh)
    local=mesh.areas[:,None,None]*np.einsum('tik,tjk->tij',gradients,gradients)
    ids=mesh.triangles
    rows=np.broadcast_to(ids[:,:,None],local.shape).ravel()
    cols=np.broadcast_to(ids[:,None,:],local.shape).ravel()
    n=len(mesh.vertices)
    K=coo_matrix((local.ravel(),(rows,cols)),shape=(n,n)).tocsr()
    K.eliminate_zeros()
    b=np.zeros(n)
    q,w=np.polynomial.legendre.leggauss(volume_order)
    q,w=(q+1)/2,w/2
    r,s=np.meshgrid(q,q,indexing='ij')
    # Reference coordinates (r,(1-r)*s), Jacobian (1-r).
    phi=np.stack(( (1-r)*(1-s),r,(1-r)*s),axis=-1).reshape(-1,3)
    weights=(w[:,None]*w[None,:]*(1-r)).ravel()
    def integrate(cells,bary,owners):
        diameter=np.linalg.norm(cells[:,[1,2,0]]-cells[:,[0,1,2]],axis=2).max(axis=1)
        split=diameter>limit
        if split.any():
            def children(v):
                a,b,c=v[:,0],v[:,1],v[:,2]
                ab,bc,ca=(a+b)/2,(b+c)/2,(c+a)/2
                return np.stack([np.stack(t,axis=1) for t in
                    [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]],axis=1).reshape((-1,3,v.shape[-1]))
            child=children(cells[split]); bary_child=children(bary[split])
            parent=np.repeat(owners[split],4)
            for start in range(0,len(child),batch_size):
                integrate(child[start:start+batch_size],bary_child[start:start+batch_size],parent[start:start+batch_size])
        cells,bary,owners=cells[~split],bary[~split],owners[~split]
        if not len(cells): return
        points=np.einsum('qi,tij->tqj',phi,cells)
        parent_phi=np.einsum('qi,tij->tqj',phi,bary)
        a,c=cells[:,1]-cells[:,0],cells[:,2]-cells[:,0]
        determinant=np.abs(a[:,0]*c[:,1]-a[:,1]*c[:,0])
        rhs=determinant[:,None]*np.einsum('tq,q,tqi->ti',_values(f,points),weights,parent_phi)
        np.add.at(b,ids[owners].ravel(),rhs.ravel())
    for start in range(0,len(ids),batch_size):
        owners=np.arange(start,min(start+batch_size,len(ids)))
        integrate(mesh.vertices[ids[owners]],np.broadcast_to(np.eye(3),(len(owners),3,3)),owners)
    q,w=np.polynomial.legendre.leggauss(edge_order)
    q,w=(q+1)/2,w/2
    edge_phi=np.stack((1-q,q),axis=1)
    edges=mesh.edges[mesh.neumann_edge_mask]
    for start in range(0,len(edges),batch_size):
        indices=edges[start:start+batch_size]
        endpoints=mesh.vertices[indices]
        points=np.einsum('qi,tij->tqj',edge_phi,endpoints)
        length=np.linalg.norm(endpoints[:,1]-endpoints[:,0],axis=1)
        rhs=length[:,None]*np.einsum('tq,q,qi->ti',_values(g_N,points),w,edge_phi)
        np.add.at(b,indices.ravel(),rhs.ravel())
    lifting=np.zeros(n)
    D=mesh.dirichlet_vertex_mask
    if D.any(): lifting[D]=_values(g_D,mesh.vertices[D])
    free=np.flatnonzero(~D)
    if not np.isfinite(K.data).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite assembled system')
    return ReferenceSystem(K,b,K[free][:,free].tocsr(),(b-K@lifting)[free],free,lifting,gradients)


def solve(mesh, config=None, **options):
    """Assemble and solve; return a finite full nodal vector in vertex order."""
    system=assemble(mesh,config,**options)
    if not len(system.free_vertices):
        return system.expand(np.empty(0))
    with warnings.catch_warnings():
        warnings.simplefilter('error',MatrixRankWarning)
        try:
            free=spsolve(system.matrix,system.rhs)
        except MatrixRankWarning as exc:
            raise np.linalg.LinAlgError('singular verifier system') from exc
    if not np.isfinite(free).all():
        raise np.linalg.LinAlgError('nonfinite verifier solution')
    return system.expand(free)
