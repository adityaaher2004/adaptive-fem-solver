"""Demo/calibration energy integration, NOT the independent exam verifier.

Exact gradients are required. On triangles touching the origin, the Duffy map
x=t**3*((1-s)*a+s*b) gives dx=3*t**5*abs(det(a,b)) dt ds.
For grad(u_s)~r**(-2/3), its squared-energy integrand is O(t),
removing the radial singularity. Gauss points never evaluate the origin.
Other triangles use an ordinary Duffy map. Optional integration-only red
subdivision resolves smooth localized features without changing FEM DOFs.
Always check convergence in order and max_cell_diameter for new problems.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from .assembly import element_geometry
from .problem import grad_u

__all__ = ['EnergyError', 'energy_error']


@dataclass(frozen=True)
class EnergyError:
    absolute: float
    reference_norm: float
    relative: float


def energy_error(mesh, nodal_values, exact_gradient=grad_u, *, order=12,
                 max_cell_diameter=0.125, batch_size=256):
    """Integrate ||grad(u-u_h)|| and ||grad(u)|| over the supplied mesh.

    The denominator includes all cross terms in exact_gradient. Boundary
    interpolation error is included. Only an origin corner singularity is
    specially treated. Defaults resolve the exam Gaussian (alpha=1000);
    they are not a rigorous integration tolerance or an error certificate.
    """
    for name, value in [('order', order), ('batch_size', batch_size)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not np.isfinite(max_cell_diameter) or max_cell_diameter <= 0:
        raise ValueError('max_cell_diameter must be positive and finite')
    u = np.asarray(nodal_values, dtype=float)
    if u.shape != (mesh.n_vertices,) or not np.isfinite(u).all():
        raise ValueError('nodal_values must be finite with one value per vertex')
    q, w = np.polynomial.legendre.leggauss(order)
    q, w = (q+1)/2, w/2
    t, s = np.meshgrid(q, q, indexing='ij')
    t, s = t.ravel(), s.ravel()
    weights = (w[:, None]*w[None, :]).ravel()
    totals = np.zeros(2)

    def integrate(cells, gradients):
        diameter = np.max(np.linalg.norm(cells[:, [1,2,0]]-cells[:, [0,1,2]], axis=2), axis=1)
        split = diameter > max_cell_diameter
        if split.any():
            a,b,c = cells[split,0], cells[split,1], cells[split,2]
            ab,bc,ca = (a+b)/2, (b+c)/2, (c+a)/2
            children = np.stack([np.stack(v, axis=1) for v in
                                 [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]],axis=1).reshape(-1,3,2)
            child_grad = np.repeat(gradients[split],4,axis=0)
            for start in range(0,len(children),batch_size):
                integrate(children[start:start+batch_size],child_grad[start:start+batch_size])
        cells, gradients = cells[~split], gradients[~split]
        if not len(cells):
            return
        at_origin = np.all(cells == 0, axis=2)
        corner = at_origin.any(axis=1)
        for special in (False, True):
            selected = corner == special
            v = cells[selected].copy()
            if not len(v):
                continue
            if special:
                indices = np.argmax(at_origin[selected], axis=1)
                v = v[np.arange(len(v))[:,None], (indices[:,None]+np.arange(3))%3]
            a,b = v[:,1]-v[:,0], v[:,2]-v[:,0]
            determinant = np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
            radius = t**3 if special else t
            jacobian = 3*t**5 if special else t
            points = v[:,None,0] + radius[None,:,None]*((1-s)[None,:,None]*a[:,None]+s[None,:,None]*b[:,None])
            exact = np.asarray(exact_gradient(points), dtype=float)
            if exact.shape != points.shape or not np.isfinite(exact).all():
                raise ValueError('exact_gradient must return finite vectors matching points')
            difference = exact-gradients[selected,None,:]
            weight = determinant[:,None]*(weights*jacobian)[None,:]
            totals[0] += np.sum(weight*np.sum(difference*difference,axis=2))
            totals[1] += np.sum(weight*np.sum(exact*exact,axis=2))

    for start in range(0,mesh.n_triangles,batch_size):
        ids = mesh.triangles[start:start+batch_size]
        _,_,_,basis_grad = element_geometry(mesh,slice(start,start+batch_size))
        gradient = np.einsum('ti,tij->tj',u[ids],basis_grad)
        integrate(mesh.vertices[ids],gradient)
    if not np.isfinite(totals).all() or totals[1] <= 0:
        raise ValueError('reference energy must be positive and finite')
    absolute, reference = np.sqrt(totals)
    return EnergyError(float(absolute),float(reference),float(absolute/reference))
