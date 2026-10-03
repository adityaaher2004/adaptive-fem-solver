"""Independent corner-aware energy error for the sealed verifier.

Direct integration uses a radial cubic Duffy map on origin triangles, with
geometrically graded radial panels, and high-order tensor Gauss rules on
integration-only subtriangles. Green's identity is a separate integration
path over the fixed domain, useful for authoring cross-checks.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from scipy.integrate import quad
from .fem_ref import Problem, element_gradients
from .mesh_checks import ValidatedMesh

__all__=['EnergyError','exact_gradient','energy_error','green_energy']


@dataclass(frozen=True)
class EnergyError:
    error_squared: float
    reference_squared: float
    relative: float


def exact_gradient(points, problem=None):
    p=Problem.from_config() if problem is None else problem
    x=np.asarray(points,dtype=float)
    if x.shape[-1:]!=(2,) or not np.isfinite(x).all():
        raise ValueError('points must be finite (...,2) coordinates')
    radius=np.hypot(x[...,0],x[...,1])
    if np.any(radius==0):
        raise ValueError('exact gradient is undefined at origin')
    angle=np.remainder(np.arctan2(x[...,1],x[...,0]),2*np.pi)
    singular=np.stack((-np.sin(2*angle/3),np.cos(2*angle/3)),axis=-1)
    singular[...,0]=np.where((x[...,0]==0)&(x[...,1]<0),0.,singular[...,0])
    return singular/(3*radius[...,None]**(2/3))-2*p.alpha*(x-p.x0)*p.peak(x)[...,None]


def energy_error(mesh, nodal_values, config=None, *, order=16,
                 max_cell_diameter=None, corner_panels=4, batch_size=128):
    """Return squared absolute/reference energies and relative energy error.

    x = t^3 ((1-s)*a+s*b) on origin triangles gives Jacobian
    3*t^5*|det(a,b)|. The singular squared gradient scales as t^-4,
    leaving a regular O(t) radial integrand. Radial panels shrink toward zero.
    No rule samples an endpoint/origin. Elsewhere use x=v0+t*((1-s)*a+s*b).

    The default cell diameter is min(.125,4/sqrt(alpha)); it resolves the
    peak with order 16. Tighten both diameter and order to check convergence.
    These are numerical settings, not a rigorous quadrature error bound.
    Boundary interpolation error and all cross terms are included directly.
    """
    if not isinstance(mesh,ValidatedMesh):
        raise TypeError('mesh must be a ValidatedMesh')
    for name,value in [('order',order),('corner_panels',corner_panels),('batch_size',batch_size)]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<1:
            raise ValueError(f'{name} must be a positive integer')
    p=Problem.from_config(config)
    limit=min(.125,4/np.sqrt(p.alpha)) if max_cell_diameter is None else max_cell_diameter
    if not np.isfinite(limit) or limit<=0:
        raise ValueError('max_cell_diameter must be finite and positive')
    raw=np.asarray(nodal_values)
    if raw.dtype.kind not in 'iuf' or raw.shape!=(len(mesh.vertices),) or not np.isfinite(raw).all():
        raise ValueError('nodal_values must be a finite real nodal vector')
    u=raw.astype(float,copy=False)
    basis=element_gradients(mesh)
    discrete=np.sum(u[mesh.triangles][...,None]*basis,axis=1)
    q,w=np.polynomial.legendre.leggauss(order)
    q,w=(q+1)/2,w/2
    total=np.zeros(2)
    panels=np.r_[0.,2.**np.arange(-(corner_panels-1),1)]

    def integrate(cells,grad):
        h=np.linalg.norm(cells[:,[1,2,0]]-cells[:,[0,1,2]],axis=2).max(axis=1)
        split=h>limit
        if split.any():
            a,b,c=cells[split,0],cells[split,1],cells[split,2]
            ab,bc,ca=(a+b)/2,(b+c)/2,(c+a)/2
            children=np.stack([np.stack(v,1) for v in [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]],1).reshape(-1,3,2)
            gradients=np.repeat(grad[split],4,axis=0)
            for start in range(0,len(children),batch_size):
                integrate(children[start:start+batch_size],gradients[start:start+batch_size])
        cells,grad=cells[~split],grad[~split]
        origin=np.all(cells==0,axis=2)
        for corner in (False,True):
            mask=origin.any(axis=1)==corner
            v=cells[mask]
            if not len(v): continue
            if corner:
                index=np.argmax(origin[mask],axis=1)
                v=v[np.arange(len(v))[:,None],(index[:,None]+np.arange(3))%3]
            a,b=v[:,1]-v[:,0],v[:,2]-v[:,0]
            det=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
            boundaries=panels if corner else [0.,1.]
            for lo,hi in zip(boundaries[:-1],boundaries[1:]):
                t=lo+(hi-lo)*q
                radius=t**3 if corner else t
                jac=3*t**5 if corner else t
                rays=(1-q)[None,:,None]*a[:,None]+q[None,:,None]*b[:,None]
                points=v[:,None,None,0]+radius[None,:,None,None]*rays[:,None,:,:]
                exact=exact_gradient(points,p)
                diff=exact-grad[mask,None,None,:]
                weight=det[:,None,None]*((hi-lo)*w*jac)[None,:,None]*w[None,None,:]
                total[0]+=np.sum(weight*np.sum(diff*diff,axis=-1))
                total[1]+=np.sum(weight*np.sum(exact*exact,axis=-1))
    for start in range(0,len(mesh.triangles),batch_size):
        integrate(mesh.vertices[mesh.triangles[start:start+batch_size]],discrete[start:start+batch_size])
    if not np.isfinite(total).all() or total[1]<=0:
        raise ValueError('invalid integrated energy')
    return EnergyError(float(total[0]),float(total[1]),float(np.sqrt(total[0]/total[1])))


def green_energy(config=None, *, epsabs=2e-12):
    """Independent reference energy from integral_boundary u*dn(u)+integral u*f.

    Uses adaptive 1D QUADPACK quadrature, nested over three fixed squares for
    the volume term; does not use a mesh, FEM gradients, or Duffy volume rules.
    Splits at Gaussian scales to prevent missed interior forcing. epsabs is
    per subintegral; agreement across tighter runs must be assessed globally.
    """
    if not np.isfinite(epsabs) or epsabs<=0: raise ValueError('epsabs must be positive')
    p=Problem.from_config(config)
    corners=np.array([[-1.,-1.],[0.,-1.],[0.,0.],[1.,0.],[1.,1.],[-1.,1.]])
    boundary=0.
    for a,b in zip(corners,np.roll(corners,-1,axis=0)):
        delta=b-a;length=np.linalg.norm(delta)
        normal=np.array([delta[1],-delta[0]])/length
        singular=np.all(a==0) or np.all(b==0)
        if np.all(b==0): a,b=b,a;delta=b-a
        def integrand(t):
            point=a+delta*(t**3 if singular else t)
            jac=3*t*t if singular else 1.
            return float(p.g_D(point)*np.dot(exact_gradient(point,p),normal)*length*jac)
        boundary+=quad(integrand,0,1,epsabs=epsabs,epsrel=epsabs,limit=200)[0]
    def breaks(lo,hi,center):
        offsets=np.array([-8.,-4.,-2.,0.,2.,4.,8.])/np.sqrt(p.alpha)+center
        return np.r_[lo,offsets[(offsets>lo)&(offsets<hi)],hi]
    volume=0.
    for xmin,xmax,ymin,ymax in [(-1.,0.,-1.,0.),(-1.,0.,0.,1.),(0.,1.,0.,1.)]:
        xs=breaks(xmin,xmax,p.x0[0]);ys=breaks(ymin,ymax,p.x0[1])
        for xl,xr in zip(xs[:-1],xs[1:]):
            for yl,yr in zip(ys[:-1],ys[1:]):
                def outer(x):
                    return quad(lambda y: float(p.g_D(np.array([x,y]))*p.f(np.array([x,y]))),
                                yl,yr,epsabs=epsabs,epsrel=epsabs,limit=150)[0]
                volume+=quad(outer,xl,xr,epsabs=epsabs,epsrel=epsabs,limit=150)[0]
    return float(boundary+volume)
