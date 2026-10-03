"""Reference AFEM for authoring/calibration; exact energy is used only to stop.

Marking uses squared residual indicators, never exact local errors. DOF counts
free vertices. N_max counts solves (including the initial solve). A full marked
set plus NVB conformity closure is accepted only if it fits the budget.
The nonhomogeneous Dirichlet interpolation error is included in the monitored
true energy error, but not in the requested three-term residual indicator.
"""
import argparse
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The provided fem package lives in /app inside the task container and in
# environment/data in the authoring checkout.
for folder in (ROOT/'environment'/'data', ROOT/'data', Path('/app')):
    if (folder/'fem').is_dir():
        sys.path.insert(0,str(folder))
        break
else:
    raise ImportError('cannot locate the fem package (looked in environment/data, data, /app)')

from fem import problem
from fem.mesh import Mesh
from fem.basis import shape_functions
from fem.quadrature import triangle_rule, edge_rule
from fem.assembly import element_geometry, boundary_masks, assemble_stiffness, assemble_load, eliminate_dirichlet
from fem.solver import solve
from fem.error import energy_error
from fem.io import load_mesh, write_submission


@lru_cache(None)
def volume_rule(level):
    q,w=triangle_rule(8)
    cells=np.array([[[0.,0.],[1.,0.],[0.,1.]]])
    for _ in range(level):
        a,b,c=cells[:,0],cells[:,1],cells[:,2]
        ab,bc,ca=(a+b)/2,(b+c)/2,(c+a)/2
        cells=np.stack([np.stack(v,1) for v in [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]],1).reshape(-1,3,2)
    points=cells[:,None,0]+np.einsum('qi,tij->tqj',q,cells[:,1:]-cells[:,:1])
    return points.reshape(-1,2),np.tile(w/len(cells),len(cells))


def volume_terms(mesh, f, integration_h=.04):
    """Load vector and h_T^2 integral f^2, with per-cell composite quadrature.

    Integration subcells have diameter <= integration_h. Default .04 resolves
    the alpha=1000 peak; halve it to check sensitivity. No FEM DOFs are added.
    """
    if not np.isfinite(integration_h) or integration_h<=0:
        raise ValueError('integration_h must be positive and finite')
    vertices=mesh.vertices[mesh.triangles]
    h=np.linalg.norm(vertices[:,[1,2,0]]-vertices[:,[0,1,2]],axis=2).max(axis=1)
    levels=np.maximum(0,np.ceil(np.log2(h/integration_h))).astype(int)
    load=np.zeros(mesh.n_vertices)
    residual=np.zeros(mesh.n_triangles)
    for level in np.unique(levels):
        q,w=volume_rule(int(level))
        phi=shape_functions(q)
        indices=np.flatnonzero(levels==level)
        batch=max(1,65536//len(w))
        for start in range(0,len(indices),batch):
            ids=indices[start:start+batch]
            v=vertices[ids]
            points=v[:,None,0]+np.einsum('qi,tij->tqj',q,v[:,1:]-v[:,:1])
            values=np.broadcast_to(np.asarray(f(points),float),points.shape[:-1])
            if not np.isfinite(values).all():
                raise ValueError('nonfinite source')
            local=2*mesh.areas[ids,None]*np.einsum('tq,q,qi->ti',values,w,phi)
            np.add.at(load,mesh.triangles[ids].ravel(),local.ravel())
            residual[ids]=h[ids]**2*2*mesh.areas[ids]*np.einsum('tq,q->t',values**2,w)
    return load,residual


def residual_estimator(mesh,u,element_squared,g_N=problem.g_N):
    """eta_T^2 = h_T^2||f||_T^2 + half interior jumps + Neumann residuals.

    Interior edge contribution is h_E integral_E [grad u_h . n]^2; half
    goes to each adjacent cell. Boundary term is h_E integral_E(g_N-grad
    u_h.n)^2, assigned in full to its cell. Dirichlet edges add no term.
    """
    _,_,_,g=element_geometry(mesh)
    grad=np.einsum('ti,tij->tj',u[mesh.triangles],g)
    eta=np.array(element_squared,copy=True)
    ends=mesh.vertices[mesh.edges]
    tangent=ends[:,1]-ends[:,0]
    length=np.linalg.norm(tangent,axis=1)
    normal=np.column_stack((tangent[:,1],-tangent[:,0]))/length[:,None]
    pair=mesh.edge_triangles
    inside=pair[:,1]>=0
    jump=np.sum((grad[pair[inside,0]]-grad[pair[inside,1]])*normal[inside],axis=1)
    term=length[inside]**2*jump**2
    for side in (0,1):
        np.add.at(eta,pair[inside,side],term/2)
    _,N=boundary_masks(mesh)
    ids=np.flatnonzero(N)
    if len(ids):
        owner=pair[ids,0]
        midpoint=ends[ids].mean(axis=1)
        centroid=mesh.vertices[mesh.triangles[owner]].mean(axis=1)
        normals=normal[ids].copy()
        inward=np.sum(normals*(centroid-midpoint),axis=1)>0
        normals[inward]*=-1
        t,w=edge_rule(8)
        points=ends[ids,None,0]+t[None,:,None]*tangent[ids,None,:]
        diff=g_N(points)-np.sum(grad[owner]*normals,axis=1)[:,None]
        np.add.at(eta,owner,length[ids]**2*np.sum(w*diff**2,axis=1))
    return eta


def dorfler(eta_squared,theta=.4):
    """Minimal descending prefix carrying theta of total squared indicator."""
    eta=np.asarray(eta_squared,float)
    if not 0<theta<=1 or eta.ndim!=1 or not np.isfinite(eta).all() or (eta<0).any():
        raise ValueError('invalid theta or indicators')
    if eta.sum()==0:
        return np.empty(0,dtype=int)
    order=np.argsort(-eta,kind='stable')
    count=min(len(eta),np.searchsorted(np.cumsum(eta[order]),theta*eta.sum())+1)
    return order[:count]


def newest_vertex_bisection(mesh,marked):
    """Conforming NVB closure; vertex 0 newest, opposite edge (1,2).

    Boundary and interior reference edges share one midpoint. Children are
    (midpoint,v0,v1) and (midpoint,v2,v0), both CCW with midpoint newest.
    Neighbours are recursively prepared before bisecting their shared edge.
    Input mesh is immutable; the returned mesh may exceed a caller's budget.
    """
    vertices=mesh.vertices.tolist()
    active={k:tuple(map(int,t)) for k,t in enumerate(mesh.triangles)}
    edges={}
    next_id=len(active)
    def edge(a,b): return tuple(sorted((a,b)))
    def attach(k,t):
        active[k]=t
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
            edges.setdefault(edge(a,b),set()).add(k)
    def detach(k):
        t=active.pop(k)
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
            e=edge(a,b)
            edges[e].remove(k)
            if not edges[e]: del edges[e]
        return t
    for k,t in list(active.items()): attach(k,t)
    def bisect(k):
        nonlocal next_id
        if k not in active: return
        t=active[k]
        e=edge(t[1],t[2])
        neighbours=edges[e]-{k}
        if neighbours:
            neighbour=next(iter(neighbours))
            n=active[neighbour]
            if edge(n[1],n[2])!=e:
                bisect(neighbour)
                bisect(k)
                return
        owners=sorted(edges[e])
        midpoint=len(vertices)
        vertices.append(((np.asarray(vertices[e[0]])+vertices[e[1]])/2).tolist())
        parents=[detach(owner) for owner in owners]
        for a,b,c in parents:
            attach(next_id,(midpoint,a,b)); next_id+=1
            attach(next_id,(midpoint,c,a)); next_id+=1
    marked=np.asarray(marked)
    if marked.ndim!=1 or (marked.size and (marked.dtype.kind not in 'iu' or (marked<0).any() or (marked>=mesh.n_triangles).any())):
        raise ValueError('marked must contain valid triangle indices')
    for k in marked: bisect(int(k))
    return Mesh(vertices,list(active.values()))


def run(mesh,*,tol=.1,budget=10000,N_max=80,theta=.4,integration_h=.04,error_order=12):
    """Reference solve: exact global error monitors tol; residuals mark cells."""
    if not np.isfinite(tol) or tol<=0 or budget<0 or N_max<1 or not 0<theta<=1:
        raise ValueError('invalid stopping limits or theta')
    if np.count_nonzero(~boundary_masks(mesh)[0])>budget:
        raise ValueError('initial mesh exceeds DOF budget')
    history=[]
    for iteration in range(N_max):
        load,element=volume_terms(mesh,problem.f,integration_h)
        load+=assemble_load(mesh,lambda p:0.,problem.g_N)
        system=eliminate_dirichlet(mesh,assemble_stiffness(mesh),load)
        u=solve(system)
        energy=energy_error(mesh,u,order=error_order)
        eta=residual_estimator(mesh,u,element)
        dof=len(system.free_vertices)
        row=dict(iter=iteration,dof=dof,vertices=mesh.n_vertices,triangles=mesh.n_triangles,
                 reported_error=energy.relative,estimator=float(np.sqrt(eta.sum())),
                 relative_estimator=float(np.sqrt(eta.sum())/energy.reference_norm))
        history.append(row)
        print(f'{iteration:3d} DOF={dof:6d} error={energy.relative:.6e} eta={row["estimator"]:.6e}',flush=True)
        if energy.relative<=tol: reason='tol'; break
        if dof>=budget: reason='budget'; break
        if iteration+1==N_max: reason='N_max'; break
        marked=dorfler(eta,theta)
        row['marked']=len(marked)
        if not len(marked): reason='zero_estimator'; break
        candidate=newest_vertex_bisection(mesh,marked)
        candidate_dof=int(np.count_nonzero(~boundary_masks(candidate)[0]))
        if candidate_dof>budget:
            row['candidate_dof']=candidate_dof
            reason='budget'; break
        mesh=candidate
    history[-1]['stop_reason']=reason
    return mesh,u,history


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path)
    parser.add_argument('--output',type=Path,default=Path('/app/output/submission.json'))
    parser.add_argument('--tol',type=float,default=.1)
    parser.add_argument('--budget',type=int,default=10000)
    parser.add_argument('--N-max', '--N_max',dest='N_max',type=int,default=80)
    parser.add_argument('--theta',type=float,default=.4)
    parser.add_argument('--integration-h',type=float,default=.04)
    parser.add_argument('--error-order',type=int,default=12)
    args=parser.parse_args(argv)
    mesh=load_mesh(args.mesh) if args.mesh else problem.initial_mesh()
    mesh,u,history=run(mesh,tol=args.tol,budget=args.budget,N_max=args.N_max,
                       theta=args.theta,integration_h=args.integration_h,error_order=args.error_order)
    last=history[-1]
    write_submission(args.output,mesh,dof=last['dof'],reported_error=last['reported_error'],history=history)
    print('Stopped:',last['stop_reason'],'Output:',args.output)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
