"""Reference adaptive solver: exact-error marking, NVB, optional coarsening.

Each step solves the P1 problem, computes the elementwise energy error
eta_T^2 = int_T |grad u - grad u_h|^2 against the publicly supplied exact
gradient (fem.problem.grad_u), marks with Doerfler, and refines with conforming
newest-vertex bisection (NVB). The task allows any marking strategy; only the
submitted mesh is judged. The verifier is never imported.

Defaults for tol / DOF budget / N_max are read from specs.txt (/app/specs.txt in
the task container, environment/data/specs.txt in the authoring checkout).
All solves, including a coarsening solve, count toward N_max.
"""
import argparse
from functools import lru_cache
from numbers import Integral
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The provided fem package lives in /app inside the task container and in
# environment/data in the authoring checkout.
for DATA in (ROOT/'environment'/'data', ROOT/'data', Path('/app')):
    if (DATA/'fem').is_dir():
        sys.path.insert(0, str(DATA))
        break
else:
    raise ImportError('cannot locate the fem package (looked in environment/data, data, /app)')

from fem import problem  # noqa: E402
from fem.mesh import Mesh  # noqa: E402
from fem.basis import shape_functions  # noqa: E402
from fem.quadrature import triangle_rule  # noqa: E402
from fem.assembly import (element_geometry, boundary_masks, assemble_stiffness,  # noqa: E402
                          assemble_load, eliminate_dirichlet)
from fem.solver import solve  # noqa: E402
from fem.problem import grad_u  # noqa: E402
from fem.io import load_mesh, write_submission  # noqa: E402

# Doerfler fraction. With the near-tolerance rule below, theta in 0.3-0.6 meets
# the task limits (3500 DOF, 50 solves); 0.2 needs too many solves and 0.8
# overshoots the DOF budget (authoring/evidence/calibration.md, theta sweep).
THETA = .35
# Refinement stops at error <= TARGET_RATIO * tol: ~1e-3 of headroom below tol
# (the verifier's recomputed error agrees with fem.error to ~1e-16).
TARGET_RATIO = .99
# Near the tolerance (error < NEAR * tol) the marked fraction shrinks to the
# share of squared error that still has to go: error^2 must fall from err^2 to
# about (AIM * tol)^2, i.e. by 1 - (AIM*tol/err)^2. GAIN over-marks to allow for
# error that refinement does not remove; FLOOR keeps every step productive.
# This reaches the tolerance with the fewest extra DOF.
NEAR, AIM, GAIN, FLOOR = 1.3, .999, 1.5, .06
# --reproduce-submission-5: budget and temporary excess that, from a cold
# start, reproduce authoring's submission_5 mesh exactly (2992 DOF).
SUBMISSION_5 = dict(budget=2992, excess=.02)


def read_specs(path=DATA/'specs.txt'):
    """Task limits (tol, budget, N_max) from specs.txt."""
    text = Path(path).read_text(encoding='utf-8')
    value = lambda label: re.search(rf'^{label}:\s*(\S+)', text, re.M).group(1)
    return dict(tol=float(value('Error tolerance')), budget=int(value('DOF budget')),
                N_max=int(value('N_max iter')))


SPECS = read_specs()


@lru_cache(None)
def volume_rule(level):
    q, w = triangle_rule(8)
    cells = np.array([[[0., 0.], [1., 0.], [0., 1.]]])
    for _ in range(level):
        a, b, c = cells[:, 0], cells[:, 1], cells[:, 2]
        ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
        cells = np.stack([np.stack(v, 1) for v in [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]], 1).reshape(-1, 3, 2)
    points = cells[:, None, 0] + np.einsum('qi,tij->tqj', q, cells[:, 1:]-cells[:, :1])
    return points.reshape(-1, 2), np.tile(w/len(cells), len(cells))


def volume_load(mesh, f, integration_h=.04):
    """Load vector with per-cell composite quadrature (subcells <= integration_h).

    Default .04 resolves the alpha=1000 peak even on the 6-triangle start mesh;
    halve it to check sensitivity. No FEM DOFs are added.
    """
    if not np.isfinite(integration_h) or integration_h <= 0:
        raise ValueError('integration_h must be positive and finite')
    vertices = mesh.vertices[mesh.triangles]
    h = np.linalg.norm(vertices[:, [1, 2, 0]]-vertices[:, [0, 1, 2]], axis=2).max(axis=1)
    levels = np.maximum(0, np.ceil(np.log2(h/integration_h))).astype(int)
    load = np.zeros(mesh.n_vertices)
    for level in np.unique(levels):
        q, w = volume_rule(int(level))
        phi = shape_functions(q)
        indices = np.flatnonzero(levels == level)
        batch = max(1, 65536//len(w))
        for start in range(0, len(indices), batch):
            ids = indices[start:start+batch]
            v = vertices[ids]
            points = v[:, None, 0] + np.einsum('qi,tij->tqj', q, v[:, 1:]-v[:, :1])
            values = np.broadcast_to(np.asarray(f(points), float), points.shape[:-1])
            if not np.isfinite(values).all():
                raise ValueError('nonfinite source')
            local = 2*mesh.areas[ids, None]*np.einsum('tq,q,qi->ti', values, w, phi)
            np.add.at(load, mesh.triangles[ids].ravel(), local.ravel())
    return load


def dorfler(eta_squared, theta=.4):
    """Minimal descending prefix carrying theta of total squared indicator."""
    eta = np.asarray(eta_squared, float)
    if not 0 < theta <= 1 or eta.ndim != 1 or not np.isfinite(eta).all() or (eta < 0).any():
        raise ValueError('invalid theta or indicators')
    if eta.sum() == 0:
        return np.empty(0, dtype=int)
    order = np.argsort(-eta, kind='stable')
    count = min(len(eta), np.searchsorted(np.cumsum(eta[order]), theta*eta.sum())+1)
    return order[:count]


def newest_vertex_bisection(mesh, marked):
    """Conforming NVB closure; vertex 0 newest, opposite edge (1,2).

    Boundary and interior reference edges share one midpoint. Children are
    (midpoint,v0,v1) and (midpoint,v2,v0), both CCW with midpoint newest.
    Neighbours are recursively prepared before bisecting their shared edge.
    Input mesh is immutable; the returned mesh may exceed a caller's budget.
    """
    vertices = mesh.vertices.tolist()
    active = {k: tuple(map(int, t)) for k, t in enumerate(mesh.triangles)}
    edges = {}
    next_id = len(active)

    def edge(a, b):
        return tuple(sorted((a, b)))

    def attach(k, t):
        active[k] = t
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges.setdefault(edge(a, b), set()).add(k)

    def detach(k):
        t = active.pop(k)
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e = edge(a, b)
            edges[e].remove(k)
            if not edges[e]:
                del edges[e]
        return t

    for k, t in list(active.items()):
        attach(k, t)

    def bisect(k):
        nonlocal next_id
        if k not in active:
            return
        t = active[k]
        e = edge(t[1], t[2])
        neighbours = edges[e]-{k}
        if neighbours:
            neighbour = next(iter(neighbours))
            n = active[neighbour]
            if edge(n[1], n[2]) != e:
                bisect(neighbour)
                bisect(k)
                return
        owners = sorted(edges[e])
        midpoint = len(vertices)
        vertices.append(((np.asarray(vertices[e[0]])+vertices[e[1]])/2).tolist())
        parents = [detach(owner) for owner in owners]
        for a, b, c in parents:
            attach(next_id, (midpoint, a, b))
            next_id += 1
            attach(next_id, (midpoint, c, a))
            next_id += 1

    marked = np.asarray(marked)
    if marked.ndim != 1 or (marked.size and (marked.dtype.kind not in 'iu' or (marked < 0).any()
                                             or (marked >= mesh.n_triangles).any())):
        raise ValueError('marked must contain valid triangle indices')
    for k in marked:
        bisect(int(k))
    return Mesh(vertices, list(active.values()))


def local_energy_error(mesh, nodal_values, exact_gradient=grad_u, *, order=12,
                       max_cell_diameter=0.125, batch_size=256):
    """Per-element squared energy error and total ||grad u||^2.

    Deliberately a per-element copy of fem.error.energy_error (same corner
    t^3 Duffy map and integration-only subdivision), so the agent-facing fem
    package stays unchanged. Returns (eta_T^2 array, ||grad u||^2).
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
    error = np.zeros(mesh.n_triangles)
    reference = 0.

    def integrate(cells, gradients, owners):
        nonlocal reference
        diameter = np.max(np.linalg.norm(cells[:, [1, 2, 0]]-cells[:, [0, 1, 2]], axis=2), axis=1)
        split = diameter > max_cell_diameter
        if split.any():
            a, b, c = cells[split, 0], cells[split, 1], cells[split, 2]
            ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
            children = np.stack([np.stack(v, axis=1) for v in
                                 [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]], axis=1).reshape(-1, 3, 2)
            child_grad = np.repeat(gradients[split], 4, axis=0)
            child_owner = np.repeat(owners[split], 4)
            for start in range(0, len(children), batch_size):
                integrate(children[start:start+batch_size], child_grad[start:start+batch_size],
                          child_owner[start:start+batch_size])
        cells, gradients, owners = cells[~split], gradients[~split], owners[~split]
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
                v = v[np.arange(len(v))[:, None], (indices[:, None]+np.arange(3)) % 3]
            a, b = v[:, 1]-v[:, 0], v[:, 2]-v[:, 0]
            determinant = np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
            radius = t**3 if special else t
            jacobian = 3*t**5 if special else t
            points = v[:, None, 0] + radius[None, :, None]*((1-s)[None, :, None]*a[:, None]+s[None, :, None]*b[:, None])
            exact = np.asarray(exact_gradient(points), dtype=float)
            if exact.shape != points.shape or not np.isfinite(exact).all():
                raise ValueError('exact_gradient must return finite vectors matching points')
            difference = exact-gradients[selected, None, :]
            weight = determinant[:, None]*(weights*jacobian)[None, :]
            np.add.at(error, owners[selected], np.sum(weight*np.sum(difference*difference, axis=2), axis=1))
            reference += float(np.sum(weight*np.sum(exact*exact, axis=2)))

    for start in range(0, mesh.n_triangles, batch_size):
        ids = mesh.triangles[start:start+batch_size]
        _, _, _, basis_grad = element_geometry(mesh, slice(start, start+batch_size))
        gradient = np.einsum('ti,tij->tj', u[ids], basis_grad)
        integrate(mesh.vertices[ids], gradient, np.arange(start, start+len(ids)))
    return error, reference


def remove_vertices(mesh, score, count):
    """Remove up to count interior vertices with the lowest incident score.

    Each removed vertex's star is re-triangulated by ear clipping; stars of
    removed vertices are disjoint. The caller re-solves to accept or reject.
    """
    triangles = [tuple(t) for t in mesh.triangles]
    v = mesh.vertices
    active = set(range(len(triangles)))
    incident = {}                                  # vertex -> active triangle ids
    for i, tri in enumerate(triangles):
        for x in tri:
            incident.setdefault(x, set()).add(i)
    vertex_cost = np.zeros(len(v))
    np.add.at(vertex_cost, mesh.triangles.ravel(), np.repeat(score, 3))
    vertex_cost[mesh.boundary_vertex_mask] = np.inf
    removed, used = [], set()

    def cross(a, b, c):
        x = v[b]-v[a]
        y = v[c]-v[a]
        return x[0]*y[1]-x[1]*y[0]

    for node in np.argsort(vertex_cost):
        if not np.isfinite(vertex_cost[node]) or len(removed) >= count:
            break
        star = sorted(incident.get(node, ()))
        if not star:
            continue
        ring = set(x for i in star for x in triangles[i] if x != node)
        if ring & used:
            continue
        # Oriented edges opposite the removed vertex form the CCW cavity.
        nxt = {}
        for i in star:
            tri = triangles[i]
            j = tri.index(node)
            nxt[tri[(j+1) % 3]] = tri[(j+2) % 3]
        start = next(iter(nxt))
        poly = [start]
        while nxt[poly[-1]] != start:
            poly.append(nxt[poly[-1]])
        new = []
        while len(poly) > 3:
            for k, b in enumerate(poly):
                a, c = poly[k-1], poly[(k+1) % len(poly)]
                if cross(a, b, c) <= 0:
                    continue
                if not any(min(cross(a, b, x), cross(b, c, x), cross(c, a, x)) >= 0
                           for x in poly if x not in (a, b, c)):
                    new.append((a, b, c))
                    poly.pop(k)
                    break
            else:
                raise RuntimeError('cavity triangulation')
        new.append(tuple(poly))
        if any(cross(*tri) <= 0 for tri in new):
            continue
        for i in star:
            active.discard(i)
            for x in triangles[i]:
                incident[x].discard(i)
        for tri in new:
            i = len(triangles)
            triangles.append(tri)
            active.add(i)
            for x in tri:
                incident.setdefault(x, set()).add(i)
        removed.append(node)
        used.update(ring)
    keep = np.ones(len(v), bool)
    keep[removed] = False
    mapping = np.cumsum(keep)-1
    return Mesh(v[keep], mapping[np.array([triangles[i] for i in sorted(active)])])


def free_dof(mesh):
    return int(np.count_nonzero(~boundary_masks(mesh)[0]))


def run(mesh, *, tol=SPECS['tol'], budget=SPECS['budget'], N_max=SPECS['N_max'], theta=THETA,
        integration_h=.04, error_order=12, excess=0., target_ratio=TARGET_RATIO):
    """Refine until the exact error is <= target_ratio * tol; stop gracefully otherwise.

    Stops (history[-1]['stop_reason']): 'tol', 'budget' (the next refinement,
    or a needed coarsening, would leave the DOF budget), 'N_max' (solve limit
    reached) or 'zero_estimator'. Never raises mid-run: the returned mesh is
    always the last solved mesh within budget, marked submitted=True in history.
    excess > 0 lets refinement temporarily exceed the budget by that fraction,
    after which cavity coarsening must bring the mesh back within budget and
    tolerance. A starting mesh over budget is rejected with ValueError.
    """
    if (not np.isfinite(tol) or tol <= 0 or budget < 1 or N_max < 1 or not 0 < theta <= 1
            or excess < 0 or not 0 < target_ratio <= 1):
        raise ValueError('invalid tolerance, budget, solve limit, theta, excess or target_ratio')
    if free_dof(mesh) > budget:
        raise ValueError('initial mesh exceeds DOF budget')
    target = target_ratio*tol
    history = []

    def evaluate(candidate, phase):
        load = volume_load(candidate, problem.f, integration_h)
        load += assemble_load(candidate, lambda p: 0., problem.g_N)
        system = eliminate_dirichlet(candidate, assemble_stiffness(candidate), load)
        u = solve(system)
        eta, norm = local_energy_error(candidate, u, order=error_order)
        err = float(np.sqrt(eta.sum()/norm))
        history.append(dict(iter=len(history), dof=len(system.free_vertices),
                            estimator=float(np.sqrt(eta.sum())), reported_error=err, phase=phase))
        print(f'{len(history):3d} {phase:8s} DOF={len(system.free_vertices):5d} error={err:.12g}', flush=True)
        return u, eta, err

    u, eta, err = evaluate(mesh, 'initial')
    best = (mesh, u, len(history)-1)          # last solved mesh within budget
    while True:
        if err <= target:
            reason = 'tol'
            break
        if len(history) >= N_max:
            reason = 'N_max'
            break
        fraction = theta
        if err < NEAR*tol:
            fraction = min(theta, max(FLOOR, GAIN*(1-(AIM*tol/err)**2)))
        marked = dorfler(eta, fraction)
        if not len(marked):
            reason = 'zero_estimator'
            break
        candidate = newest_vertex_bisection(mesh, marked)
        candidate_dof = free_dof(candidate)
        if candidate_dof > budget*(1+excess) or candidate.n_vertices > 4*budget+16:
            history[-1]['candidate_dof'] = candidate_dof
            reason = 'budget'
            break
        mesh = candidate
        u, eta, err = evaluate(mesh, 'refine')
        if history[-1]['dof'] <= budget:
            best = (mesh, u, len(history)-1)

    if history[-1]['dof'] > budget:
        # Only reachable with excess > 0: coarsen the cheapest disjoint interior
        # stars; the exact error after re-solving decides acceptance.
        if len(history) >= N_max:
            reason = 'N_max'
        else:
            candidate = remove_vertices(mesh, eta, history[-1]['dof']-budget)
            if free_dof(candidate) > budget:
                reason = 'budget'
            else:
                cu, _, cerr = evaluate(candidate, 'coarsen')
                if cerr <= tol:
                    best = (candidate, cu, len(history)-1)
                else:
                    reason = 'budget'
    mesh, u, index = best
    history[index]['submitted'] = True
    history[-1]['stop_reason'] = reason
    return mesh, u, history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mesh', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/app/output/submission.json'))
    parser.add_argument('--tol', type=float, default=SPECS['tol'])
    parser.add_argument('--budget', type=int, default=SPECS['budget'])
    parser.add_argument('--N-max', '--N_max', dest='N_max', type=int, default=SPECS['N_max'])
    parser.add_argument('--theta', type=float, default=THETA)
    parser.add_argument('--target-ratio', type=float, default=TARGET_RATIO,
                        help='stop refining at error <= target_ratio * tol')
    parser.add_argument('--excess', type=float, default=0.,
                        help='allowed temporary DOF excess fraction (requires coarsening)')
    parser.add_argument('--reproduce-submission-5', action='store_true',
                        help=f'use {SUBMISSION_5} (overrides --budget/--excess)')
    parser.add_argument('--integration-h', type=float, default=.04)
    parser.add_argument('--error-order', type=int, default=12)
    args = parser.parse_args(argv)
    if args.reproduce_submission_5:
        args.budget, args.excess = SUBMISSION_5['budget'], SUBMISSION_5['excess']
    mesh = load_mesh(args.mesh) if args.mesh else problem.initial_mesh()
    mesh, u, history = run(mesh, tol=args.tol, budget=args.budget, N_max=args.N_max, theta=args.theta,
                           integration_h=args.integration_h, error_order=args.error_order,
                           excess=args.excess, target_ratio=args.target_ratio)
    submitted = next(r for r in reversed(history) if r.get('submitted'))
    write_submission(args.output, mesh, dof=submitted['dof'],
                     reported_error=submitted['reported_error'], history=history)
    print('Stopped:', history[-1]['stop_reason'], 'Output:', args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
