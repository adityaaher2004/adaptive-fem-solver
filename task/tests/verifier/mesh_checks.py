"""Independent schema, planar triangulation, and mixed-boundary checks.

No imports from the supplied fem package. Coordinates are interpreted as the
exact submitted floating-point values: no snapping or vertex merging. Boundary
coordinates must lie exactly on the six domain segments. Area uses a small
summation tolerance, not a geometric tolerance. Near-collinear predicates fall
back to exact rational arithmetic on the input floats.

validate_mesh returns immutable arrays for subsequent independent assembly.
validate_submission additionally checks the public JSON schema, but deliberately
does not trust/check the reported error or DOF claim against computed values.
Those comparisons and the sealed limits belong in test_submission.py.
"""
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree

__all__ = ['MeshValidationError', 'ValidatedMesh', 'load_config',
           'load_submission', 'read_submission', 'validate_schema', 'validate_submission', 'validate_mesh']

CORNERS = np.array([[-1.,-1.],[0.,-1.],[0.,0.],[1.,0.],[1.,1.],[-1.,1.]])
CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config.json'


class MeshValidationError(ValueError):
    """Malformed submission or invalid domain triangulation."""


def _fail(message):
    raise MeshValidationError(message)


def _object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            _fail(f'duplicate JSON key: {key}')
        obj[key] = value
    return obj


def _constant(value):
    _fail(f'nonfinite JSON constant: {value}')


def _read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=_object, parse_constant=_constant)


def load_config(path=CONFIG_PATH):
    """Read verifier-owned limits; never obtain configuration from a submission."""
    config = _read(path)
    if not isinstance(config, dict):
        _fail('config must be an object')
    for key in ('dof_budget', 'max_iter'):
        _integer(config.get(key), f'config.{key}')
    _real(config.get('error_tol'), 'config.error_tol')
    if config['max_iter'] < 1 or config['error_tol'] <= 0:
        _fail('config requires positive max_iter and error_tol')
    peak = config.get('problem')
    if not isinstance(peak, dict) or not {'A', 'alpha', 'x0'} <= peak.keys():
        _fail('config.problem must contain A, alpha and x0')
    for key in ('A', 'alpha'):
        value = peak[key]
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value)):
            _fail(f'config.problem.{key} must be a finite number')
    if peak['alpha'] <= 0:
        _fail('config.problem.alpha must be positive')
    x0 = peak['x0']
    if (not isinstance(x0, list) or len(x0) != 2
            or any(isinstance(c, bool) or not isinstance(c, Real) or not math.isfinite(c) for c in x0)):
        _fail('config.problem.x0 must be two finite numbers')
    if not (abs(x0[0]) < 1 and abs(x0[1]) < 1 and (x0[0] < 0 or x0[1] > 0)):
        _fail('config.problem.x0 must lie in the open L-shaped domain')
    return config


def _integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        _fail(f'{label} must be a nonnegative integer')


def _real(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        _fail(f'{label} must be a finite nonnegative number')
    try:
        valid = math.isfinite(value) and value >= 0
    except (OverflowError, TypeError):
        valid = False
    if not valid:
        _fail(f'{label} must be a finite nonnegative number')


def _finite_json(value):
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            _fail('JSON object keys must be strings')
        for v in value.values():
            _finite_json(v)
    elif isinstance(value, list):
        for v in value:
            _finite_json(v)
    elif value is None or isinstance(value, (str, bool, int)):
        pass
    elif isinstance(value, float):
        if not math.isfinite(value):
            _fail('nonfinite JSON number')
    else:
        _fail('unsupported JSON value')


def validate_schema(data):
    """Check metadata schema. Geometry is validated separately; extras are finite.

    Required history fields are iter, dof, estimator. Sequence numbering,
    history length, budget and reported-value agreement are separate checks;
    this routine imposes no requirements absent from the current public schema.
    """
    if not isinstance(data, dict):
        _fail('submission must be a JSON object')
    required = {'vertices', 'triangles', 'dof', 'reported_error', 'history'}
    if not required <= data.keys():
        _fail('missing submission fields: ' + ', '.join(sorted(required-data.keys())))
    _finite_json(data)
    _integer(data['dof'], 'dof')
    _real(data['reported_error'], 'reported_error')
    if not isinstance(data['history'], list):
        _fail('history must be a list')
    for i, record in enumerate(data['history']):
        if not isinstance(record, dict):
            _fail(f'history[{i}] must be an object')
        for key in ('iter', 'dof'):
            _integer(record.get(key), f'history[{i}].{key}')
        _real(record.get('estimator'), f'history[{i}].estimator')


def _size_limit(data, max_vertices):
    if max_vertices is None:
        return
    _integer(max_vertices, 'max_vertices')
    if max_vertices < 6:
        _fail('max_vertices must be at least six')
    if isinstance(data, dict):
        vertices, triangles = data.get('vertices'), data.get('triangles')
        if isinstance(vertices, list) and len(vertices) > max_vertices:
            _fail('vertex resource limit exceeded')
        # A planar triangular disk has T=2V-B-2 < 2V.
        if isinstance(triangles, list) and len(triangles) > 2*max_vertices:
            _fail('triangle resource limit exceeded')


def read_submission(path, *, max_vertices=None):
    """Strict JSON parse and early size guard, before schema/geometry work."""
    data = _read(path)
    _size_limit(data, max_vertices)
    return data


def validate_submission(data, *, max_vertices=None):
    """Validate schema and geometry, with optional early resource bounds."""
    _size_limit(data, max_vertices)
    validate_schema(data)
    return validate_mesh(data['vertices'], data['triangles'], max_vertices=max_vertices)


def load_submission(path, *, max_vertices=None):
    """Return (JSON object, validated mesh); reject duplicate keys/oversize."""
    data = read_submission(path, max_vertices=max_vertices)
    return data, validate_submission(data, max_vertices=max_vertices)


def _frozen(a):
    a = np.ascontiguousarray(a)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class ValidatedMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    areas: np.ndarray
    edges: np.ndarray
    triangle_edges: np.ndarray
    edge_triangles: np.ndarray
    boundary_edge_mask: np.ndarray
    boundary_vertex_mask: np.ndarray
    dirichlet_vertex_mask: np.ndarray
    neumann_edge_mask: np.ndarray

    @property
    def dof(self):
        return int(np.count_nonzero(~self.dirichlet_vertex_mask))

    @property
    def area(self):
        return math.fsum(self.areas)


def _orientation(a, b, c):
    """Robust orientation sign for floating-point input coordinates."""
    x, y = b-a, c-a
    left, right = x[0]*y[1], x[1]*y[0]
    determinant = left-right
    if abs(determinant) > 8*np.finfo(float).eps*(abs(left)+abs(right)):
        return 1 if determinant > 0 else -1
    ax, ay, bx, by, cx, cy = map(Fraction, map(float, (*a, *b, *c)))
    exact = (bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
    return (exact > 0)-(exact < 0)


def _geometry_checks(v, edges):
    # A midpoint ball contains the entire edge. KD queries are padded only
    # for candidate retrieval; exact predicates decide geometric incidence.
    points = cKDTree(v)
    a, b = v[edges[:,0]], v[edges[:,1]]
    mid = (a+b)/2
    radius = np.linalg.norm(b-a, axis=1)/2
    padding = 32*np.finfo(float).eps
    for i, (x, y) in enumerate(edges):
        candidates = points.query_ball_point(mid[i], radius[i]+padding)
        lo, hi = np.minimum(a[i],b[i]), np.maximum(a[i],b[i])
        for k in candidates:
            if k != x and k != y and np.all(v[k]>=lo) and np.all(v[k]<=hi):
                if _orientation(a[i],b[i],v[k]) == 0:
                    _fail(f'hanging vertex {k} lies inside edge ({x}, {y})')
    # Reject transverse intersections as well: edge incidence alone cannot
    # certify a planar triangulation. Collinear overlap was caught above.
    # Segments i, j can only meet if |mid_i - mid_j| <= r_i + r_j <=
    # 2*max(r_i, r_j), so each pair is examined once, from its longer edge
    # (ties by index), with radius 2*r_i. A global max-radius search would be
    # quadratic on graded meshes.
    tree = cKDTree(mid)
    for i in range(len(edges)):
        candidates = np.asarray(tree.query_ball_point(mid[i],2*radius[i]+padding),dtype=int)
        candidates = candidates[(radius[candidates]<radius[i])
                                | ((radius[candidates]==radius[i]) & (candidates>i))]
        if not len(candidates):
            continue
        keep = (np.maximum(a[candidates],b[candidates]) >= np.minimum(a[i],b[i])).all(axis=1)
        keep &= (np.minimum(a[candidates],b[candidates]) <= np.maximum(a[i],b[i])).all(axis=1)
        for j in candidates[keep]:
            if (edges[i,0] == edges[j,0] or edges[i,0] == edges[j,1]
                    or edges[i,1] == edges[j,0] or edges[i,1] == edges[j,1]):
                continue
            o1,o2 = _orientation(a[i],b[i],a[j]),_orientation(a[i],b[i],b[j])
            if o1*o2 < 0 and _orientation(a[j],b[j],a[i])*_orientation(a[j],b[j],b[i]) < 0:
                _fail(f'crossing edges {i} and {j}')


def validate_mesh(vertices, triangles, *, max_vertices=None):
    """Validate a complete CCW triangulation of the closed L-shaped domain.

    Checks all edge geometry, not just vertices/centroids: triangles cannot
    bridge the missing quadrant. Boundary segments must partition the six
    domain sides once. Together with positive orientation, planar edges and
    opposite interior incidences, this excludes holes and overlapping coverage.

    max_vertices, if given, rejects oversized meshes (and more than
    2*max_vertices triangles) before any per-element work.
    """
    if max_vertices is not None:
        for name, rows, limit in (('vertices', vertices, max_vertices),
                                  ('triangles', triangles, 2*max_vertices)):
            try:
                count = len(rows)
            except TypeError:
                count = 0  # malformed; reported by the shape checks below
            if count > limit:
                _fail(f'too many {name}: {count} > {limit}')
    # Object arrays preserve mixed bool/int and string/numeric entries so that
    # dtype coercion cannot silently sanitize a malformed submission.
    try:
        rv, rt = np.asarray(vertices,dtype=object), np.asarray(triangles,dtype=object)
    except (TypeError, ValueError) as exc:
        raise MeshValidationError('invalid array structure') from exc
    if rv.ndim!=2 or rv.shape[1]!=2 or len(rv)<6:
        _fail('vertices must have shape (N,2), N >= 6')
    if rt.ndim!=2 or rt.shape[1]!=3 or not len(rt):
        _fail('triangles must have nonempty shape (T,3)')
    for value in rv.flat:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real):
            _fail('vertex coordinates must be real numbers, not booleans')
    for value in rt.flat:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral):
            _fail('triangle indices must be integers, not booleans or floats')
        if value<0 or value>=len(rv):
            _fail('triangle index out of range')
    try:
        v,t = np.array(rv,dtype=float),np.array(rt,dtype=np.int64)
    except (ValueError,OverflowError) as exc:
        raise MeshValidationError('unrepresentable numeric value') from exc
    if not np.isfinite(v).all():
        _fail('vertex coordinates must be finite')
    if np.any(np.abs(v)>1) or np.any((v[:,0]>0)&(v[:,1]<0)):
        _fail('vertex outside domain or in cut-out')
    if len(np.unique(v,axis=0))!=len(v):
        _fail('duplicate vertices')
    if len(np.unique(t))!=len(v):
        _fail('unused vertices')
    ordered = np.sort(t,axis=1)
    if np.any(np.diff(ordered,axis=1)==0):
        _fail('triangle repeats a vertex')
    if len(np.unique(ordered,axis=0))!=len(t):
        _fail('duplicate triangles')
    tv=v[t]
    x,y=tv[:,1]-tv[:,0],tv[:,2]-tv[:,0]
    determinants=x[:,0]*y[:,1]-x[:,1]*y[:,0]
    for i in range(len(t)):
        if determinants[i]<=0 or _orientation(*tv[i])<=0:
            _fail(f'triangle {i} must have positive area and CCW orientation')
    areas=determinants/2
    if np.any(areas<=0):
        _fail('triangle area underflow')
    centroid=tv.mean(axis=1)
    if np.any((centroid[:,0]>0)&(centroid[:,1]<0)):
        _fail('triangle centroid in cut-out')
    for corner in CORNERS:
        if not np.any(np.all(v==corner,axis=1)):
            _fail(f'missing domain corner {corner.tolist()}')
    if not math.isclose(math.fsum(areas),3.,rel_tol=0,abs_tol=1e-11):
        _fail('total area must equal 3')
    directed=t[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    edges,inverse,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    if np.any(counts>2):
        _fail('nonmanifold edge has more than two incident triangles')
    if len(np.unique(directed,axis=0))!=len(directed):
        _fail('shared edge has inconsistent orientation (overlap)')
    order=np.argsort(inverse,kind='stable')
    starts=np.r_[0,np.cumsum(counts)[:-1]]
    owners=np.full((len(edges),2),-1,dtype=np.int64)
    owners[:,0]=order[starts]//3
    interior=counts==2
    owners[interior,1]=order[starts[interior]+1]//3
    boundary=counts==1
    ep=v[edges]
    side=np.full(len(edges),-1,dtype=int)
    for k,(a,b) in enumerate(zip(CORNERS,np.roll(CORNERS,-1,axis=0))):
        on=((ep>=np.minimum(a,b))&(ep<=np.maximum(a,b))).all(axis=(1,2))
        side[on]=k
        # Exact interval partition: gaps and overlaps in the boundary fail.
        axis=0 if a[0]!=b[0] else 1
        intervals=np.sort(ep[boundary&on,:,axis],axis=1)
        intervals=intervals[np.argsort(intervals[:,0])]
        if (not len(intervals) or intervals[0,0]!=min(a[axis],b[axis]) or
                intervals[-1,1]!=max(a[axis],b[axis]) or
                np.any(intervals[:-1,1]!=intervals[1:,0])):
            _fail(f'boundary does not partition domain side {k}')
    if np.any(side[boundary]<0):
        _fail('boundary edge or vertex is not on domain boundary')
    if np.any(side[interior]>=0):
        _fail('edge on domain boundary has two incident triangles')
    # If both endpoints are allowed, an edge enters the missing quadrant iff
    # its open parameter intervals for x>0 and y<0 intersect.
    for a,b in ep:
        lower,upper=0.,1.
        possible=True
        for start,delta in ((a[0],b[0]-a[0]),(-a[1],a[1]-b[1])):
            if delta==0:
                if start<=0: possible=False; break
            elif delta>0: lower=max(lower,-start/delta)
            else: upper=min(upper,-start/delta)
        if possible and lower<upper:
            _fail('triangle edge crosses the cut-out')
    _geometry_checks(v,edges)
    bv=np.zeros(len(v),bool)
    bv[edges[boundary].ravel()]=True
    neumann=boundary&(side==1)
    d=bv&~((v[:,0]==0)&(v[:,1]>-1)&(v[:,1]<0))
    fields=(v,t,areas,edges,inverse.reshape(-1,3),owners,boundary,bv,d,neumann)
    return ValidatedMesh(*map(_frozen,fields))
