"""Mesh and submission JSON I/O (UTF-8, zero-based CCW triangles).

Mesh files contain vertices (N,2) and triangles (T,3). Submission files add
an explicit integer dof, a finite nonnegative reported_error, and history (a
list of JSON objects, one per adaptive iteration). Each history record must
contain:

    iter       nonnegative integer iteration index
    dof        nonnegative integer DOF count of that iteration's mesh
    estimator  finite nonnegative real a posteriori error estimate

Additional keys are allowed and passed through unchanged. DOF follows
specs.txt: the number of vertices NOT on Gamma_D, i.e. len(free_vertices) of
the DirichletSystem. This module checks types and ranges only; it does not
certify the reported DOF, error, or history against the mesh.
"""
import json
import os
from pathlib import Path
import tempfile
from numbers import Integral, Real

import numpy as np

from .mesh import Mesh

__all__ = ['load_mesh', 'save_mesh', 'write_submission']


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f'nonfinite JSON number: {value}')


def load_mesh(path):
    """Read a mesh or submission file and reconstruct Mesh connectivity.

    Extra top-level fields are ignored so submissions can be loaded as meshes.
    Geometry/topology checks are those of Mesh; this is not the independent
    verifier's domain-coverage, overlap, or hanging-node validation.
    """
    with Path(path).open('r', encoding='utf-8') as stream:
        data = json.load(stream, object_pairs_hook=_unique_object,
                         parse_constant=_invalid_constant)
    if not isinstance(data, dict) or not {'vertices', 'triangles'} <= data.keys():
        raise ValueError(
            'mesh JSON must be an object with vertices and triangles')
    # Check scalar types before NumPy can coerce mixed booleans into integers.
    for name, width, integer in [('vertices', 2, False), ('triangles', 3, True)]:
        rows = data[name]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'{name} must be a nonempty array')
        for row in rows:
            if not isinstance(row, list) or len(row) != width:
                raise ValueError(f'{name} rows must have length {width}')
            for value in row:
                if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
                    raise TypeError(f'invalid numeric value in {name}')
    return Mesh(data['vertices'], data['triangles'])


def _mesh_data(mesh):
    if not isinstance(mesh, Mesh):
        raise TypeError('mesh must be a Mesh')
    return {'vertices': mesh.vertices.tolist(), 'triangles': mesh.triangles.tolist()}


def _json_value(value):
    """Normalize NumPy values while enforcing JSON types and string keys."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError('history object keys must be strings')
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'unsupported JSON value: {type(value).__name__}')


def _write(path, data):
    # Validate/serialize before touching the destination, including NaN/Inf in
    # nested history. Same-directory replacement is atomic on local filesystems.
    text = json.dumps(data, indent=2, ensure_ascii=False,
                      allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def save_mesh(path, mesh):
    """Atomically write vertices and triangles; create missing parent folders."""
    _write(path, _mesh_data(mesh))


def _history_record(index, record):
    for key in ('iter', 'dof'):
        if key not in record:
            raise ValueError(f'history[{index}] is missing required key {key!r}')
        value = record[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f'history[{index}][{key!r}] must be an integer')
        if value < 0:
            raise ValueError(f'history[{index}][{key!r}] must be nonnegative')
    if 'estimator' not in record:
        raise ValueError(f"history[{index}] is missing required key 'estimator'")
    value = record['estimator']
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"history[{index}]['estimator'] must be a real number")
    if not np.isfinite(float(value)) or value < 0:
        raise ValueError(
            f"history[{index}]['estimator'] must be finite and nonnegative")


def write_submission(path, mesh, *, dof, reported_error, history):
    """Write exactly vertices, triangles, dof, reported_error, and history.

    dof is the specs.txt count of vertices not on Gamma_D: pass
    len(system.free_vertices) from assemble_system. It must lie in [0,N].
    reported_error is the caller's finite nonnegative relative energy error.
    history must be a list of dicts, each with the required iter, dof and
    estimator keys (see the module docstring); extra keys are kept. NumPy
    scalars/arrays are converted to JSON. No consistency checks between
    history, the final mesh, the tolerance, or the DOF budget are performed.
    """
    data = _mesh_data(mesh)
    if isinstance(dof, (bool, np.bool_)) or not isinstance(dof, Integral):
        raise TypeError('dof must be an integer')
    if not 0 <= dof <= mesh.n_vertices:
        raise ValueError(
            'dof must lie between zero and the number of vertices')
    if isinstance(reported_error, (bool, np.bool_)) or not isinstance(reported_error, Real):
        raise TypeError('reported_error must be a real number')
    error = float(reported_error)
    if not np.isfinite(error) or error < 0:
        raise ValueError('reported_error must be finite and nonnegative')
    if not isinstance(history, list) or not all(isinstance(item, dict) for item in history):
        raise TypeError('history must be a list of objects')
    for index, record in enumerate(history):
        _history_record(index, record)
    data.update(dof=int(dof), reported_error=error,
                history=_json_value(history))
    _write(path, data)
