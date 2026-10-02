"""Generate environment/data/initial_mesh.json from fem.problem.initial_mesh().

Usage (from the task root):

    python authoring/provenance/make_initial_mesh.py          # write the file
    python authoring/provenance/make_initial_mesh.py --check  # verify, exit 1 on drift

problem.initial_mesh() is the single source of truth. Before writing, this
script checks the properties the task relies on:

  - Mesh validity (CCW, conforming, manifold) via fem.mesh.Mesh
  - total area 3 and all six domain corners present
  - boundary classification: one Neumann edge {x=0, -1<=y<=0}, and both
    junction vertices (0,0), (0,-1) are Dirichlet
  - newest-vertex-bisection labelling: the refinement edge of each triangle
    (local vertices 1-2, opposite local vertex 0) is its longest edge, and
    every interior refinement edge is the refinement edge of both of its
    triangles (compatible labelling)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
DATA = TASK / 'environment' / 'data'
OUTPUT = DATA / 'initial_mesh.json'
sys.path.insert(0, str(DATA))

from fem import problem  # noqa: E402
from fem.io import load_mesh, save_mesh  # noqa: E402


def check(mesh):
    if mesh.area != 3:
        raise AssertionError(f'area {mesh.area} != 3')
    dirichlet, neumann = problem.boundary_masks(mesh)  # also checks corners
    if neumann.sum() != 1:
        raise AssertionError('expected exactly one Neumann edge')
    for corner in ([0, 0], [0, -1]):
        if not dirichlet[np.all(mesh.vertices == corner, axis=1)].all():
            raise AssertionError(f'junction {corner} is not Dirichlet')

    v, t = mesh.vertices, mesh.triangles
    lengths = np.linalg.norm(v[t[:, [1, 2, 0]]] - v[t[:, [2, 0, 1]]], axis=2)
    # lengths[:, 0] is edge (1,2), opposite local vertex 0.
    if not np.all(lengths[:, 0] > lengths[:, 1:].max(axis=1)):
        raise AssertionError('refinement edge is not the strict longest edge')

    # triangle_edges uses local order (0,1),(1,2),(2,0): edge (1,2) is column 1.
    refinement = mesh.triangle_edges[:, 1]
    for k, e in enumerate(refinement):
        other = mesh.edge_triangles[e]
        other = other[(other >= 0) & (other != k)]
        if len(other) and refinement[other[0]] != e:
            raise AssertionError(
                f'triangle {k}: refinement edge not shared by neighbour {other[0]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true',
                        help='verify the existing file instead of writing it')
    args = parser.parse_args()

    mesh = problem.initial_mesh()
    check(mesh)
    if args.check:
        try:
            stored = load_mesh(OUTPUT)
        except (OSError, ValueError, TypeError) as exc:
            print(f'{OUTPUT.relative_to(TASK)}: OUT OF DATE (unreadable: {exc})')
            return 1
        same = (np.array_equal(stored.vertices, mesh.vertices)
                and np.array_equal(stored.triangles, mesh.triangles))
        print(f'{OUTPUT.relative_to(TASK)}: {"OK" if same else "OUT OF DATE"}')
        return 0 if same else 1
    save_mesh(OUTPUT, mesh)
    print(f'wrote {OUTPUT.relative_to(TASK)}: {mesh.n_vertices} vertices, '
          f'{mesh.n_triangles} triangles')
    return 0


if __name__ == '__main__':
    sys.exit(main())
