"""Generate the cheat-attempt submissions in authoring/evidence/cheat_attempts/.

Run from anywhere: python authoring/provenance/make_cheat_attempts.py

Each file is a plausible wrong answer that the verifier must score 0. Apart
from fake_error.json, every file reports its DOF and true relative energy
error honestly, so it fails only for the reason it was built for. Expected
failing verifier tests are in EXPECTED and checked by
authoring/provenance/check_cheat_attempts.py and
authoring/tests/test_cheat_attempts.py.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'authoring/evidence/cheat_attempts'
sys.path.insert(0, str(ROOT/'environment/data'))
from fem import problem  # noqa: E402
from fem.mesh import Mesh  # noqa: E402

spec = importlib.util.spec_from_file_location('calibrate', ROOT/'authoring/provenance/calibrate.py')
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)
afem = calibrate.afem
BUDGET = json.loads((ROOT/'tests/config.json').read_text(encoding='utf-8'))['dof_budget']

# Verifier tests that must fail (FAILED or ERROR) for each file; all others pass.
# Invalid meshes fail the shared `validated` fixture, so every test using it errors.
MESH_INVALID = {'test_mesh_valid', 'test_dof_matches_reported', 'test_dof_budget',
                'test_computed_error', 'test_reported_error'}
EXPECTED = {
    'uniform_mesh': {'test_computed_error'},
    'corner_only_graded': {'test_computed_error'},
    'peak_ignored': {'test_computed_error'},
    'fake_error': {'test_computed_error', 'test_reported_error'},
    'hanging_nodes': MESH_INVALID,
    'wrong_domain': MESH_INVALID,
    # The computed_error fixture refuses to solve an over-budget mesh.
    'over_budget': {'test_dof_budget', 'test_computed_error', 'test_reported_error'},
}
WHY = {
    'uniform_mesh': 'Largest uniform refinement within the DOF budget; error far above tolerance',
    'corner_only_graded': 'Geometric grading toward the corner only (never looks at f); misses the peak',
    'peak_ignored': 'Residual AFEM that solves and marks with f = 0; peak never refined',
    'fake_error': 'Coarse uniform mesh with a fabricated reported_error of 0.04',
    'hanging_nodes': 'One triangle red-refined without closure: hanging vertices on its edges',
    'wrong_domain': 'Uniform mesh of the full square, covering the removed quadrant',
    'over_budget': 'Residual AFEM run to tol 0.05: meets the error target with too many DOF (still under the vertex cap)',
}


def dirichlet_dof(vertices):
    """Vertices not on Gamma_D, by coordinates (works for invalid meshes too)."""
    v = np.asarray(vertices, float)
    x, y = v[:, 0], v[:, 1]
    on_side = ((np.isclose(abs(x), 1) | np.isclose(abs(y), 1))
               | ((x == 0) & (y <= 0)) | ((y == 0) & (x >= 0)))
    neumann = (x == 0) & (y > -1) & (y < 0)
    return int(np.count_nonzero(~on_side | neumann))


def submission(mesh_or_vt, reported_error, history, dof=None):
    if isinstance(mesh_or_vt, Mesh):
        vertices, triangles = mesh_or_vt.vertices.tolist(), mesh_or_vt.triangles.tolist()
    else:
        vertices, triangles = mesh_or_vt
    return dict(vertices=vertices, triangles=triangles,
                dof=dirichlet_dof(vertices) if dof is None else dof,
                reported_error=float(reported_error), history=history)


def measured(mesh):
    return calibrate.measure(mesh)['relative_error']


def uniform_meshes():
    mesh, out = problem.initial_mesh(), []
    while calibrate.dofs(mesh) <= BUDGET:
        out.append(mesh)
        mesh = mesh.uniform_refine()
    return out


def history_of(meshes):
    return [dict(iter=k, dof=calibrate.dofs(m), estimator=measured(m)) for k, m in enumerate(meshes)]


def graded(indicators):
    """Dörfler(0.4) + NVB on the given indicators until the next mesh exceeds the budget."""
    meshes = [problem.initial_mesh()]
    while True:
        candidate = afem.newest_vertex_bisection(meshes[-1], afem.dorfler(indicators(meshes[-1]), .4))
        if calibrate.dofs(candidate) > BUDGET:
            return meshes
        meshes.append(candidate)


def last_records(history, limit=14):
    return [dict(r, iter=k) for k, r in enumerate(history[-limit:])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}

    meshes = uniform_meshes()
    files['uniform_mesh'] = submission(meshes[-1], measured(meshes[-1]), history_of(meshes))
    coarse = meshes[2]
    files['fake_error'] = submission(coarse, 0.04, history_of(meshes[:3]))

    for name, ind in (('corner_only_graded', calibrate.corner_indicators),
                      ('peak_ignored', calibrate.peak_blind_indicators)):
        seq = graded(ind)
        rows = [dict(iter=k, dof=calibrate.dofs(m), estimator=float(np.sqrt(ind(m).sum())))
                for k, m in enumerate(seq)]
        files[name] = submission(seq[-1], measured(seq[-1]), last_records(rows))

    # Red-refine triangle 0 of the level-2 mesh only: three hanging midpoints.
    v, t = coarse.vertices.tolist(), coarse.triangles.tolist()
    a, b, c = t[0]
    mids = []
    for p, q in ((a, b), (b, c), (c, a)):
        v.append(((np.array(v[p]) + v[q])/2).tolist())
        mids.append(len(v) - 1)
    ab, bc, ca = mids
    t[0:1] = [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
    files['hanging_nodes'] = submission((v, t), measured(coarse),
                                        [dict(iter=0, dof=dirichlet_dof(v), estimator=1.0)])

    # Full square [-1,1]^2: four unit squares, two CCW triangles each, refined twice.
    sq = Mesh([[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]],
              [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]])
    sq = sq.uniform_refine().uniform_refine()
    files['wrong_domain'] = submission(sq, 0.04, [dict(iter=0, dof=dirichlet_dof(sq.vertices), estimator=1.0)])

    with contextlib.redirect_stdout(io.StringIO()):
        mesh, _, hist = afem.run(problem.initial_mesh(), tol=0.05, budget=10**6, N_max=200)
    hist = [dict(r) for r in hist]
    files['over_budget'] = submission(mesh, hist[-1]['reported_error'], last_records(hist))

    for name, data in files.items():
        (OUT/f'{name}.json').write_text(json.dumps(data) + '\n', encoding='utf-8')
        print(f"{name:20s} vertices {len(data['vertices']):6d}  dof {data['dof']:6d}  "
              f"reported_error {data['reported_error']:.4f}  history {len(data['history'])}")


if __name__ == '__main__':
    main()
