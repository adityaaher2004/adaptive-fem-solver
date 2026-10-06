# Reference adaptive solver

`solve.sh` runs `afem.py` and writes `/app/output/submission.json`. It is the
oracle for the task: in the task container the provided `fem` package is in
`/app`; in the authoring checkout `afem.py` finds it in `environment/data`.

From the task root (authoring checkout):

```sh
bash solution/solve.sh --output /tmp/submission.json   # extra args pass through to afem.py
python -m pytest -q solution
```

Defaults for `--tol`, `--budget` and `--N-max` are read from `specs.txt`
(tol 0.05, budget 3500, N_max 50); `solve.sh` also passes them explicitly.
`--mesh initial_mesh.json` optionally loads the starting mesh; otherwise
`problem.initial_mesh()` supplies the same mesh and NVB labels.

Result: 44 solves, 3073 DOF, relative energy error 0.0488 (stop reason `tol`).

## Algorithm

The task allows any marking strategy; this reference marks with the exact
error. Each iteration:

1. assembles and solves the P1 system. The volume load uses a composite
   degree-8 rule with integration subcells of diameter <= 0.04, so the
   alpha = 1000 peak is resolved even on the 6-triangle starting mesh;
2. computes the elementwise energy error eta_T^2 = int_T |grad u - grad u_h|^2
   against `fem.problem.grad_u`, with the corner t^3 Duffy map
   (`local_energy_error`, a per-element copy of `fem.error.energy_error`);
3. marks with Dörfler, theta = 0.35 (`THETA`). Near the tolerance
   (error < 1.3 tol) the marked fraction shrinks to the share of squared error
   that still has to go, `max(0.06, 1.5 (1 - (0.999 tol / err)^2))`, which
   reaches the tolerance with the fewest extra DOF;
4. refines with newest-vertex bisection: local vertex 0 is the newest vertex
   and edge (1,2) the refinement edge; neighbours are bisected first when their
   refinement edge differs, so shared edges get one midpoint and no hanging
   nodes appear.

Refinement stops at error <= 0.99 tol (`TARGET_RATIO`), about 1e-3 below the
tolerance. The calibration theta sweep (`authoring/evidence/calibration.md`)
shows which theta values meet the limits.

Optional coarsening (`--excess`, default 0): refinement may temporarily exceed
the budget by that fraction; interior vertices with the lowest incident error
are then removed (disjoint stars, cavities re-triangulated by ear clipping) and
the mesh is re-solved; the exact error decides acceptance.
`--reproduce-submission-5` (budget 2992, excess 0.02) reproduces authoring's
submission_5 mesh exactly from a cold start.

## Stopping

DOF counts free vertices (not on Gamma_D, including vertices inside Gamma_N).
N_max counts solves, including the first and any coarsening solve. The solver
stops with `history[-1]['stop_reason']`:

1. `tol`: error <= 0.99 tol;
2. `budget`: the next refinement (or a needed coarsening) would leave the DOF
   budget; the candidate mesh is discarded;
3. `N_max`: the solve limit is reached;
4. `zero_estimator`: nothing left to mark.

It never raises mid-run: the output is always the last solved mesh within
budget, marked `submitted: true` in the history. A zero exit status means a
submission was written, not that tol was met. A starting mesh over budget is
rejected with `ValueError`.

## Tests (`test_afem.py`)

- Dörfler: minimal prefix, theta = 1, all-zero indicators, invalid input.
- NVB: closure on the starting mesh, empty marking, invalid indices, and 12
  rounds of random marking checked for hanging nodes (geometric test), area,
  bisection of every marked triangle, 45° minimum angle, and boundary layout.
- Peak load quadrature unchanged when integration_h is halved.
- Stopping: tol, budget, N_max, and a starting mesh over budget.
- End to end: `main()` writes a submission that loads, has the required history
  keys, matches the recomputed DOF count, and has no hanging nodes.
- Task limits agree across selected_config.json, specs.txt, tests/config.json
  and solve.sh.

The residual estimator used by the calibration's peak-blind baseline lives in
`authoring/provenance/calibrate.py` (tests in `authoring/tests/test_baselines.py`).
Calibration runs and their provenance are in `authoring/evidence/calibration.md`.
