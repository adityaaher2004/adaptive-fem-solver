# Reference adaptive solver

`solve.sh` runs `afem.py` and writes `/app/output/submission.json`. It is the
oracle for the task: in the task container the provided `fem` package is in
`/app`; in the authoring checkout `afem.py` finds it in `environment/data`.

From the task root (authoring checkout):

```sh
bash solution/solve.sh --output /tmp/submission.json   # extra args pass through to afem.py
python -m pytest -q solution                            # 21 tests, ~6 s
```

The stopping limits in `solve.sh` (tol 0.05, budget 10000, N_max 54) are the
task's limits from `specs.txt`, selected by `authoring/provenance/calibrate.py`.
`--mesh initial_mesh.json` optionally loads the starting mesh; otherwise
`problem.initial_mesh()` supplies the same mesh and NVB labels.

## Algorithm

Each iteration assembles the P1 system, applies the Dirichlet values, solves,
evaluates the true energy error, computes residual indicators, marks, and
refines with newest-vertex bisection (NVB). The squared indicator of each
triangle is the sum of:

- h_T^2 times integral_T f^2, with h_T the longest edge (Delta u_h = 0 for P1).
- Half of h_E times integral_E [grad u_h . n]^2 for each interior edge.
- h_E times integral_E (g_N - grad u_h . n)^2 for each Neumann edge.

Interior normals can have either sign; Neumann normals are made outward using
the adjacent triangle's centroid. Dirichlet edges contribute nothing (no
Dirichlet-data oscillation term). Dörfler marking takes the smallest sorted
set carrying theta = 0.4 of the total squared indicator.

NVB: local vertex 0 is the newest vertex and edge (1,2) the refinement edge.
Neighbours are bisected first when their refinement edge differs, so the shared
edge gets one midpoint. Children (midpoint, v0, v1) and (midpoint, v2, v0) keep
CCW orientation and the labelling. No uniform-refinement fallback is used.

Volume loads and element indicators use a composite degree-8 rule with
integration subcells of diameter <= 0.04, so the alpha = 1000 peak is resolved
even on the 6-triangle starting mesh. Neumann integrals use Gauss-Legendre.

## Stopping

DOF counts free vertices (not on Gamma_D, including vertices inside Gamma_N).
N_max counts solves, including the first. The solver stops when:

1. the true relative energy error <= tol;
2. the DOF count reaches the budget, or refining the marked set (with closure)
   would exceed it, in which case the candidate mesh is discarded;
3. N_max solves have been done;
4. the indicator is zero before tol is met.

The output always describes the last solved mesh within budget, and
`history[-1]['stop_reason']` records which rule applied. A zero exit status
means a submission was written, not that tol was met. A starting mesh over
budget is rejected.

This is the authoring reference: the exact global energy error (`fem.error`)
is used for stopping and for `reported_error`. Exact local errors are never
used for marking.

## Tests (`test_afem.py`)

- Dörfler: minimal prefix, theta = 1, all-zero indicators, invalid input.
- NVB: closure on the starting mesh, empty marking, invalid indices, and 12
  rounds of random marking checked for hanging nodes (geometric test), area,
  bisection of every marked triangle, 45° minimum angle, and boundary layout.
- Estimator: jump and Neumann terms against a loop-based reference built on
  `fem.basis`; zero indicator for a linear solution with matching Neumann data
  (catches a wrongly oriented normal); element term for constant f; peak
  quadrature unchanged when integration_h is halved.
- Stopping: tol, budget, N_max, and a starting mesh over budget.
- End to end: `main()` writes a submission that loads, has the required history
  keys, matches the recomputed DOF count, and has no hanging nodes.

Calibration runs and their provenance are in `authoring/evidence/calibration.md`.
