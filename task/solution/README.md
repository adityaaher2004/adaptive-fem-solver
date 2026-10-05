# Reference adaptive solver

`solve.sh` runs `afem.py` and writes `/app/output/submission.json`. It is the
oracle for the task: in the task container the provided `fem` package is in
`/app`; in the authoring checkout `afem.py` finds it in `environment/data`.

From the task root (authoring checkout):

```sh
bash solution/solve.sh --output /tmp/submission.json   # extra args pass through to afem.py
```

The stopping limits in `solve.sh` (tol 0.05, budget 2930, N_max 14) are the
task's limits from `specs.txt`, selected by `authoring/provenance/calibrate.py`.
`--mesh initial_mesh.json` optionally loads the starting mesh; otherwise
`problem.initial_mesh()` supplies the same mesh and NVB labels.

## Algorithm (`--strategy targeted`, the default)

1. **A priori mesh sequence, no solves.** Starting from the 6-triangle mesh,
   repeatedly compute each triangle's exact interpolation error
   ||grad(u - I_h u)||_T^2 (the exact solution is known), mark with Dörfler
   (theta = 0.1, then theta = 0.02 once the DOF reach 2400, for fine steps near
   the budget) and refine with newest-vertex bisection (NVB). Stop when the
   next mesh would exceed the DOF budget. This equidistributes the error, which
   concentrates about half the triangles around the narrow peak and grades
   strongly toward the corner.
2. **Binary search with solves.** Solve the P1 problem on meshes of that nested
   sequence, bisecting on the index, to find the smallest mesh whose true
   relative energy error is <= tol. Each solve is one history record. If the
   last solve was not on the chosen mesh, it is solved once more so that
   `history[-1]` describes the submitted mesh.

Result: 8 solves, 2868 DOF, relative error 0.04995 (about 5 minutes, almost all
of it building the sequence).

Interpolation errors use Duffy rules on integration-only red subcells; subcells
touching the origin use the map x = t^3((1-s)a + s b), which removes the
r^(-2/3) gradient singularity. Volume loads use composite degree-8 quadrature
with subcells of diameter <= 0.04, so the alpha = 1000 peak is resolved.

NVB: local vertex 0 is the newest vertex and edge (1,2) the refinement edge.
Neighbours are bisected first when their refinement edge differs, so the shared
edge gets one midpoint. Children (midpoint, v0, v1) and (midpoint, v2, v0) keep
CCW orientation and the labelling.

## Residual AFEM (`--strategy residual`)

The classic SOLVE-ESTIMATE-MARK-REFINE loop is kept as a calibration baseline:
residual indicators (h_T^2 ||f||^2, half of each interior edge's
h_E ||[grad u_h . n]||^2, and the Neumann flux residual), Dörfler marking and
NVB. At the current limits it cannot pass: it needs about 3600-5400 DOF and
36-50 solves to reach the tolerance (see `authoring/evidence/calibration.md`).

## Stopping and output

DOF counts free vertices (not on Gamma_D, including vertices inside Gamma_N).
N_max counts solves (history records). `history[-1]['stop_reason']` is `tol`
when the submitted mesh meets the tolerance and `budget` otherwise. A zero exit
status means a submission was written, not that tol was met.

This is the authoring reference: it uses the exact solution both to build the
mesh (interpolation error) and to stop (true energy error, also written as
`reported_error`).

Calibration runs and their provenance are in `authoring/evidence/calibration.md`.
