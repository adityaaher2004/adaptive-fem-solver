# Adaptive FEM Solver

Your task is to build an adaptive FEM solver for an elliptic PDE and use it to produce a FEM mesh that meets specific constraints. You will be given a FEM codebase to work with. You must find where your current approximation is inaccurate, refine the mesh there, and repeat until the requirements are met. You must submit the mesh as a JSON.

In brief, the problem is the Poisson equation on an L-shaped domain with mixed boundary conditions. The error norm is relative energy error.

Your submission must contain the following:
- A mesh to reassemble and solve the system
- The relative energy error of the submitted mesh
- The DOF count of the mesh
- A history of SOLVE steps with one record per SOLVE

Your submission must meet the following constraints:
- the mesh must have **≤ 10000** DOF
- the submitted mesh must have error **≤ 0.05** (error is recomputed by verifier)
- **≤ 54** SOLVE steps, counting the solve on the initial mesh, so it may hold at most 54 records
- empty history is invalid
- the verifier will reassemble the mesh submitted by you and recompute the error. The relative energy error submitted by you must be within a factor of 2 of the verifier recomputed error.

You are given the following:

`/app/specs.txt` -> The full problem details.
`/app/fem/` -> Working FEM P1 utils, JSON I/O utility, and unit tests.
`/app/initial_mesh.json` -> Initial mesh
`/app/fem/example_uniform.py` -> Runnable demo (`cd /app && python -m fem.example_uniform`)

You may change, extend or replace any of this code and install any tools. Only the submission file is collected; nothing else you write survives.

You submit `/app/output/submission.json`

```json
{
  "vertices":  [[x, y], ...],
  "triangles": [[i, j, k], ...],
  "dof": 1234,
  "reported_error": 0.0121,
  "history": [
    {"iter": 0, "dof": 0, "estimator": 9.3},
    {"iter": 1, "dof": 1, "estimator": 6.1}
  ]
}
```

- `vertices`: coordinates of mesh vertices.
- `triangles`: zero based vertex indices, oriented counter clockwise.
- `dof`: the DOF count of the mesh.
- `reported_error`: the true relative energy error of your solution on the submitted mesh. This can be computed against the exact solution with `fem.error.energy_error`
- `history`: one record per SOLVE step, in order. Each record needs `iter` (integer ≥ 0), `dof` (integer ≥ 0) and `estimator` (your finite non negative error estimate at that step). Extra keys are allowed. At least one record and at most 54.

`fem.io.write_submission(path, mesh, dof=..., reported_error=..., history=...)` writes this format and checks the types; `len(system.free_vertices)` from `fem.assembly.assemble_system(mesh)` is the DOF count.

Some notes on mesh submission:
- triangles must be counter clockwise
- boundary vertices must be exactly on domain sides
- all six corners must be vertices (corners being: (−1,−1), (0,−1), (0,0), (1,0), (1,1), (−1,1))
- the verifier allows at most 40016 vertices and 80032 triangles

Upon submission, the verifier only receives your JSON submission file. No other code is imported. The verifier does not import the fem codebase.

Using this submission, it checks the following:
- parse JSON for validity and sanity checks
- DOF count meets constraints
- solve history count meets constraints
- check that the mesh is a valid conforming triangulation, with additional sanity checks

The verifier then reassembles the mesh and solves the system and the relative energy error is computed against the analytic exact solution. The verifier requires that your reported error must be within a factor of 2 of the recomputed one.

You have 14400 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.