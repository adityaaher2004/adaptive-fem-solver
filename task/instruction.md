Your task is to complete an adaptive FEM solver for an elliptic PDE by producing a FEM mesh that meets specific constraints.

`/app/specs.txt` outlines the full problem details.

You submit a mesh at `/app/output/submission.json`

```json
{
  "vertices":  [[x, y], ...],
  "triangles": [[i, j, k], ...],
  "dof": 1234,
  "reported_error": 0.0481,
  "history": [{"iter": 0, "dof": 0, "estimator": 79.3}]
}
```

JSON Key guide:
- vertices: coordinates of mesh vertices.
- triangles: zero based and oriented counter clockwise.
- dof: the DOF count of the mesh.
- reported_error: the true relative energy error of your solution on the submitted mesh
- history: one record per SOLVE. Each record needs iter (integer ≥ 0), dof (integer ≥ 0) and estimator (your finite non negative error estimate at that step). 

JSON Requirements:
- It must contain a mesh to reassemble and solve the system which must have ≤ 2930 DOF
- It must contain the relative energy error of the submitted mesh which must be ≤ 0.05
- It must contain the DOF count of the mesh
- It must contain a history of at most 14 SOLVE steps counting the solve on the initial mesh.
- empty history is invalid
- boundary vertices must be exactly on domain sides
- all six corners must be vertices (corners being: (−1,−1), (0,−1), (0,0), (1,0), (1,1), (−1,1))
- the verifier allows at most 11736 vertices and 23472 triangles

Upon submission, the verifier only receives your JSON submission file. No other code is imported.

The verifier conducts validity and sanity checks on the submitted JSON, then reassembles the submitted mesh, solves the system and recomputes the relative energy error against the analytic exact solution. The verifier requires that your reported error must be within a factor of 2 of the recomputed one.

You have 14400 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.