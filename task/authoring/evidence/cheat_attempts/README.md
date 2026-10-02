# Cheat attempts

Each submission below must score 0.

| File | Why it fails |
|---|---|
| uniform_mesh.json | Honest uniform mesh: over DOF budget or above tolerance |
| corner_only_graded.json | A priori grading toward the origin only; under-resolves the interior peak |
| peak_ignored.json | Adaptive mesh for the corner singularity alone (f = 0); peak error exceeds tolerance |
| fake_error.json | Coarse mesh with fabricated low reported error |
| hanging_nodes.json | Nonconforming refinement |
| wrong_domain.json | Full square / missing cut-out |
| over_budget.json | Meets tolerance but exceeds DOF budget |
