# adaptive-fem-solver

The agent is given a FEM codebase for solving elliptic PDEs. The agent's task is to design an adaptive FEM solver by constructing a mesh subject to DOF constraints and meeting an error tolerance.

The agent must produce the submission as a JSON in `/app/output/submission.json`. The submission must contain the final mesh, its DOF count, and relative energy error, and a history with a record for every solve. Specifically, the submission must be constrained to:
- 0.05 maximum relative energy error
- 10,000 DOF at most
- 54 solves at most

## Difficulty

The problem is synthetic and has an exact closed-form solution. It is the Poisson problem on an L-shaped domain with Neumann data on one re-entrant edge and Dirichlet data elsewhere. Its exact solution is a corner singularity r^(1/3) sin(theta/3) plus a narrow Gaussian peak at (-0.43, 0.61). The Neumann edge strengthens the corner singularity to r^(1/3), so uniform refinement converges only as N^(-1/6). The interior peak defeats grading toward the corner alone, and it is so narrow (about 0.03) that low-order quadrature on coarse triangles misses it, so a naive estimator never refines there.

At the 10000-DOF budget, the obvious shortcuts all fail by a wide margin:

| Strategy | DOF | Relative energy error |
|----------|-----|-----------------------|
| Uniform refinement | 2976 | 0.431 (8.6x tol)|
| Grading toward the corner only | 9844 | 0.385 (7.7x tol) |
| Adaptive, blind to the peak (f = 0) | 9097 | 0.385 (7.7x tol) |
| Reference solver | 5383 | 0.049 (pass) |

The agent needs non-trivial elements to succeed, namely:
- residual estimator with element, Neumann and jump terms
- peak resolving quadrature
- conforming local refinement with no hanging nodes
- efficient marking to meet DOF and solve budget

## Reference solution

`solve.sh` runs `afem.py`. It is built on the provided package. At every solve, it computes residual indicators. It uses composite quadrature, Dorfler marking, NVB, and recursive conformity closure.

Solver stopping specs:
- 36 solves
- 5383 DOF
- 0.0490 error

## Verification

The verifier is robust on correctness. It never imports the agent's code or the provided FEM codebase. It only sees the agent's submission. Using this submission, it checks the following:
- parse JSON for validity and sanity checks
- DOF count meets constraints
- solve count meets constraints
- check the submitted mesh for validity and sanity checks

The verifier then reassembles the mesh and solves the system and the relative energy error is computed against the analytic exact solution. The verifier requires that the reported error must be within a factor of 2 of the recomputed one.

The verifier's ‖∇u‖² matches an independent Green's identity computation to 1.4e-14, matches the oracle error with the provided codebase to 6e-17, and is fail closed.

The oracle (`solution/solve.sh`) scores 1 and the nop run scores 0. Seven cheat submissions (`authoring/evidence/cheat_attempts/`) and 38 edge cases (`authoring/evidence/edge_cases.md`) score as expected, and both the oracle and the verifier are deterministic (`authoring/evidence/determinism.md`).
