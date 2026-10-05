# adaptive-fem-solver

The agent is given a FEM codebase for solving elliptic PDEs. The agent's task is to design an adaptive FEM solver by constructing a mesh subject to DOF constraints and meeting an error tolerance.

The agent must produce the submission as a JSON in `/app/output/submission.json`. The submission must contain the final mesh, its DOF count, and relative energy error, and a history with a record for every solve. Specifically, the submission must be constrained to:
- 0.05 maximum relative energy error
- 2930 DOF at most
- 14 solves at most

## Difficulty

The problem is synthetic and has an exact closed-form solution. It is the Poisson problem on an L-shaped domain with Neumann data on one re-entrant edge and Dirichlet data elsewhere. Its exact solution is a corner singularity r^(1/3) sin(theta/3) plus a narrow Gaussian peak at (-0.43, 0.61). The Neumann edge strengthens the corner singularity to r^(1/3), so uniform refinement converges only as N^(-1/6). The interior peak defeats grading toward the corner alone, and it is so narrow (about 0.03) that low-order quadrature on coarse triangles misses it, so a naive estimator never refines there.

At the 2930-DOF budget, the obvious shortcuts all fail by a wide margin:

| Strategy | DOF | Relative energy error |
|----------|-----|-----------------------|
| Uniform refinement | 720 | 0.601 (12.0x tol)|
| Grading toward the corner only | 2436 | 0.525 (10.5x tol) |
| Adaptive, blind to the peak (f = 0) | 2744 | 0.485 (9.7x tol) |
| Reference solver | 2868 | 0.0499 (pass) |

The agent needs non-trivial elements to succeed, namely:
- residual estimator with element, Neumann and jump terms
- peak resolving quadrature
- conforming local refinement with no hanging nodes
- efficient marking to meet DOF and solve budget

While Poisson on the L shaped domain appears in standard adaptivity problem demonstration, this particular problem is novel and hard because it involves understanding the corner singularity , why uniform refinement fails to meet the budget, and a conforming adaptive refinement loop that keeps DOF ≤ 2930 while hitting 0.05 error. This requires real numerical PDE expertise. The implementation is substantive and not a canned library call or using external tools such as AFEM@matlab, iFEM, FEniCS/skfem adaptive demos. The approach to meeting the error tolerance and adhering to the budget needs to be reasoned out rather than putting in busywork and recalling already available libraries. 

The quintessence of the problem lies in truly understanding how mesh refinement works and the agent's ability to demonstrate just that. The agent must work on its own to develop a working valid mesh.  In the real world a numerical analyst validating an adaptive solver performs this task.

## Reference solution

`solve.sh` runs `afem.py`. It is built on the provided package. At every solve, it computes residual indicators. It uses composite quadrature, Dorfler marking, NVB, and recursive conformity closure.

Solver stopping specs:
- 8 solves
- 2868 DOF
- 0.0499 error

## Verification

The verifier is robust on correctness. It never imports the agent's code or the provided FEM codebase. It only sees the agent's submission. Using this submission, it checks the following:
- parse JSON for validity and sanity checks
- DOF count meets constraints
- solve count meets constraints
- check the submitted mesh for validity and sanity checks

The verifier then reassembles the mesh and solves the system and the relative energy error is computed against the analytic exact solution. The verifier requires that the reported error must be within a factor of 2 of the recomputed one. The verifier applies the check on the reported to error to catch cheating attempts at reporting false passable values.

The verifier's ‖∇u‖² matches an independent Green's identity computation to 1.4e-14, matches the oracle error with the provided codebase to 6e-17, and is fail closed.

The oracle (`solution/solve.sh`) scores 1 and the nop run scores 0. Seven cheat submissions (`authoring/evidence/cheat_attempts/`) and 38 edge cases (`authoring/evidence/edge_cases.md`) score as expected, and both the oracle and the verifier are deterministic (`authoring/evidence/determinism.md`).
