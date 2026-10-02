# adaptive-fem-solver: build checklist

Work top to bottom. Each phase depends on the one before it.

## Phase 1: FEM implementation (agent-facing code + reference solution)

- [x] 1.1 `environment/data/fem/quadrature.py`: Dunavant triangle rules (degrees 1–8ish) and Gauss–Legendre edge rules; unit tests that integrate polynomials exactly
- [x] 1.2 `fem/basis.py`: P1 shape functions, reference gradients, affine map + Jacobian helpers
- [x] 1.3 `fem/mesh.py`: Mesh class (vertices, CCW triangles, edge list, edge→triangle map, boundary edges/vertices), area check, uniform red refinement
- [x] 1.4 `fem/problem.py`: per `specs.txt` — u = u_s + u_p with u_s = r^(1/3) sin(θ/3), u_p = A·exp(−α|x−x₀|²); grad u; f = 4α(1−α|x−x₀|²)u_p; g_D; g_N = 2α·x₀ₓ·u_p on x = 0 (≈1e-240 at the provisional x₀: effectively zero flux); θ = mod(atan2(y,x), 2π) so −0.0 maps to 0; boundary-edge classifier (Γ_N = open segment {x=0, −1<y<0}, everything else Γ_D). Open decision: expose u itself, or only f, g_D, g_N
- [x] 1.5 `fem/assembly.py`: vectorised stiffness + load assembly (scipy.sparse COO→CSR) with batched geometry; Neumann edge load via `edge_rule`; Dirichlet elimination on Γ_D vertices only
- [x] 1.6 `fem/solver.py`: spsolve wrapper returning the full nodal vector
- [x] 1.7 `fem/io.py`: load/save mesh JSON; write submission JSON (vertices, triangles, dof, reported_error, history)
- [x] 1.8 `authoring/provenance/make_initial_mesh.py` → `environment/data/initial_mesh.json` (coarse conforming L-shape with the lower-right quadrant removed, all six domain corners as vertices so Γ_N/Γ_D are unions of edges, refinement edges set for NVB) — done: generated from `problem.initial_mesh()` (single source); `--check` detects drift; NVB convention = local vertex 0 is newest, refinement edge is local (1,2), compatible diagonal labelling. Document the convention in `instruction.md` (3.2)
- [ ] 1.9 `fem/example_uniform.py`: uniform-refinement demo printing DOF and relative energy error; confirm the O(N^-1/6) rate caused by the r^(1/3) singularity
- [ ] 1.10 `solution/afem.py`: residual estimator (element residual h_T‖f‖, interior edge jumps, Neumann edge residual h_E^(1/2)‖g_N − ∂u_h/∂n‖), Dörfler marking (θ≈0.3–0.5), newest-vertex bisection with conforming closure, stops at tol / budget / N_max
- [ ] 1.11 Confirm the reference AFEM reaches the optimal O(N^-1/2) rate
- [ ] 1.12 `authoring/provenance/calibrate.py`: DOF–error curves for uniform, corner-only graded, and adaptive meshes → choose A, α, x₀, tolerance, DOF budget, N_max so that uniform and corner-only grading both clearly fail while the reference AFEM passes with margin; record in `authoring/evidence/calibration.md`
- [ ] 1.13 Fill TBDs in `environment/data/specs.txt` and drop the "provisional" note on the peak parameters
- [ ] 1.14 `solution/solve.sh`: run afem.py and write `/app/output/submission.json`

## Phase 2: Verifier (independent; must not import `fem/`)

- [ ] 2.1 `tests/verifier/mesh_checks.py`: schema/index checks, positive area, no duplicate vertices, conformity (each interior edge shared by exactly 2 triangles, no hanging nodes), total area = 3, no triangle centroid/vertex inside the cut-out, boundary vertices on ∂Ω, all six domain corners present as vertices
- [ ] 2.2 `tests/verifier/fem_ref.py`: independent P1 assembly + solve with g_D interpolated at Γ_D vertices and Neumann load on Γ_N edges
- [ ] 2.3 `tests/verifier/error.py`: relative energy error vs exact grad u; Duffy transform centred on the singular vertex (∇u ~ r^(−2/3)) plus graded subdivision for corner elements, high-order rules near the peak; compute ‖∇u‖² to ~1e-10 by two independent methods and record in `authoring/evidence/`; check that quadrature error ≪ tol
- [ ] 2.4 `tests/config.json`: sealed tolerance, DOF budget, max iter (same as specs)
- [ ] 2.5 `tests/test_submission.py`: one pytest per check (exists/parses, schema, mesh valid, DOF recomputed as vertices not on Γ_D, ≤ budget and equal to reported, len(history) ≤ N_max, computed error ≤ tol, reported_error within loose factor of computed)
- [ ] 2.6 `tests/test.sh`: run pytest with `--ctrf /logs/verifier/ctrf.json`; write exactly 1 or 0 to `/logs/verifier/reward.txt`; handle missing artifact
- [ ] 2.7 `tests/Dockerfile`: python:3.11-slim, pinned numpy/scipy, pytest==8.4.1, pytest-json-ctrf==0.3.5, COPY → /tests
- [ ] 2.8 Cross-check: verifier error matches `fem/` error on the oracle mesh to several digits

## Phase 3: Polishing and bundling

- [ ] 3.1 `environment/Dockerfile`: python:3.11-slim, pinned numpy/scipy, COPY data → /app, mkdir /app/output; no verifier files or thresholds beyond specs leak in
- [ ] 3.2 `instruction.md`: goal, file locations, mixed BCs (Γ_D/Γ_N), DOF definition, output path and exact JSON schema, pass criteria, tooling limits (numpy/scipy in the verifier)
- [ ] 3.3 `task.toml`: apply the provided template; artifacts = ["/app/output/submission.json"]; timeouts
- [ ] 3.4 Generate cheat attempts in `authoring/evidence/cheat_attempts/` (uniform, corner-only graded, peak-ignored, fake error, hanging nodes, wrong domain, over budget); confirm each scores 0
- [ ] 3.5 Oracle run: build both images, run solve.sh, verify reward = 1; record in `authoring/evidence/oracle_run.md`
- [ ] 3.6 Nop run: reward = 0; record in `authoring/evidence/nop_run.md`
- [ ] 3.7 Edge cases: clockwise triangles, float-noise vertices on ∂Ω, NaN/inf values, huge meshes (verifier time limit), extra JSON keys
- [ ] 3.8 Determinism: fixed parameters; oracle result reproducible across runs
- [ ] 3.9 `README.md`: Difficulty · Reference solution · Verification sections, with calibration numbers
- [ ] 3.10 Final sweep: no TODO stubs left, no leftover `_todo` JSON, delete this file or mark it complete, and check that `authoring/` is never COPY'd into either image
