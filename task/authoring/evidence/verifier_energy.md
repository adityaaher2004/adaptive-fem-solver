# Independent verifier energy integration

Reproduce from task root:

```sh
python authoring/provenance/check_verifier_energy.py
python -m pytest -q authoring/tests/test_verifier_error.py
```

All verifier imports stay within `verifier/` and NumPy/SciPy/standard library.
The oracle nodal vector is recomputed with `verifier.fem_ref.solve`.

| Method/settings | Reference energy squared | Oracle relative energy error |
|---|---:|---:|
| Direct, order 16, diameter 0.125, 4 corner panels | 1.6325361700004277 | 0.04883766451166076 |
| Direct, order 24, diameter 0.0625, 6 corner panels | 1.632536170000428 | 0.04883766451166085 |
| Green identity, adaptive tolerance 2e-12 per subintegral | 1.6325361700004422 | — |
| Green identity, adaptive tolerance 5e-13 per subintegral | 1.6325361700004422 | — |

The independent-method discrepancy is about 1.4e-14, below the requested
1e-10. The supplied target 1.632536170000 is its rounded value.
The observed oracle relative-error change is about 9e-17, negligible against
the 0.05 pass threshold and the oracle's approximately 1.16e-03 margin
(oracle submission: 3073 DOF, 44 solves).
These are empirical convergence checks, not rigorous interval error bounds.

## Direct method

Integrate both squared norms directly on the submitted mesh. Integration-only
red subdivision resolves the Gaussian without changing FEM functions. For a
triangle with origin vertex and remaining vectors a,b, map
x=t^3*((1-s)*a+s*b). Its Jacobian is 3*t^5*abs(det(a,b)). The squared singular
gradient scales like t^-4, leaving an O(t) integrand. Geometric panels in t
concentrate quadrature near the singular vertex. Other cells use ordinary
Duffy quadrature. Gauss rules never evaluate the origin. Full gradient sums
and differences are squared, so all cross terms and Dirichlet interpolation
error are retained.

## Independent method

Green's identity uses integral_boundary u*(grad u dot n) + integral_domain u*f.
It uses adaptive QUADPACK integration on the six boundary sides and nested
adaptive integration over three fixed squares. Gaussian-scale breakpoints
avoid missing localized forcing. This route does not use submitted mesh
triangles, finite-element gradients, or the direct volume quadrature rule.
It shares analytic PDE data with the direct method. A singular-only identity
check and finite-difference gradient test provide additional checks on that data.

The default order was raised from 12 to 16 after a coarse-mesh test detected
an approximately 1.8e-10 discrepancy at order 12. Tests cover the initial mesh
as well as the oracle used for the evidence above. Phase 2 submission tests
should tighten integration near a pass/fail threshold rather than assume
these observed errors bound all possible submitted meshes.
