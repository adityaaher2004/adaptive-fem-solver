"""Uniform-refinement calibration demo; run python -m fem.example_uniform.

DOF counts unconstrained P1 unknowns. Level zero has no free unknowns, so
its rate is undefined. Error quadrature is independent of load quadrature.
For u_s=r**(1/3) sin(theta/3), uniform h gives energy error O(h**(1/3));
N~h**(-2) therefore gives O(N**(-1/6)). The Gaussian delays this asymptotic
regime. Use --singular-only to isolate the corner rate, and rerun with higher
--error-order / --load-refinements to check integration sensitivity.
"""
import argparse
import math
from . import problem
from .assembly import assemble_system
from .solver import solve
from .error import energy_error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels',type=int,default=7,help='maximum refinement level (inclusive)')
    parser.add_argument('--singular-only',action='store_true')
    parser.add_argument('--error-order',type=int,default=12)
    parser.add_argument('--error-cell-diameter',type=float,default=.125)
    parser.add_argument('--load-refinements',type=int,default=5,
                        help='integration subdivisions on level zero; decreases with mesh level')
    args = parser.parse_args(argv)
    if args.levels < 0 or args.load_refinements < 0:
        parser.error('levels and load-refinements must be nonnegative')
    p = problem.PoissonProblem(A=0) if args.singular_only else problem.DEFAULT_PROBLEM
    mesh = problem.initial_mesh()
    previous = None
    print('case:', 'singular only' if args.singular_only else 'corner + Gaussian')
    print('level vertices triangles dof relative_energy_error rate_N')
    for level in range(args.levels+1):
        system = assemble_system(mesh,p.f,p.g_D,p.g_N,
                                 load_refinements=0 if args.singular_only else max(0,args.load_refinements-level))
        u = solve(system)
        error = energy_error(mesh,u,p.grad_u,order=args.error_order,
                             max_cell_diameter=args.error_cell_diameter).relative
        dof = len(system.free_vertices)
        rate = None if previous is None or dof == 0 else math.log(previous[1]/error)/math.log(dof/previous[0])
        print(f'{level:2d} {mesh.n_vertices:8d} {mesh.n_triangles:9d} {dof:8d} {error:.10e} '+
              ('   --' if rate is None else f'{rate:.6f}'),flush=True)
        if dof:
            previous = dof,error
        if level < args.levels:
            mesh = mesh.uniform_refine()
    print('Expected asymptotic rate_N: 1/6 = 0.166667')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
