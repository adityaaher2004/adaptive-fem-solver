"""Calibrate tolerance, DOF budget and N_max for the adaptive-FEM task.

Run from anywhere: python authoring/provenance/calibrate.py

Measures the reference AFEM (solution/afem.py) against three baselines, all
scored with the full corner + Gaussian problem:

  uniform       uniform red refinement
  corner        corner-only geometric grading (never looks at f)
  peak_ignored  residual AFEM that solves and marks with f = 0, i.e. an
                estimator blind to the interior peak

and a Dörfler theta sweep of the reference AFEM under the selected limits.
Also computes ||grad u||^2 independently by Green's identity.

Outputs (paths overridable):
  authoring/provenance/selected_config.json      selected limits + peak parameters
  authoring/evidence/calibration/measurements.json  all measurements
  authoring/evidence/calibration.md              human-readable report

Never edits specs.txt or tests/config.json; copying the values there is TODO
1.13. A failed calibration still writes the evidence, removes any stale
selected_config.json, and exits 1.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
for folder in (ROOT/'environment/data',ROOT/'data'):
    if (folder/'fem').is_dir():
        sys.path.insert(0,str(folder)); break
import numpy as np
from fem import problem
from fem.assembly import assemble_load,assemble_stiffness,eliminate_dirichlet,boundary_masks
from fem.solver import solve
from fem.error import energy_error
spec=importlib.util.spec_from_file_location('reference_afem',ROOT/'solution/afem.py')
afem=importlib.util.module_from_spec(spec)
spec.loader.exec_module(afem)

THETAS=(0.2,0.3,0.4,0.5,0.6,0.8)


def dofs(mesh):
    return int(np.count_nonzero(~boundary_masks(mesh)[0]))


def measure(mesh,integration_h=.04,error_order=12):
    load,_=afem.volume_terms(mesh,problem.f,integration_h)
    load+=assemble_load(mesh,lambda p:0.,problem.g_N)
    system=eliminate_dirichlet(mesh,assemble_stiffness(mesh),load)
    u=solve(system)
    error=energy_error(mesh,u,order=error_order).relative
    return dict(dof=len(system.free_vertices),vertices=mesh.n_vertices,
                triangles=mesh.n_triangles,relative_error=error)


def corner_indicators(mesh):
    """Purely geometric corner grading; never inspects forcing or full error.

    h^2 |T| (r+h)^(-10/3) approximates squared interpolation energy for
    |D^2 u_s|~r^(-5/3). Equidistribution gives h~r^(5/6), regularized near
    zero. NVB closure keeps cells shape regular. This is one reproducible
    corner-only baseline, not a proof over every possible graded mesh.
    """
    v=mesh.vertices[mesh.triangles]
    h=np.linalg.norm(v[:,[1,2,0]]-v[:,[0,1,2]],axis=2).max(axis=1)
    r=np.linalg.norm(v.mean(axis=1),axis=1)
    return mesh.areas*h*h/(r+h)**(10/3)


def peak_blind_indicators(mesh):
    """Residual indicators of the f = 0 problem: the peak is invisible."""
    load=assemble_load(mesh,lambda p:0.,problem.g_N)
    u=solve(eliminate_dirichlet(mesh,assemble_stiffness(mesh),load))
    return afem.residual_estimator(mesh,u,np.zeros(mesh.n_triangles))


def baseline(kind,budget,max_steps,integration_h,error_order):
    """Refine until the next mesh exceeds budget; that mesh is also measured
    (flagged over_budget) to show the baseline still fails beyond the budget."""
    mesh=problem.initial_mesh()
    rows=[]
    for step in range(max_steps):
        row=measure(mesh,integration_h,error_order)
        row.update(step=step,over_budget=False)
        rows.append(row)
        print(kind,row,flush=True)
        if kind=='uniform':
            candidate=mesh.uniform_refine()
        else:
            eta=corner_indicators(mesh) if kind=='corner' else peak_blind_indicators(mesh)
            candidate=afem.newest_vertex_bisection(mesh,afem.dorfler(eta,.4))
        if dofs(candidate)>budget:
            row=measure(candidate,integration_h,error_order)
            row.update(step=step+1,over_budget=True)
            rows.append(row)
            return rows,dict(reason='budget',next_dof=row['dof'])
        mesh=candidate
    return rows,dict(reason='step_limit')


def choose(tol,budgets,oracle,baselines,min_margin=1.5,baseline_margin=2.,n_max=None):
    """Smallest budget with budget >= min_margin * oracle DOF at which every
    baseline's best within-budget error is >= baseline_margin * tol."""
    successful=[r for r in oracle if r['reported_error']<=tol]
    if not successful:
        return None
    oracle_dof=successful[0]['dof']
    for budget in sorted(budgets):
        best=[min((r['relative_error'] for r in rows if r['dof']<=budget),default=None)
              for rows in baselines.values()]
        if (budget>=min_margin*oracle_dof and all(b is not None for b in best)
                and min(best)>=baseline_margin*tol):
            return dict(tol=tol,dof_budget=budget,
                        N_max=n_max if n_max else max(1,math.ceil(1.5*len(oracle))))
    return None


def green_energy(n=200):
    """||grad u||^2 = oint u du/dn + int u f, independent of 2D singular quadrature.

    u_s = 0 on the theta=0 edge and du_s/dn = 0 on Gamma_N, so only the outer
    sides contribute; f is below e^-90 outside x0 +/- 0.3.
    """
    pb=problem.DEFAULT_PROBLEM
    z,w=np.polynomial.legendre.leggauss(n)
    total=0.
    for a,b,normal in [((-1,-1),(0,-1),(0,-1)),((-1,1),(-1,-1),(-1,0)),((1,1),(-1,1),(0,1)),((1,0),(1,1),(1,0))]:
        a,b,normal=map(np.array,(a,b,normal))
        for lo,hi in [(0,.5),(.5,1)]:
            t=lo+(hi-lo)*(z+1)/2; pts=a+t[:,None]*(b-a)
            total+=np.linalg.norm(b-a)*(hi-lo)/2*np.sum(w*pb.u(pts)*(pb.grad_u(pts)@normal))
    x0=np.array(pb.x0)
    X,Y=np.meshgrid(x0[0]+.3*z,x0[1]+.3*z,indexing='ij')
    pts=np.stack((X,Y),-1)
    return total+.09*np.sum(np.outer(w,w)*pb.u(pts)*pb.f(pts))


def theta_sweep(selected,integration_h,error_order):
    rows=[]
    for theta in THETAS:
        with contextlib.redirect_stdout(io.StringIO()):
            _,_,h=afem.run(problem.initial_mesh(),tol=selected['tol'],budget=selected['dof_budget'],
                           N_max=10*selected['N_max'],theta=theta,integration_h=integration_h,error_order=error_order)
        last=h[-1]
        passed=last['reported_error']<=selected['tol'] and len(h)<=selected['N_max']
        rows.append(dict(theta=theta,solves=len(h),dof=last['dof'],relative_error=last['reported_error'],
                         stop_reason=last['stop_reason'],passes=passed))
        print('theta',rows[-1],flush=True)
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--tol',type=float,default=.05)
    p.add_argument('--budgets',type=int,nargs='+',default=[2930])
    p.add_argument('--max-solves',type=int,default=100)
    p.add_argument('--baseline-steps',type=int,default=100)
    p.add_argument('--baseline-margin',type=float,default=2.)
    p.add_argument('--min-margin',type=float,default=1.5,help='required budget / oracle DOF')
    p.add_argument('--n-max',type=int,default=None,help='fixed N_max (default: 1.5 x oracle solves)')
    p.add_argument('--integration-h',type=float,default=.04)
    p.add_argument('--error-order',type=int,default=12)
    p.add_argument('--config',type=Path,default=ROOT/'authoring/provenance/selected_config.json')
    p.add_argument('--evidence-dir',type=Path,default=ROOT/'authoring/evidence/calibration')
    p.add_argument('--report',type=Path,default=ROOT/'authoring/evidence/calibration.md')
    args=p.parse_args(argv)
    if (not np.isfinite(args.tol) or args.tol<=0 or min(args.budgets)<1
            or min(args.max_solves,args.baseline_steps)<1 or args.baseline_margin<1):
        p.error('positive finite tolerance, budgets, iteration limits and margin >= 1 required')
    args.evidence_dir.mkdir(parents=True,exist_ok=True)
    args.config.parent.mkdir(parents=True,exist_ok=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.config.unlink(missing_ok=True)

    with contextlib.redirect_stdout(io.StringIO()):
        mesh,u,history=afem.run_targeted(problem.initial_mesh(),tol=args.tol,budget=max(args.budgets),
                                N_max=args.max_solves,integration_h=args.integration_h,error_order=args.error_order)
    # Finer settings on the same final mesh: quadrature sensitivity (same code paths).
    checked=measure(mesh,args.integration_h/2,max(20,args.error_order+4))
    stable=abs(checked['relative_error']-history[-1]['reported_error'])<=max(1e-8,args.tol*1e-3)
    reference=energy_error(mesh,u,order=args.error_order).reference_norm**2
    green=green_energy()
    baselines,stops={},{}
    for kind in ('uniform','corner','peak_ignored'):
        baselines[kind],stops[kind]=baseline(kind,max(args.budgets),args.baseline_steps,
                                             args.integration_h,args.error_order)
    selected=choose(args.tol,args.budgets,history,baselines,min_margin=args.min_margin,
                    baseline_margin=args.baseline_margin,n_max=args.n_max)
    if selected and len(history)>selected['N_max']:
        selected=None   # the oracle itself must fit the solve limit
    if (not stable or checked['relative_error']>args.tol or abs(reference-green)>1e-8*green
            or any(s['reason']!='budget' for s in stops.values())):
        selected=None
    sweep=theta_sweep(selected,args.integration_h,args.error_order) if selected else []
    if selected:
        pb=problem.DEFAULT_PROBLEM
        selected=dict(selected,A=float(pb.A),alpha=float(pb.alpha),x0=[float(c) for c in pb.x0])

    def rel(path):
        path=path.resolve()
        return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
    result=dict(parameters={k:rel(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                oracle=history,oracle_recheck=checked,quadrature_stable=stable,
                reference_energy=dict(fem_error=reference,green_identity=green),
                baselines=baselines,baseline_stops=stops,theta_sweep=sweep,selected=selected)
    (args.evidence_dir/'measurements.json').write_text(
        json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')

    budget=selected['dof_budget'] if selected else max(args.budgets)
    def best(rows):
        within=[r for r in rows if r['dof']<=budget]
        return min(within,key=lambda r:r['relative_error'])
    names=dict(uniform='Uniform',corner='Corner-only grading',peak_ignored='Peak-ignored AFEM (f = 0)')
    lines=['# Calibration','',
           f'Generated by `{rel(Path(__file__))}`; full data in `{rel(args.evidence_dir/"measurements.json")}`.','',
           f'Target relative energy error: {args.tol}. DOF counts free P1 vertices (not on Gamma_D); '
           'N_max counts solves including the initial solve.','',
           f'## Strategies at DOF budget {budget}','',
           '| Strategy | DOF | Relative energy error | Error / tol |','|---|---:|---:|---:|',
           f'| Reference oracle (targeted, {len(history)} solves) | {history[-1]["dof"]} | '
           f'{history[-1]["reported_error"]:.6g} | {history[-1]["reported_error"]/args.tol:.2f} |']
    for kind,rows in baselines.items():
        r=best(rows)
        lines.append(f'| {names[kind]}, best within budget | {r["dof"]} | {r["relative_error"]:.6g} | {r["relative_error"]/args.tol:.2f} |')
        over=[r for r in rows if r['over_budget']]
        if over:
            o=over[0]
            lines.append(f'| {names[kind]}, first mesh over budget | {o["dof"]} | {o["relative_error"]:.6g} | {o["relative_error"]/args.tol:.2f} |')
    lines+=['',f'Baselines must reach at least {args.baseline_margin:g} x tol within the budget to count as failing.',
            'Corner grading uses geometry only; peak-ignored AFEM solves and marks with f = 0. '
            'All errors are for the full corner + Gaussian problem.',
            'These are the measured mesh sequences, not a proof over every possible mesh.','',
            '## Checks','',
            f'- Finer-quadrature AFEM error on the final mesh: {checked["relative_error"]:.12g} '
            f'(stable: {stable}).',
            f'- ||grad u||^2: fem.error {reference:.12f}, Green\'s identity {green:.12f} '
            f'(relative difference {abs(reference-green)/green:.1e}).','']
    if sweep:
        lines+=['## Dörfler theta sweep under the selected limits','',
                '| theta | Solves | DOF | Relative error | Stop | Passes |','|---:|---:|---:|---:|---|---|']
        lines+=[f'| {r["theta"]} | {r["solves"]} | {r["dof"]} | {r["relative_error"]:.4f} | {r["stop_reason"]} | '
                f'{"yes" if r["passes"] else "no"} |' for r in sweep]
        lines+=['','Small theta needs more solves (N_max); large theta overshoots the DOF budget.','']
    if selected:
        args.config.write_text(json.dumps(selected,indent=2)+'\n',encoding='utf-8')
        lines+=['## Selected','',f'`{rel(args.config)}`:','','```json',json.dumps(selected,indent=2),'```','',
                f'Budget / oracle DOF: {selected["dof_budget"]/history[-1]["dof"]:.3f}. '
                + (f'N_max fixed at {args.n_max} (the oracle uses {len(history)} solves).' if args.n_max else
                   'N_max is 1.5 x the oracle\'s solves, rounded up.'),
                'Copy tol, dof_budget and N_max into specs.txt and tests/config.json, and freeze the peak '
                'parameters in specs.txt (TODO 1.13).','',
                '## Reference oracle run','',
                'The oracle iteration log and submission in `authoring/evidence/calibration/` '
                '(`run.txt`, `submission.json`) are produced by:','','```bash',
                'bash solution/solve.sh --output authoring/evidence/calibration/submission.json '
                '> authoring/evidence/calibration/run.txt','```','']
    else:
        lines+=['Calibration FAILED: no threshold set passed all selection and quadrature checks.']
    args.report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Selected:',selected,flush=True)
    return 0 if selected else 1


if __name__=='__main__':
    raise SystemExit(main())
