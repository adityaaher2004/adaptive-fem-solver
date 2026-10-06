"""Reproduce the independent verifier energy convergence evidence.

Writes authoring/evidence/verifier_energy.json and regenerates the results
table and margin line in authoring/evidence/verifier_energy.md, so the
evidence always describes the current oracle submission
(authoring/evidence/calibration/submission.json).
"""
from pathlib import Path
import json
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
from verifier.mesh_checks import load_config,load_submission
from verifier.fem_ref import solve
from verifier.error import energy_error,green_energy

ORACLE=ROOT/'authoring/evidence/calibration/submission.json'
REPORT=ROOT/'authoring/evidence/verifier_energy.md'


def update_report(result,data,tol):
    a,b=result['default'],result['refined']
    table='\n'.join([
        '| Method/settings | Reference energy squared | Oracle relative energy error |',
        '|---|---:|---:|',
        f"| Direct, order 16, diameter 0.125, 4 corner panels | {a['reference_squared']!r} | {a['relative']!r} |",
        f"| Direct, order 24, diameter 0.0625, 6 corner panels | {b['reference_squared']!r} | {b['relative']!r} |",
        f"| Green identity, adaptive tolerance 2e-12 per subintegral | {result['green']!r} | — |",
        f"| Green identity, adaptive tolerance 5e-13 per subintegral | {result['green_refined']!r} | — |"])
    summary=(f"The independent-method discrepancy is about {abs(b['reference_squared']-result['green_refined']):.1e}, below the requested\n"
             '1e-10. The supplied target 1.632536170000 is its rounded value.\n'
             f"The observed oracle relative-error change is about {abs(a['relative']-b['relative']):.0e}, negligible against\n"
             f"the {tol:g} pass threshold and the oracle's approximately {tol-a['relative']:.2e} margin\n"
             f"(oracle submission: {data['dof']} DOF, {len(data['history'])} solves).\n"
             'These are empirical convergence checks, not rigorous interval error bounds.')
    text=REPORT.read_text(encoding='utf-8')
    text=re.sub(r'\| Method/settings .*?\n(?:\|.*\n)+',table+'\n',text,count=1)
    text=re.sub(r'The independent-method discrepancy.*?not rigorous interval error bounds\.',summary,text,count=1,flags=re.S)
    REPORT.write_text(text,encoding='utf-8')


def main():
    config=load_config()
    data,mesh=load_submission(ORACLE,max_vertices=4*config['dof_budget']+16)
    u=solve(mesh)
    a=energy_error(mesh,u)
    b=energy_error(mesh,u,order=24,max_cell_diameter=.0625,corner_panels=6)
    g=green_energy();g2=green_energy(epsabs=5e-13)
    result=dict(default=a.__dict__,refined=b.__dict__,green=g,green_refined=g2)
    assert abs(b.reference_squared-g2)<1e-10
    assert abs(a.relative-b.relative)<1e-8
    assert abs(b.relative-data['reported_error'])<1e-9     # verifier agrees with the oracle's own error
    assert abs(g-g2)<1e-10
    (ROOT/'authoring/evidence/verifier_energy.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    update_report(result,data,config['error_tol'])
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
