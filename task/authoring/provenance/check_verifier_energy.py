"""Reproduce the independent verifier energy convergence evidence."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
from verifier.mesh_checks import load_submission
from verifier.fem_ref import solve
from verifier.error import energy_error,green_energy


def main():
    _,mesh=load_submission(ROOT/'authoring/evidence/calibration/submission.json',max_vertices=40000)
    u=solve(mesh)
    a=energy_error(mesh,u)
    b=energy_error(mesh,u,order=24,max_cell_diameter=.0625,corner_panels=6)
    g=green_energy();g2=green_energy(epsabs=5e-13)
    result=dict(default=a.__dict__,refined=b.__dict__,green=g,green_refined=g2)
    assert abs(b.reference_squared-g2)<1e-10
    assert abs(a.relative-b.relative)<1e-8
    assert abs(b.relative-.0490148156)<1e-9
    assert abs(g-g2)<1e-10
    (ROOT/'authoring/evidence/verifier_energy.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
