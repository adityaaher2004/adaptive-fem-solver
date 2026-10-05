"""Determinism of the oracle and the verifier (TODO 3.8).

Run from anywhere: python authoring/provenance/check_determinism.py [--no-docker]
Writes authoring/evidence/determinism.md; exits 1 on any mismatch.

Oracle: solution/afem.py is rerun under different Python hash seeds and BLAS
thread counts and, if Docker is available, in the environment image at 1 and
4 CPUs. Every run must reproduce the committed oracle submission
(authoring/evidence/calibration/submission.json): identical mesh, DOF, history
iter/dof/marked sequence and stop reason; floats (reported_error, estimators)
equal to within 1e-12 relative.

Verifier: the relative energy error of that submission is recomputed with the
sealed verifier modules in-process (twice, two hash seeds) and, with Docker,
in the verifier image at 1 and 4 CPUs. All values must agree to 1e-12 relative,
far below the 9.85e-4 gap between the oracle error and the tolerance.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT/'authoring/evidence/calibration/submission.json'
SOLVE_ARGS = ['--tol', '0.05', '--budget', '2930', '--N-max', '14']
FLOAT_RTOL = 1e-12
ERROR_SNIPPET = ("import sys; sys.path.insert(0, {tests!r}); "
                 "from verifier.mesh_checks import load_submission; from verifier.fem_ref import solve; "
                 "from verifier.error import energy_error; "
                 "_, m = load_submission({path!r}, max_vertices=11736); "
                 "print(repr(energy_error(m, solve(m)).relative))")


def compare(run, ref):
    """Return (ok, max relative float difference, reason)."""
    if run['vertices'] != ref['vertices'] or run['triangles'] != ref['triangles']:
        return False, None, 'mesh differs'
    if run['dof'] != ref['dof'] or len(run['history']) != len(ref['history']):
        return False, None, 'dof or history length differs'
    keys = ('iter', 'dof', 'marked', 'stop_reason')
    if [[r.get(k) for k in keys] for r in run['history']] != [[r.get(k) for k in keys] for r in ref['history']]:
        return False, None, 'history sequence differs'
    pairs = [(run['reported_error'], ref['reported_error'])]
    pairs += [(a[k], b[k]) for a, b in zip(run['history'], ref['history'])
              for k in ('estimator', 'reported_error', 'relative_estimator') if k in b]
    worst = max(abs(x - y)/abs(y) if y else abs(x) for x, y in pairs)
    return worst <= FLOAT_RTOL, worst, '' if worst <= FLOAT_RTOL else 'floats differ'


def local_oracle(out, hashseed, threads):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed), PYTHONDONTWRITEBYTECODE='1')
    if threads:
        env.update(OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    subprocess.run([sys.executable, str(ROOT/'solution/afem.py'), *SOLVE_ARGS, '--output', str(out)],
                   env=env, check=True, capture_output=True)


def docker(*args):
    return subprocess.run(['docker', *args], check=True, capture_output=True, text=True).stdout


def docker_oracle(out_dir, cpus):
    docker('run', '--rm', '--cpus', str(cpus), '--memory', '2g', '--network', 'none',
           '-v', f'{ROOT/"solution"}:/solution:ro', '-v', f'{out_dir}:/app/output',
           'afem-env:determinism', 'bash', '/solution/solve.sh')


def local_error(path, hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed), PYTHONDONTWRITEBYTECODE='1')
    code = ERROR_SNIPPET.format(tests=str(ROOT/'tests'), path=str(path))
    return float(subprocess.run([sys.executable, '-c', code], env=env, check=True,
                                capture_output=True, text=True).stdout.strip())


def docker_error(path, cpus):
    code = ERROR_SNIPPET.format(tests='/tests', path='/s.json')
    return float(docker('run', '--rm', '--cpus', str(cpus), '--memory', '2g', '--network', 'none',
                        '-v', f'{path}:/s.json:ro', 'afem-verifier:determinism', 'python3', '-c', code).strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--no-docker', action='store_true')
    args = parser.parse_args(argv)
    use_docker = not args.no_docker and shutil.which('docker') is not None
    if use_docker:
        try:
            docker('info')
        except (subprocess.CalledProcessError, OSError):
            use_docker = False
    if use_docker:
        docker('build', '-q', '-t', 'afem-env:determinism', str(ROOT/'environment'))
        docker('build', '-q', '-t', 'afem-verifier:determinism', str(ROOT/'tests'))

    ref = json.loads(REFERENCE.read_text(encoding='utf-8'))
    oracle_rows, verifier_rows, ok = [], [], True
    with tempfile.TemporaryDirectory(dir=ROOT/'authoring') as tmp:
        tmp = Path(tmp)
        runs = [('host, PYTHONHASHSEED=0', lambda o: local_oracle(o, 0, None)),
                ('host, PYTHONHASHSEED=12345', lambda o: local_oracle(o, 12345, None)),
                ('host, 1 BLAS thread', lambda o: local_oracle(o, 0, 1)),
                ('host, 4 BLAS threads', lambda o: local_oracle(o, 0, 4))]
        if use_docker:
            runs += [(f'environment image, {c} CPU', (lambda c: lambda o: docker_oracle(o.parent, c))(c)) for c in (1, 4)]
        for k, (label, run) in enumerate(runs):
            out = tmp/f'run{k}'/'submission.json'
            out.parent.mkdir()
            run(out)
            good, worst, reason = compare(json.loads(out.read_text(encoding='utf-8')), ref)
            ok &= good
            oracle_rows.append((label, good, worst, reason))
            print(f'oracle   {label:28s} {"OK" if good else "MISMATCH"}  max rel float diff {worst}', flush=True)

        values = [('host, PYTHONHASHSEED=0', local_error(REFERENCE, 0)),
                  ('host, PYTHONHASHSEED=12345', local_error(REFERENCE, 12345))]
        if use_docker:
            values += [(f'verifier image, {c} CPU', docker_error(REFERENCE, c)) for c in (1, 4)]
        base = values[0][1]
        for label, value in values:
            good = abs(value - base) <= FLOAT_RTOL*abs(base)
            ok &= good
            verifier_rows.append((label, value, good))
            print(f'verifier {label:28s} {value!r}  {"OK" if good else "MISMATCH"}', flush=True)

    lines = ['# Determinism', '',
             'Generated by `authoring/provenance/check_determinism.py`.', '',
             'Neither the oracle nor the verifier uses random numbers or wall-clock time; sorts that',
             'affect results are stable, and the only set iteration (NVB neighbour lookup) is over',
             'sets of at most one element. The runs below confirm this empirically.', '',
             f'## Oracle (`solution/afem.py {" ".join(SOLVE_ARGS)}`)', '',
             'Each run is compared with the committed `authoring/evidence/calibration/submission.json`:',
             'mesh, DOF, history sequence and stop reason must be identical; floats within '
             f'{FLOAT_RTOL:g} relative.', '',
             '| Run | Result | Max relative float difference |', '|---|---|---:|']
    lines += [f'| {label} | {"identical" if good else "**" + reason + "**"} | '
              f'{"—" if worst is None else f"{worst:.1e}"} |' for label, good, worst, reason in oracle_rows]
    lines += ['', '## Verifier (relative energy error of the oracle submission)', '',
              '| Run | Relative energy error | Agrees |', '|---|---:|---|']
    lines += [f'| {label} | {value!r} | {"yes" if good else "**no**"} |' for label, value, good in verifier_rows]
    if not use_docker:
        lines += ['', '_Docker was not available: container runs skipped._']
    lines += ['', f'Gap between the oracle error and the tolerance: {0.05 - ref["reported_error"]:.3e}.',
              '', 'All runs agree.' if ok else '**CHECK FAILED.**', '']
    (ROOT/'authoring/evidence/determinism.md').write_text('\n'.join(lines), encoding='utf-8')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
