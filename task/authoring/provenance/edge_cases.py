"""Verifier edge cases (TODO 3.7): variants of the oracle submission.

Run from anywhere: python authoring/provenance/edge_cases.py
Writes authoring/evidence/edge_cases.md; exits 1 on any unexpected reward.
authoring/tests/test_edge_cases.py runs the same cases under pytest.

Each case transforms the oracle submission (or writes raw text) and states the
reward the sealed verifier must give. Reward-1 cases guard against rejecting
legitimate submissions; reward-0 cases guard against accepting malformed ones.
"""
import copy
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT/'authoring/evidence/calibration/submission.json'
spec = importlib.util.spec_from_file_location('check_cheats', ROOT/'authoring/provenance/check_cheat_attempts.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
CAP = 4*check.make.BUDGET + 16


def oracle():
    return json.loads(ORACLE.read_text(encoding='utf-8'))


def renumbered(d):
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(d['vertices']))           # new index of old vertex
    v = [None]*len(perm)
    for old, new in enumerate(perm):
        v[new] = d['vertices'][old]
    t = [[int(perm[i]) for i in tri] for tri in d['triangles']]
    order = rng.permutation(len(t))
    return dict(d, vertices=v, triangles=[t[k] for k in order])


def rotated(d):
    return dict(d, triangles=[tri[k % 3:] + tri[:k % 3] for k, tri in enumerate(d['triangles'])])


def integer_coordinates(d):
    return dict(d, vertices=[[int(c) if float(c).is_integer() else c for c in p] for p in d['vertices']])


def negative_zero(d):
    return dict(d, vertices=[[-0.0 if c == 0 else c for c in p] for p in d['vertices']])


def extra_keys(d):
    h = [dict(r, note='extra', nested={'a': [1, 2.5, None]}) for r in d['history']]
    return dict(d, history=h, solver='afem', metadata={'theta': 0.4, 'ok': True})


def history_len(n):
    def f(d):
        h = (d['history']*2)[:n]
        return dict(d, history=[dict(r, iter=k) for k, r in enumerate(h)])
    return f


def scaled_error(factor):
    return lambda d: dict(d, reported_error=d['reported_error']*factor)


def clockwise(d):
    t = copy.deepcopy(d['triangles'])
    t[0] = [t[0][0], t[0][2], t[0][1]]
    return dict(d, triangles=t)


def boundary_noise(d):
    v = copy.deepcopy(d['vertices'])
    k = next(i for i, p in enumerate(v) if p[0] == -1 and abs(p[1]) < 1)
    v[k] = [np.nextafter(-1.0, 0.0), v[k][1]]                # one ulp inside the domain
    return dict(d, vertices=v)


def boundary_outside(d):
    v = copy.deepcopy(d['vertices'])
    k = next(i for i, p in enumerate(v) if p[0] == -1 and abs(p[1]) < 1)
    v[k] = [np.nextafter(-1.0, -2.0), v[k][1]]               # -1.0000000000000002
    return dict(d, vertices=v)


def replace(path, value):
    def f(d):
        d = copy.deepcopy(d)
        target = d
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return d
    return f


def drop(key):
    return lambda d: {k: v for k, v in d.items() if k != key}


def text(raw):
    return lambda d: raw


def json_text(fn, old, new):
    return lambda d: json.dumps(fn(d)).replace(old, new, 1)


def oversize_vertices(d):
    v = d['vertices'] + [[-.5, .5 + 1e-7*k] for k in range(CAP + 1 - len(d['vertices']))]
    return dict(d, vertices=v)


def oversize_triangles(d):
    return dict(d, triangles=d['triangles'] + [d['triangles'][0]]*(2*CAP + 1 - len(d['triangles'])))


# (name, transform, expected reward, what it checks)
CASES = [
    ('oracle', lambda d: d, 1, 'unchanged oracle submission'),
    ('renumbered', renumbered, 1, 'vertices and triangles randomly permuted'),
    ('rotated_triangles', rotated, 1, 'each triangle cyclically rotated (still CCW)'),
    ('integer_coordinates', integer_coordinates, 1, 'whole-number coordinates written as JSON integers'),
    ('negative_zero', negative_zero, 1, 'every 0.0 coordinate written as -0.0 (Neumann edge, theta=0 edge, origin)'),
    ('extra_keys', extra_keys, 1, 'extra top-level and history keys, nested values'),
    ('history_54', history_len(54), 1, 'exactly N_max history records'),
    ('reported_1.9x', scaled_error(1.9), 1, 'reported_error 1.9x the true error (inside factor 2)'),
    ('reported_0.55x', scaled_error(0.55), 1, 'reported_error 0.55x the true error (inside factor 2)'),
    ('pretty_printed', json_text(lambda d: d, '"vertices"', '\n  "vertices"'), 1, 'extra whitespace/newlines in JSON'),

    ('clockwise_triangle', clockwise, 0, 'one triangle clockwise'),
    ('boundary_ulp_inside', boundary_noise, 0, 'boundary vertex at x = -1 + 1 ulp (not exactly on the side)'),
    ('boundary_ulp_outside', boundary_outside, 0, 'boundary vertex at x = -1.0000000000000002'),
    ('nan_literal', json_text(lambda d: d, '-1.0', 'NaN'), 0, 'NaN literal in vertices'),
    ('infinity_literal', json_text(lambda d: d, '-1.0', '-Infinity'), 0, '-Infinity literal in vertices'),
    ('overflow_number', json_text(lambda d: d, '-1.0', '-1e999'), 0, 'number that overflows to -inf'),
    ('nan_in_history_extra', json_text(extra_keys, '"note": "extra"', '"note": NaN'), 0, 'NaN hidden in an extra history key'),
    ('duplicate_key', json_text(lambda d: d, '"dof": ', '"dof": 0, "dof": '), 0, 'duplicate "dof" key'),
    ('dof_as_float', replace(['dof'], 5383.0), 0, 'dof written as 5383.0'),
    ('dof_as_string', replace(['dof'], '5383'), 0, 'dof written as a string'),
    ('iter_as_float', replace(['history', 0, 'iter'], 0.0), 0, 'history iter written as 0.0'),
    ('estimator_negative', replace(['history', 0, 'estimator'], -1.0), 0, 'negative estimator'),
    ('reported_error_string', replace(['reported_error'], '0.049'), 0, 'reported_error as a string'),
    ('reported_error_negative', replace(['reported_error'], -0.049), 0, 'negative reported_error'),
    ('reported_2.1x', scaled_error(2.1), 0, 'reported_error 2.1x the true error'),
    ('reported_0.45x', scaled_error(0.45), 0, 'reported_error 0.45x the true error'),
    ('history_55', history_len(55), 0, 'N_max + 1 history records'),
    ('history_empty', history_len(0), 0, 'empty history'),
    ('missing_history', drop('history'), 0, 'history key missing'),
    ('bool_coordinate', replace(['vertices', 0, 0], False), 0, 'boolean coordinate'),
    ('index_out_of_range', replace(['triangles', 0, 0], 10**6), 0, 'triangle index out of range'),
    ('index_huge', replace(['triangles', 0, 0], 2**70), 0, 'triangle index beyond int64'),
    ('top_level_array', text('[]'), 0, 'top-level JSON array'),
    ('empty_file', text(''), 0, 'empty file'),
    ('truncated_json', json_text(lambda d: d, '"history"', '"history'), 0, 'syntactically broken JSON'),
    ('utf8_bom', lambda d: '﻿' + json.dumps(d), 0, 'UTF-8 byte-order mark (not allowed by RFC 8259)'),
    ('oversize_vertices', oversize_vertices, 0, f'{CAP + 1} vertices (cap {CAP})'),
    ('oversize_triangles', oversize_triangles, 0, f'{2*CAP + 1} triangles (cap {2*CAP})'),
]


def write_case(name, transform, directory):
    data = transform(oracle())
    path = Path(directory)/f'{name}.json'
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding='utf-8')
    return path


def run_case(name, transform, directory):
    path = write_case(name, transform, directory)
    start = time.perf_counter()
    reward, failing = check.run_verifier(path)
    return reward, failing, time.perf_counter() - start


def main():
    import tempfile
    rows, ok = [], True
    with tempfile.TemporaryDirectory() as tmp:
        for name, transform, expected, what in CASES:
            reward, failing, seconds = run_case(name, transform, tmp)
            ok &= reward == expected
            rows.append((name, what, expected, reward, failing, seconds))
            print(f'{name:26s} expected {expected} got {reward}  {seconds:5.1f}s  {sorted(failing)}', flush=True)
    lines = ['# Verifier edge cases', '',
             'Generated by `authoring/provenance/edge_cases.py` (also run by',
             '`authoring/tests/test_edge_cases.py`). Each case is a variant of the oracle submission,',
             'scored by the sealed `tests/test_submission.py` with the `tests/test.sh` reward rule.', '',
             '| Case | Variant | Expected | Reward | Failing tests | Time |',
             '|---|---|---:|---:|---|---:|']
    for name, what, expected, reward, failing, seconds in rows:
        tests = ', '.join(f'`{t}`' for t in sorted(failing)) or '—'
        lines.append(f'| `{name}` | {what} | {expected} | {reward} | {tests} | {seconds:.1f} s |')
    lines += ['', 'All cases scored as expected.' if ok else '**CHECK FAILED: unexpected reward.**', '']
    (ROOT/'authoring/evidence/edge_cases.md').write_text('\n'.join(lines), encoding='utf-8')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
