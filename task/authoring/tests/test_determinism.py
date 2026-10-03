"""Oracle and verifier are deterministic on the host (TODO 3.8).

Container runs (1 and 4 CPUs) are in authoring/provenance/check_determinism.py,
which also writes authoring/evidence/determinism.md. About 30 s.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('check_determinism', ROOT/'authoring/provenance/check_determinism.py')
det = importlib.util.module_from_spec(spec)
spec.loader.exec_module(det)


@pytest.mark.parametrize('hashseed, threads', [(0, None), (12345, 1)])
def test_oracle_reproduces_committed_submission(tmp_path, hashseed, threads):
    out = tmp_path/'submission.json'
    det.local_oracle(out, hashseed, threads)
    good, worst, reason = det.compare(json.loads(out.read_text(encoding='utf-8')),
                                      json.loads(det.REFERENCE.read_text(encoding='utf-8')))
    assert good, (reason, worst)


def test_verifier_error_independent_of_hash_seed():
    a, b = det.local_error(det.REFERENCE, 0), det.local_error(det.REFERENCE, 12345)
    assert a == b
    assert abs(a - 0.0490148156) < 1e-9
