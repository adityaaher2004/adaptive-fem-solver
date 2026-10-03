"""Verifier edge cases (TODO 3.7); cases defined in authoring/provenance/edge_cases.py.

Each case runs the sealed tests/test_submission.py with the tests/test.sh
reward rule. About 2 minutes.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('edge_cases', ROOT/'authoring/provenance/edge_cases.py')
edge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edge)


@pytest.mark.parametrize('name, transform, expected', [c[:3] for c in edge.CASES], ids=[c[0] for c in edge.CASES])
def test_edge_case_reward(tmp_path, name, transform, expected):
    reward, failing, _ = edge.run_case(name, transform, tmp_path)
    assert reward == expected, sorted(failing)
