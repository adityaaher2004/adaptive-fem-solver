"""Every cheat attempt scores 0 for its intended reason; the oracle scores 1.

Runs the sealed tests/test_submission.py with the tests/test.sh reward rule
(via authoring/provenance/check_cheat_attempts.py). About 1 minute.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('check_cheats', ROOT/'authoring/provenance/check_cheat_attempts.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.mark.parametrize('name', sorted(check.make.EXPECTED))
def test_cheat_attempt_scores_zero(name):
    reward, failing = check.run_verifier(check.CHEATS/f'{name}.json')
    assert reward == 0
    assert failing == check.make.EXPECTED[name]


def test_oracle_control_scores_one():
    # Same harness must be able to award 1, or the zeros above prove nothing.
    reward, failing = check.run_verifier(ROOT/'authoring/evidence/calibration/submission.json')
    assert (reward, failing) == (1, set())
