"""Sealed submission checks; never import the agent's fem package.

Run on /app/output/submission.json, or set SUBMISSION_PATH for authoring runs.
DOF means vertices outside Gamma_D. Empty history is rejected: at least the
initial SOLVE must be represented. Extra history fields remain allowed.
A 4*budget+16 vertex cap is an explicit resource policy, not a mathematical
consequence of a free-DOF budget (boundary refinement can add arbitrarily many
constrained vertices). Publish this cap in the agent-facing instructions.
"""
import math
import os
from pathlib import Path

import numpy as np
import pytest
from verifier.mesh_checks import load_config, read_submission, load_submission, validate_schema
from verifier.fem_ref import solve
from verifier.error import energy_error

DEFAULT_SUBMISSION = Path('/app/output/submission.json')
ERROR_FACTOR = 2.0
NEAR_THRESHOLD = 1e-6


@pytest.fixture(scope='session')
def config():
    return load_config(Path(__file__).with_name('config.json'))


@pytest.fixture(scope='session')
def submission_path():
    return Path(os.environ.get('SUBMISSION_PATH', str(DEFAULT_SUBMISSION)))


@pytest.fixture(scope='session')
def max_vertices(config):
    return 4*config['dof_budget']+16


@pytest.fixture(scope='session')
def parsed(submission_path, max_vertices):
    assert submission_path.is_file(), f'missing submission: {submission_path}'
    return read_submission(submission_path, max_vertices=max_vertices)


@pytest.fixture(scope='session')
def validated(submission_path, max_vertices):
    # Required guard is passed to the independent loader, before geometry.
    return load_submission(submission_path, max_vertices=max_vertices)


def compute_error(mesh, config):
    """Re-solve independently and use refined integration near the threshold."""
    u = solve(mesh, config)
    result = energy_error(mesh, u, config)
    assert math.isfinite(result.relative), 'nonfinite computed energy error'
    if abs(result.relative-config['error_tol']) < NEAR_THRESHOLD:
        result = energy_error(mesh, u, config, order=24,
                              max_cell_diameter=0.0625, corner_panels=6)
    assert math.isfinite(result.relative), 'nonfinite refined energy error'
    return result.relative


@pytest.fixture(scope='session')
def computed_error(validated, config):
    _, mesh = validated
    # Avoid expensive assembly on an over-budget candidate even when pytest
    # continues running after the independent budget test fails.
    assert mesh.dof <= config['dof_budget'], 'DOF budget exceeded'
    return compute_error(mesh, config)


def test_exists_and_parses(parsed):
    assert isinstance(parsed, dict), 'submission must be a JSON object'


def test_schema(parsed):
    validate_schema(parsed)
    # Arrays/index/geometry validation belongs to test_mesh_valid.
    assert isinstance(parsed['vertices'], list), 'vertices must be an array'
    assert isinstance(parsed['triangles'], list), 'triangles must be an array'


def test_mesh_valid(validated):
    _, mesh = validated
    assert mesh.area == pytest.approx(3., rel=0, abs=1e-11)


def test_dof_matches_reported(validated):
    data, mesh = validated
    actual = int(np.count_nonzero(~mesh.dirichlet_vertex_mask))
    assert data['dof'] == actual, f'reported DOF {data["dof"]} != recomputed {actual}'


def test_dof_budget(validated, config):
    _, mesh = validated
    assert mesh.dof <= config['dof_budget'], f'{mesh.dof} DOFs exceeds budget'


def test_history_length(parsed, config):
    history = parsed['history']
    assert isinstance(history, list), 'history must be a list'
    assert 1 <= len(history) <= config['max_iter'], (
        f'history must contain 1..{config["max_iter"]} records, one per SOLVE')


def test_computed_error(computed_error, config):
    assert computed_error <= config['error_tol'], (
        f'recomputed relative energy error {computed_error:.12g} exceeds '
        f'{config["error_tol"]:.12g}')


def test_reported_error(parsed, computed_error):
    reported = parsed['reported_error']
    assert not isinstance(reported, bool) and isinstance(reported, (int,float))
    assert math.isfinite(reported) and reported >= 0
    # Tiny additive slack only for numerically zero errors; no pass/fail slack.
    slack = 1e-12
    assert computed_error/ERROR_FACTOR-slack <= reported <= ERROR_FACTOR*computed_error+slack, (
        f'reported error {reported:.12g} not within factor {ERROR_FACTOR:g} '
        f'of recomputed {computed_error:.12g}')
