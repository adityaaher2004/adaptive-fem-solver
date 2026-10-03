"""instruction.md must agree with the sealed config and the shipped fem API."""
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'environment/data'))
from fem import assembly, error, io, problem  # noqa: E402

TEXT = (ROOT/'instruction.md').read_text(encoding='utf-8')
CONFIG = json.loads((ROOT/'tests/config.json').read_text(encoding='utf-8'))


def test_limits_match_sealed_config():
    assert f"**≤ {CONFIG['error_tol']:g}**" in TEXT
    assert f"**≤ {CONFIG['dof_budget']}**" in TEXT
    assert f"**≤ {CONFIG['max_iter']}**" in TEXT
    assert f"at most {CONFIG['max_iter']}" in TEXT
    # test_submission.py: max_vertices = 4*budget + 16, triangles <= 2*max_vertices
    cap = 4*CONFIG['dof_budget'] + 16
    assert f'at most {cap} vertices' in TEXT and f'{2*cap} triangles' in TEXT
    assert 'within a factor of 2' in TEXT


def test_paths_and_schema():
    for path in ('/app/specs.txt', '/app/fem/', '/app/initial_mesh.json',
                 '/app/output/submission.json', 'python -m fem.example_uniform'):
        assert path in TEXT
    for key in ('"vertices"', '"triangles"', '"dof"', '"reported_error"', '"history"',
                '"iter"', '"estimator"'):
        assert key in TEXT
    sample = json.loads(re.search(r'```json\n(.*?)```', TEXT, re.S).group(1)
                        .replace('[[x, y], ...]', '[]').replace('[[i, j, k], ...]', '[]'))
    assert set(sample) == {'vertices', 'triangles', 'dof', 'reported_error', 'history'}
    assert all({'iter', 'dof', 'estimator'} <= set(r) for r in sample['history'])


def test_cited_api_exists():
    assert 'load_refinements' in inspect.signature(assembly.assemble_load).parameters
    assert {'dof', 'reported_error', 'history'} <= set(inspect.signature(io.write_submission).parameters)
    mesh = problem.initial_mesh().uniform_refine()
    system = assembly.assemble_system(mesh, load_refinements=3)
    assert len(system.free_vertices) == 6
    assert hasattr(error.energy_error(mesh, system.lifting), 'relative')
    for name in ('u', 'grad_u', 'f', 'g_D', 'g_N', 'boundary_masks', 'initial_mesh'):
        assert callable(getattr(problem, name))


@pytest.mark.parametrize('corner', ['(−1,−1)', '(0,−1)', '(0,0)', '(1,0)', '(1,1)', '(−1,1)'])
def test_corners_listed(corner):
    assert corner in TEXT
