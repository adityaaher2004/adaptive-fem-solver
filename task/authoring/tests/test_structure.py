"""Bundle structure rules checked at submission (run before every submit).

Mirrors the submission structure check: pinned pip installs, apt hygiene,
network modes, timeouts and resource sizes, no --platform pins, GPU fields,
author facts, no canary markers; plus bundle hygiene (LF scripts, no editor
swap files or caches, required files present).
"""
import re
import tomllib
from pathlib import Path

import pytest

TASK = Path(__file__).resolve().parents[2]
TOML_TEXT = (TASK/'task.toml').read_text(encoding='utf-8')
TOML = tomllib.loads(TOML_TEXT)
DOCKERFILES = sorted(TASK.rglob('Dockerfile'))
SCRIPTS = sorted(TASK.rglob('*.sh'))
MANIFESTS = DOCKERFILES + SCRIPTS + sorted(TASK.rglob('requirements*.txt'))


def logical_lines(path):
    """Dockerfile/shell lines with backslash continuations joined."""
    text = path.read_text(encoding='utf-8').replace('\\\n', ' ')
    return [line.strip() for line in text.splitlines()]


def text_files():
    for path in TASK.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            data = path.read_bytes()
            if b'\0' not in data[:8192]:
                yield path, data


def test_required_files():
    for rel in ('instruction.md', 'task.toml', 'README.md', 'environment/Dockerfile',
                'solution/solve.sh', 'tests/test.sh', 'tests/Dockerfile'):
        assert (TASK/rel).is_file(), rel
    readme = (TASK/'README.md').read_text(encoding='utf-8')
    for heading in ('## Difficulty', '## Reference solution', '## Verification'):
        assert re.search(rf'^{re.escape(heading)}\s*$', readme, re.M), heading


def test_pip_and_uv_installs_pinned():
    found = 0
    for path in MANIFESTS:
        for line in logical_lines(path):
            for cmd in re.findall(r'(?:pip3?|uv\s+pip|uv)\s+(?:install|add)\s+([^&;|]*)', line):
                for arg in cmd.split():
                    if arg.startswith('-') or arg in ('\\',):
                        continue
                    found += 1
                    assert re.fullmatch(r'[A-Za-z0-9_.\-\[\],]+==[A-Za-z0-9_.+\-]+', arg), \
                        f'{path.relative_to(TASK)}: unpinned install "{arg}"'
    assert found, 'no pip installs found at all'


def test_verifier_pins():
    text = (TASK/'tests/Dockerfile').read_text(encoding='utf-8')
    assert 'pytest==8.4.1' in text and 'pytest-json-ctrf==0.3.5' in text


def test_apt_hygiene():
    for path in DOCKERFILES:
        for line in logical_lines(path):
            if re.search(r'apt(-get)?\s+install', line):
                assert 'apt-get update' in line, f'{path}: install without apt-get update'
                assert 'rm -rf /var/lib/apt/lists/*' in line, f'{path}: apt lists not removed'
                pkgs = line.split('install', 1)[1].split('&&')[0].split()
                assert not [p for p in pkgs if '=' in p and not p.startswith('-')], \
                    f'{path}: apt packages must not be version-pinned'


def test_no_platform_pins():
    for path in DOCKERFILES:
        for line in logical_lines(path):
            assert not re.match(r'FROM\s+--platform', line, re.I), f'{path}: FROM --platform'


def walk_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from walk_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_keys(value)


def test_task_toml_rules():
    assert TOML['environment']['network_mode'] == 'public'
    assert TOML['verifier']['environment']['network_mode'] == 'no-network'
    assert 'allow_internet' not in set(walk_keys(TOML))
    assert 9000 <= TOML['agent']['timeout_sec'] <= 28800
    assert 0 < TOML['verifier']['timeout_sec'] <= 28800
    env = TOML['environment']
    assert env['cpus'] in (1, 2, 4, 8, 16)
    assert env['memory_mb'] in (1024, 2048, 4096, 8192, 16384)
    assert 0 < env['storage_mb'] <= 40960
    assert env['gpus'] in (0, 1)
    if env['gpus'] == 1:
        assert isinstance(env.get('gpu_types'), list) and len(env['gpu_types']) == 1
        assert not (TASK/'environment/docker-compose.yaml').exists()


def test_artifacts_above_first_section():
    first_section = re.search(r'^\[', TOML_TEXT, re.M).start()
    assert re.search(r'^artifacts\s*=', TOML_TEXT[:first_section], re.M)
    assert TOML['artifacts'] == ['/app/output/submission.json']


def test_author_facts():
    meta, task = TOML['metadata'], TOML['task']
    placeholders = {'', 'todo', 'tbd', 'n/a', 'na', 'none yet', 'your name', 'xxx'}
    for key in ('author_name', 'author_email', 'author_organization', 'author_profile',
                'conflicts_of_interest', 'relevant_experience', 'difficulty_explanation',
                'solution_explanation', 'verification_explanation'):
        assert str(meta.get(key, '')).strip().lower() not in placeholders, key
    assert re.match(r'https?://\S+\.\S+', meta['author_profile'])
    assert meta['expert_time_estimate_hours'] > 0
    assert task['description'].strip()
    assert task['keywords'] == meta['tags'] and meta['tags']
    assert task['authors'] == [{'name': meta['author_name'], 'email': meta['author_email']}]


def test_no_canary_markers():
    me = Path(__file__).resolve()
    for path, data in text_files():
        if path.resolve() == me:
            continue
        text = data.decode('utf-8', 'replace').lower()
        assert 'canary' not in text, f'{path.relative_to(TASK)} contains a canary marker'


def test_scripts_and_dockerfiles_use_lf():
    for path in SCRIPTS + DOCKERFILES + sorted(TASK.rglob('.dockerignore')):
        assert b'\r' not in path.read_bytes(), f'{path.relative_to(TASK)} has CRLF line endings'
    for path in SCRIPTS:
        assert path.read_text(encoding='utf-8').startswith('#!'), f'{path} lacks a shebang'


# Python caches are excluded from both images by .dockerignore; pytest itself
# creates them, so remove them when packaging rather than testing for them here.
@pytest.mark.parametrize('pattern', ['*.swp', '*.swo', '*~', '.DS_Store', 'Thumbs.db'])
def test_no_editor_files(pattern):
    hits = [str(p.relative_to(TASK)) for p in TASK.rglob(pattern)]
    assert not hits, f'remove before submitting: {hits}'


def test_verifier_creates_artifact_parent():
    # Static submission check "artifact-parent-not-created": a literal RUN line.
    text = (TASK/'tests/Dockerfile').read_text(encoding='utf-8')
    for artifact in TOML['artifacts']:
        parent = artifact.rsplit('/', 1)[0]
        assert re.search(rf'^RUN mkdir -p {re.escape(parent)}\s*$', text, re.M), \
            f'tests/Dockerfile needs a literal "RUN mkdir -p {parent}" line'


def test_test_sh_writes_reward_literally():
    # Static submission check "reward-txt-missing": the literal path in a write.
    text = (TASK/'tests/test.sh').read_text(encoding='utf-8')
    writes = re.findall(r'(?:printf|echo)\s+([01])\s*>\s*/logs/verifier/reward\.txt', text)
    assert {'0', '1'} <= set(writes), 'test.sh must write 0 and 1 to /logs/verifier/reward.txt literally'
