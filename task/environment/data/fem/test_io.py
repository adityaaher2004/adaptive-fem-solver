import json

import numpy as np
import pytest

from fem.io import load_mesh, save_mesh, write_submission
from fem.mesh import Mesh


REC = {'iter': 0, 'dof': 1, 'estimator': .5}


def mesh():
    return Mesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]]).uniform_refine()


def test_mesh_roundtrip(tmp_path):
    m = mesh()
    path = tmp_path / 'nested' / 'mesh.json'
    save_mesh(path, m)
    assert set(json.loads(path.read_text(encoding='utf-8'))) == {'vertices', 'triangles'}
    restored = load_mesh(path)
    for name in m.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(restored, name), getattr(m, name))


def test_submission_roundtrip(tmp_path):
    m = mesh()
    path = tmp_path / 'submission.json'
    history = [{'iter': np.int64(0), 'dof': np.int64(1), 'estimator': np.float64(.5),
                'counts': np.array([3, 6]), 'note': 'Δ'}]
    write_submission(path, m, dof=np.int64(1), reported_error=np.float64(.1), history=history)
    data = json.loads(path.read_text(encoding='utf-8'))
    assert set(data) == {'vertices', 'triangles', 'dof', 'reported_error', 'history'}
    assert data['dof'] == 1 and data['reported_error'] == .1
    assert data['history'][0] == {'iter': 0, 'dof': 1, 'estimator': .5,
                                  'counts': [3, 6], 'note': 'Δ'}
    np.testing.assert_array_equal(load_mesh(path).triangles, m.triangles)
    assert isinstance(history[0]['counts'], np.ndarray)


@pytest.mark.parametrize('text', [
    '[]', '{}', '{"vertices": [], "triangles": []}',
    '{"vertices":[],"vertices":[],"triangles":[]}',
    '{"vertices":[[0,0],[1,0],[0,NaN]],"triangles":[[0,1,2]]}',
    '{"vertices":[[0,0],[1,0],[0,1]],"triangles":[[0,1.0,2]]}',
    '{"vertices":[[0,0],[1,0],[0,1]],"triangles":[[false,1,2]]}',
    '{"vertices":[[false,0],[1,0],[0,1]],"triangles":[[0,1,2]]}',
    '{"vertices":[[0,0],[1,0],[0,1]],"triangles":[[0,2,1]]}',
    '{"vertices":[[0,0],[1,0],[0,1]],"triangles":[[0,1,3]]}',
    '{bad json',
])
def test_invalid_mesh_json(tmp_path, text):
    path = tmp_path / 'bad.json'
    path.write_text(text, encoding='utf-8')
    with pytest.raises((ValueError, TypeError)):
        load_mesh(path)


@pytest.mark.parametrize('changes', [
    {'dof': True}, {'dof': 1.5}, {'dof': -1}, {'dof': 7},
    {'reported_error': float('nan')}, {'reported_error': float('inf')},
    {'reported_error': -.1}, {'reported_error': True},
    {'history': {}}, {'history': [1]},
    {'history': [{**REC, 1: 'bad'}]},
    {'history': [dict(REC, nested=[np.inf])]},
])
def test_invalid_submission_preserves_destination(tmp_path, changes):
    path = tmp_path / 'submission.json'
    path.write_text('previous content', encoding='utf-8')
    args = dict(dof=0, reported_error=0., history=[])
    args.update(changes)
    with pytest.raises((ValueError, TypeError)):
        write_submission(path, mesh(), **args)
    assert path.read_text(encoding='utf-8') == 'previous content'
    assert list(tmp_path.iterdir()) == [path]


def test_replace_failure_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / 'mesh.json'
    path.write_text('previous content', encoding='utf-8')
    def fail(*args):
        raise OSError('replace failed')
    monkeypatch.setattr('fem.io.os.replace', fail)
    with pytest.raises(OSError, match='replace failed'):
        save_mesh(path, mesh())
    assert path.read_text(encoding='utf-8') == 'previous content'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('record, error', [
    ({'dof': 1, 'estimator': .5}, ValueError),
    ({'iter': 0, 'estimator': .5}, ValueError),
    ({'iter': 0, 'dof': 1}, ValueError),
    (dict(REC, iter=-1), ValueError),
    (dict(REC, dof=-1), ValueError),
    (dict(REC, iter=1.0), TypeError),
    (dict(REC, dof=True), TypeError),
    (dict(REC, dof=np.bool_(True)), TypeError),
    (dict(REC, estimator='0.5'), TypeError),
    (dict(REC, estimator=False), TypeError),
    (dict(REC, estimator=-1e-3), ValueError),
    (dict(REC, estimator=np.nan), ValueError),
    (dict(REC, estimator=np.inf), ValueError),
])
def test_history_schema(tmp_path, record, error):
    path = tmp_path / 'submission.json'
    with pytest.raises(error):
        write_submission(path, mesh(), dof=0, reported_error=0.,
                         history=[REC, record])
    assert not path.exists()


def test_history_multiple_records_and_empty(tmp_path):
    path = tmp_path / 'submission.json'
    history = [dict(iter=k, dof=3*k, estimator=1/(k+1)) for k in range(4)]
    write_submission(path, mesh(), dof=2, reported_error=.2, history=history)
    assert json.loads(path.read_text(encoding='utf-8'))['history'] == history
    write_submission(path, mesh(), dof=2, reported_error=.2, history=[])
    assert json.loads(path.read_text(encoding='utf-8'))['history'] == []
