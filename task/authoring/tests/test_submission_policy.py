"""Authoring tests of submission-test policy, without costly FEM solves."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
spec=importlib.util.spec_from_file_location('submission_policy',ROOT/'tests/test_submission.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
from verifier.mesh_checks import load_submission,MeshValidationError


@pytest.mark.parametrize('initial,refined',[(.0500005,.049999),(.0499995,.050001)])
def test_refined_result_controls_decision(monkeypatch,initial,refined):
    calls=[]
    monkeypatch.setattr(policy,'solve',lambda *a:None)
    def error(*args,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(relative=initial if len(calls)==1 else refined)
    monkeypatch.setattr(policy,'energy_error',error)
    assert policy.compute_error(None,{'error_tol':.05})==refined
    assert calls==[{},dict(order=24,max_cell_diameter=.0625,corner_panels=6)]


def test_far_from_threshold_no_reintegration(monkeypatch):
    calls=[]
    monkeypatch.setattr(policy,'solve',lambda *a:None)
    def error(*args,**kwargs):
        calls.append(kwargs);return SimpleNamespace(relative=.04)
    monkeypatch.setattr(policy,'energy_error',error)
    assert policy.compute_error(None,{'error_tol':.05})==.04
    assert len(calls)==1


@pytest.mark.parametrize('count',[0,51])
def test_history_bounds(count):
    with pytest.raises(AssertionError):
        policy.test_history_length({'history':[{}]*count},{'max_iter':50})


def test_fabricated_zero_error_rejected():
    with pytest.raises(AssertionError):
        policy.test_reported_error({'reported_error':0.},.049)


def test_resource_cap_precedes_geometry(tmp_path,monkeypatch):
    path=tmp_path/'oversize.json'
    path.write_text('{"vertices":'+str([[0,0]]*9)+'}')
    def forbidden(*args):
        raise AssertionError('geometry was called')
    monkeypatch.setattr('verifier.mesh_checks.validate_mesh',forbidden)
    with pytest.raises(MeshValidationError,match='vertex resource limit'):
        load_submission(path,max_vertices=8)
