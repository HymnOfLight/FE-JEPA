"""Provenance is mandatory (plan B1): the writer refuses without it."""

import json

import pytest

from fejepa.fe.synthetic import generate_synthetic_dataset
from fejepa.report import config_sha256, provenance, write_report


def test_provenance_fields(tmp_path):
    d = generate_synthetic_dataset(tmp_path / "ds", n=2, seed=0)
    p = provenance({"a": 1}, [d], seeds=[0, 1, 2])
    for key in ("timestamp_utc", "git", "config_sha256", "datasets", "seeds",
                "versions"):
        assert key in p
    assert p["datasets"][0]["manifest_sha256"]
    assert p["datasets"][0]["n_instances"] == 2


def test_writer_refuses_without_provenance(tmp_path):
    with pytest.raises(ValueError):
        write_report(tmp_path / "r.json", {"results": {}})


def test_write_and_config_hash(tmp_path):
    d = generate_synthetic_dataset(tmp_path / "ds", n=2, seed=0)
    payload = {"results": {"x": 1}, "provenance": provenance({"a": 1}, [d], [0])}
    out = write_report(tmp_path / "r.json", payload)
    back = json.loads(out.read_text())
    assert back["results"]["x"] == 1
    assert config_sha256({"a": 1}) != config_sha256({"a": 2})
