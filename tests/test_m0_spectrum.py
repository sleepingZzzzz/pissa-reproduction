import json
import struct
from argparse import Namespace

import pytest
import requests
import torch

from scripts.m0 import plot_singular as s


class Response:
    def __init__(self, body, status=206, content_range="bytes 0-7/100", etag="a"):
        self.body, self.status_code = body, status
        self.headers = {"Content-Range": content_range, "ETag": etag}
        self.read = False
        self.closed = False

    def iter_content(self, chunk_size):
        self.read = True
        yield self.body

    def close(self):
        self.closed = True


def test_range_rejects_200_before_body(monkeypatch):
    response = Response(b"huge file", status=200)
    monkeypatch.setattr(requests, "get", lambda *a, **k: response)
    with pytest.raises(ValueError):
        s._range_get("https://invalid/model", 0, 7, retries=0)
    assert not response.read and response.closed


@pytest.mark.parametrize("body,range_header", [(b"short", "bytes 0-7/100"),
    (b"12345678", "bytes 1-8/100"), (b"12345678", "")])
def test_bad_ranges(monkeypatch, body, range_header):
    response = Response(body, content_range=range_header)
    monkeypatch.setattr(requests, "get", lambda *a, **k: response)
    with pytest.raises(ValueError):
        s._range_get("https://invalid/model", 0, 7, retries=0)
    assert response.closed


def test_range_retry_and_etag_consistency(monkeypatch):
    responses = [requests.Timeout("temporary"), Response(b"12345678"),
                 Response(b"12345678", etag="changed")]
    def get(*args, **kwargs):
        assert kwargs["stream"] is True
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(requests, "get", get)
    state, trace = {}, []
    assert s._range_get("https://invalid/model", 0, 7, retries=1,
                        etag_state=state, trace=trace) == b"12345678"
    assert len(trace) == 2
    with pytest.raises(ValueError, match="ETag"):
        s._range_get("https://invalid/model", 0, 7, retries=0,
                     etag_state=state, trace=trace)


def test_tensor_length_and_header_limit(monkeypatch):
    monkeypatch.setattr(s, "_range_get", lambda *a, **k: struct.pack("<Q", 2**40))
    with pytest.raises(ValueError):
        s._load_tensor_via_range("https://invalid", "m.safetensors", "w")


def test_valid_remote_tensor_and_bad_offsets(monkeypatch):
    def fixture(offset_end):
        header = json.dumps({"w": {"dtype": "F32", "shape": [2, 2],
                                  "data_offsets": [0, offset_end]}}).encode()
        blob = struct.pack("<Q", len(header))+header+torch.arange(4, dtype=torch.float32).numpy().tobytes()
        def get(url, start, end, **kwargs):
            return blob[start:end+1]
        return get
    monkeypatch.setattr(s, "_range_get", fixture(16))
    weight, meta = s._load_tensor_via_range("https://invalid", "m.safetensors", "w")
    assert torch.equal(weight, torch.arange(4.).reshape(2, 2))
    assert meta["original_dtype"] == "torch.float32"
    monkeypatch.setattr(s, "_range_get", fixture(12))
    with pytest.raises(ValueError, match="length mismatch"):
        s._load_tensor_via_range("https://invalid", "m.safetensors", "w")


def test_missing_remote_key_never_substituted(monkeypatch):
    monkeypatch.setattr(s, "_index_json", lambda *a: {"weight_map": {"other.q_proj.weight": "x.safetensors"}})
    with pytest.raises(KeyError):
        s._resolve_key("https://invalid", "missing")


@pytest.mark.parametrize("sharded", [False, True])
def test_local_safetensors_dtype_and_strict_key(tmp_path, sharded):
    from safetensors.torch import save_file
    key = "model.layers.0.self_attn.q_proj.weight"
    original = torch.randn(12, 10).bfloat16()
    filename = "part.safetensors" if sharded else "model.safetensors"
    save_file({key: original}, str(tmp_path / filename))
    if sharded:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: filename}}))
    weight, meta = s.load_local_weight(tmp_path, key)
    assert torch.equal(weight, original.float())
    assert meta["original_dtype"] == "torch.bfloat16"
    with pytest.raises(KeyError):
        s.load_local_weight(tmp_path, "missing.weight")


def test_real_failure_not_silently_random(tmp_path):
    args = Namespace(random=False, local_path=tmp_path, layer="missing", source="auto",
                     model="missing", allow_random_fallback=False)
    with pytest.raises((FileNotFoundError, KeyError)):
        s.load_weight(args)


def test_plot_smoke_and_provenance(tmp_path, capsys):
    assert s.main(["--random", "--in-features", "12", "--out-features", "16", "--r", "3",
                   "--output", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / "metrics/singular.json").read_text())
    assert payload["status"] == "passed"
    assert payload["provenance"]["mode"] == "random"
    assert payload["provenance"]["actual_model"] is None
    result = payload["results"]
    assert len(result["histogram_bins"]) > 2
    assert sum(result["W_histogram_counts"]) == 16 * 12
    assert sum(result["W_res_histogram_counts"]) == 16 * 12
    assert (tmp_path / "figures/singular_values.png").exists()
    output = capsys.readouterr().out
    assert "任务已成功完成" in output
    assert str((tmp_path / "metrics/singular.json").resolve()) in output


def test_plot_failure_overwrites_previous_success_status(tmp_path, capsys):
    target = tmp_path / "metrics/singular.json"
    target.parent.mkdir()
    target.write_text(json.dumps({"status": "passed", "exit_code": 0}))
    assert s.main(["--local-path", str(tmp_path / "missing.safetensors"),
                   "--output", str(tmp_path)]) == 1
    payload = json.loads(target.read_text())
    assert payload["status"] == "error" and payload["exit_code"] == 1
    assert "任务已成功完成" not in capsys.readouterr().out


def test_final_save_failure_does_not_print_success(tmp_path, monkeypatch, capsys):
    original_finish = s.finish_report

    def fail_success_save(payload, path, status, code, started):
        if status == "passed":
            raise OSError("deliberate final save failure")
        return original_finish(payload, path, status, code, started)

    monkeypatch.setattr(s, "finish_report", fail_success_save)
    assert s.main(["--random", "--in-features", "8", "--out-features", "10", "--r", "2",
                   "--output", str(tmp_path)]) == 1
    assert "任务已成功完成" not in capsys.readouterr().out
    payload = json.loads((tmp_path / "metrics/singular.json").read_text())
    assert payload["status"] == "error"


def test_spectrum_parameter_error_is_64():
    with pytest.raises(SystemExit) as exc:
        s.main(["--r", "0"])
    assert exc.value.code == 64
