"""Tests for secbench.hfexport, the build of dist/hf/, on the small repository of
tests/hf_fixture.py."""

from __future__ import annotations

import json
from pathlib import Path

import hf_fixture
import pytest

from secbench import hfcommon, hfexport, hfscan
from secbench.hfcommon import ExportError, Sources


def build(root: Path, out: Path, user: str = "test-user") -> tuple[dict, list[str]]:
    said: list[str] = []
    record = hfexport.build(Sources(root), out, user, say=said.append)
    return record, said


def test_build_writes_every_folder_and_the_manifest(hf_repo, tmp_path):
    out = tmp_path / "hf"
    record, said = build(hf_repo.root, out)
    assert sorted(record["repos"]) == ["dataset", "model", "space"]
    assert record["repos"]["model"] == {
        "id": "test-user/qwen3-8b-sec-earnings-lora",
        "type": "model",
        "folder": "model",
        "private": True,
    }
    assert record["repos"]["space"]["sdk"] == "static"
    assert record["repos"]["dataset"]["id"] == "test-user/sec-earnings-bench"
    assert "int4-awq card: results/serving_results.json not found" in record["skipped"]
    assert any("serving" in note for note in record["skipped"])
    assert said[0] == "built model/: 5 files"
    assert not (out / "int4-awq").exists()
    assert not out.with_name("hf.part").exists()
    on_disk = hfexport.file_entries(out)
    assert record["files"] == on_disk
    assert json.loads((out / "manifest.json").read_text()) == record
    assert record["hf_user"] == "test-user"
    assert set(record["build"]) == {"commit", "dirty"}
    for kind in ("model", "dataset", "space"):
        assert any(f["path"].startswith(f"{kind}/") for f in record["files"])


def test_two_builds_are_byte_identical(hf_repo, tmp_path):
    first, _ = build(hf_repo.root, tmp_path / "a")
    second, _ = build(hf_repo.root, tmp_path / "b")
    assert first == second
    assert (tmp_path / "a" / "manifest.json").read_bytes() == (
        tmp_path / "b" / "manifest.json"
    ).read_bytes()


def test_a_rebuild_replaces_the_old_build(hf_repo, tmp_path):
    out = tmp_path / "hf"
    build(hf_repo.root, out)
    (out / "stale.txt").write_text("from an older build")
    build(hf_repo.root, out)
    assert not (out / "stale.txt").exists()


def test_serving_results_add_the_int4_card(hf_repo_copy, tmp_path):
    hf_fixture.add_serving_results(hf_repo_copy.root)
    record, _ = build(hf_repo_copy.root, tmp_path / "hf")
    assert record["repos"]["int4-awq"] == {
        "id": "test-user/qwen3-8b-sec-earnings-int4-awq",
        "type": "model",
        "folder": "int4-awq",
        "private": True,
        "card_only": True,
    }
    assert [
        f["path"] for f in record["files"] if f["path"].startswith("int4-awq/")
    ] == ["int4-awq/README.md"]
    assert not any("serving_results.json not found" in n for n in record["skipped"])


def test_a_privacy_finding_stops_the_build(hf_repo, tmp_path, monkeypatch):
    real = hfexport.BUILDERS["space"]

    def leaky(context, out):
        written = real(context, out)
        (out / "data" / "note.json").write_text(
            json.dumps({"org": hf_fixture.SECRET_ORG})
        )
        return written

    monkeypatch.setitem(hfexport.BUILDERS, "space", leaky)
    out = tmp_path / "hf"
    out.mkdir()
    (out / "earlier.txt").write_text("an earlier build")
    with pytest.raises(ExportError, match="privacy scan"):
        build(hf_repo.root, out)
    assert (out / "earlier.txt").exists()  # not replaced
    assert (tmp_path / "hf.part" / "space" / "data" / "note.json").exists()


def test_a_bad_user_stops_the_build(hf_repo, tmp_path):
    with pytest.raises(ExportError, match="Hub name"):
        build(hf_repo.root, tmp_path / "hf", user="a--b")


def test_verify(hf_repo, tmp_path):
    out = tmp_path / "hf"
    build(hf_repo.root, out)
    assert hfexport.verify(out) == []
    readme = out / "model" / "README.md"
    readme.write_text(readme.read_text() + "\nedited by hand\n")
    (out / "space" / "extra.txt").write_text("not built")
    (out / "dataset" / "prompts" / "extract_v1.txt").unlink()
    problems = hfexport.verify(out)
    assert "changed since the build: model/README.md" in problems
    assert "not in the manifest: space/extra.txt" in problems
    assert "missing: dataset/prompts/extract_v1.txt" in problems
    assert hfexport.verify(tmp_path / "nothing") == [
        f"{tmp_path / 'nothing' / 'manifest.json'} not found; build first"
    ]


def test_verify_scans_again(hf_repo, tmp_path):
    out = tmp_path / "hf"
    build(hf_repo.root, out)
    path = out / "model" / "README.md"
    path.write_text(path.read_text() + f"\n{hf_fixture.SECRET_EMAIL}\n")
    problems = hfexport.verify(out)
    assert "changed since the build: model/README.md" in problems
    assert any(p.startswith("privacy: model/README.md") for p in problems)
    assert not any(hf_fixture.SECRET_EMAIL in p for p in problems)  # masked


def test_main(hf_repo, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(hf_repo.root)
    out = tmp_path / "hf"
    assert hfexport.main(["--hf-user", "test-user", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert f"built {out} for test-user" in printed
    assert "-> test-user/sec-earnings-explorer (space)" in printed
    assert "privacy scan: no findings" in printed
    assert hfexport.main(["--verify", "--out", str(out)]) == 0
    assert "every file matches its manifest" in capsys.readouterr().out
    assert hfexport.main(["--hf-user=-bad", "--out", str(tmp_path / "x")]) == 1
    assert "isn't a Hub name" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        hfexport.main(["--out", str(out)])  # --hf-user is required to build
    assert hfexport.main(["--verify", "--out", str(tmp_path / "x")]) == 1


def test_main_reports_a_scan_that_cant_run(hf_repo, tmp_path, monkeypatch, capsys):
    out = tmp_path / "hf"
    build(hf_repo.root, out)

    def broken(repo):
        raise hfscan.ScanError("git can't run")

    monkeypatch.setattr(hfscan, "known_emails", broken)
    monkeypatch.chdir(hf_repo.root)
    assert hfexport.main(["--verify", "--out", str(out)]) == 1
    assert "error: git can't run" in capsys.readouterr().err
    again = tmp_path / "again"
    assert hfexport.main(["--hf-user", "test-user", "--out", str(again)]) == 1
    assert "error: git can't run" in capsys.readouterr().err
    assert not again.exists()


@pytest.mark.parametrize(
    "size, shown",
    [(0, "0 B"), (1023, "1,023 B"), (1024, "1.0 KiB"), (5 * 1024**2, "5.0 MiB")],
)
def test_human_size(size, shown):
    assert hfexport.human_size(size) == shown


def test_placeholder_user_builds(hf_repo, tmp_path):
    record, _ = build(hf_repo.root, tmp_path / "hf", user=hfcommon.PLACEHOLDER_USER)
    assert record["repos"]["space"]["id"] == "HF_USER/sec-earnings-explorer"
