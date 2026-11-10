"""Tests for scripts/hf_publish.sh, run with bash on a small fake export (dist/hf/
with a manifest.json as secbench.hfexport writes it), with a fake `uv` first on PATH:
it records every call, runs `uv run python` with this interpreter, and fails if `hf`
is called, unless a test lets it answer as huggingface_hub's CLI would (whoami, repos
create, <type>s info, upload). No network, no token: the tokens here are made up."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from secbench import hfcommon

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "hf_publish.sh"
BASH = shutil.which("bash")
USER = "someone"
TOKEN = "hf" + "_" + "FakeTokenForTests" * 2  # joined: no whole token in the file
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")

FAKE_UV = '''"""A stand-in for uv: records each call (with what the token's variables
hold), runs `uv run python` with this interpreter, and answers `uv run hf` only when
FAKE_HF is 1: whoami as FAKE_HF_USER (of FAKE_HF_ORGS), and `<type>s info` as the
Hub would: a repo of FAKE_HF_MISSING (ids, comma-separated) isn't found until
`repos create` made it, one of FAKE_HF_UNREADABLE never is, one of FAKE_HF_PUBLIC is
public, and a repo's files (siblings) are FAKE_HF_FILES."""
import json, os, sys


def ids(name):
    return [x for x in os.environ.get(name, "").split(",") if x]


args = sys.argv[1:]
log_path = os.environ["FAKE_UV_LOG"]
token_path = os.environ.get("HF_TOKEN_PATH")
seen = {k: bool(os.environ.get(k)) for k in ("HF_TOKEN", "HF_OIDC_RESOURCE")}
seen["token_path"] = token_path
seen["token_file"] = bool(token_path) and os.path.exists(token_path)
seen["token_folder"] = bool(token_path) and os.path.isdir(os.path.dirname(token_path))
with open(log_path, "a") as log:
    log.write(json.dumps({"args": args, "env": seen}) + "\\n")
if args[:4] == ["run", "python", "-m", "secbench.hfexport"]:
    sys.exit(int(os.environ.get("FAKE_VERIFY_EXIT", "0")))
if args[:2] == ["run", "python"]:
    os.execv(sys.executable, [sys.executable, *args[2:]])
if args[:2] == ["run", "hf"]:
    if os.environ.get("FAKE_HF") != "1":
        print("fake uv: hf must not be called", file=sys.stderr)
        sys.exit(97)
    hf = args[2:]
    if hf[:2] == ["auth", "whoami"]:
        orgs = os.environ.get("FAKE_HF_ORGS") or None
        print(json.dumps({"user": os.environ["FAKE_HF_USER"], "orgs": orgs, "endpoint": None}))
    elif hf[1:2] == ["info"]:
        repo = hf[2]
        with open(log_path) as log:
            made = [json.loads(line)["args"] for line in log]
        created = {a[4] for a in made if a[2:4] == ["repos", "create"]}
        if repo in ids("FAKE_HF_UNREADABLE") or (
            repo in ids("FAKE_HF_MISSING") and repo not in created
        ):
            print(f"Error: Repository '{repo}' not found.", file=sys.stderr)
            sys.exit(1)
        info = {"id": repo, "private": repo not in ids("FAKE_HF_PUBLIC")}
        if "siblings" in hf[hf.index("--expand") + 1].split(","):
            info["siblings"] = [{"rfilename": f} for f in ids("FAKE_HF_FILES")]
        print(json.dumps(info))
    sys.exit(0)
print(f"fake uv: unexpected {args}", file=sys.stderr)
sys.exit(98)
'''
# The 4-bit model's files on the Hub, as the GPU instance uploads them.
INT4_FILES = ".gitattributes,config.json,model.safetensors,recipe.yaml"
NOTHING_UPLOADED = (
    "error: no file was uploaded (a repo this run created is private and empty)\n"
)

FILES = {
    "model/README.md": b"# A model card\n",
    "model/adapter_config.json": b"{" + b" " * 1534 + b"}",  # 1,536 bytes
    "model/adapter_model.safetensors": bytes(5 << 19),  # 2.5 MiB
    "int4-awq/README.md": b"int4 card\n",  # 10 bytes
    "dataset/README.md": b"# A dataset card\n",
    "dataset/data/answers/validation.jsonl": b'{"value": "1"}\n',
    "space/README.md": b"---\nsdk: gradio\n---\n",
    "space/app.py": b"print('app')\n",
    "space/data/texts/validation/0000001001-26-000010.md": b"# A release\n",
}


def build_dist(root: Path, user: str = USER, files: dict | None = None) -> Path:
    """An export as secbench.hfexport lays it out, with its manifest."""
    files = FILES if files is None else files
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    kinds = [
        k for k in hfcommon.REPO_NAMES if any(n.startswith(k + "/") for n in files)
    ]
    repos = {}
    for kind in kinds:
        repo = {
            "id": hfcommon.repo_id(user, kind),
            "type": hfcommon.REPO_TYPES[kind],
            "folder": kind,
            "private": True,
        }
        if kind == hfcommon.INT4:
            repo["card_only"] = True
        if kind == hfcommon.SPACE:
            repo["sdk"] = "gradio"
        repos[kind] = repo
    manifest = {
        "hf_user": user,
        "build": {"commit": "3624f3c62c7e" + "0" * 28, "dirty": False},
        "repos": repos,
        "skipped": [],
        "files": [
            {
                "path": name,
                "size": len(files[name]),
                "sha256": hashlib.sha256(files[name]).hexdigest(),
            }
            for name in sorted(files)
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


@pytest.fixture
def dist(tmp_path) -> Path:
    return build_dist(tmp_path / "dist" / "hf")


def run(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    """The script under bash, with the fake uv first on PATH and no Hugging Face
    variable from this environment; the int4-awq repo holds its model's files unless
    FAKE_HF_FILES says otherwise."""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    (fake / "fake_uv.py").write_text(FAKE_UV)
    uv = fake / "uv"
    uv.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake / "fake_uv.py"}" "$@"\n')
    uv.chmod(0o755)
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("HF_", "HUGGING_FACE_", "FAKE_"))
    }
    clean.update(
        {
            "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}",
            "FAKE_UV_LOG": str(tmp_path / "uv.log"),
            "FAKE_HF_FILES": INT4_FILES,
            **env,
        }
    )
    return subprocess.run(
        [BASH, str(SCRIPT), *args],
        check=False,
        cwd=tmp_path,
        env=clean,
        capture_output=True,
        text=True,
        timeout=120,
    )


def calls(tmp_path: Path) -> list[dict]:
    log = tmp_path / "uv.log"
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def hf_calls(tmp_path: Path) -> list[list[str]]:
    return [c["args"][2:] for c in calls(tmp_path) if c["args"][:2] == ["run", "hf"]]


def uploads(tmp_path: Path) -> list[list[str]]:
    return [c for c in hf_calls(tmp_path) if c[0] == "upload"]


# The calls a real run makes, as the fake uv records them.


def script(dist: Path, *args: str) -> list[str]:
    """The script's own Python (the manifest helper)."""
    return ["run", "python", "-", str(dist), str(dist), *args]


def create(repo: str, kind: str, *extra: str) -> list[str]:
    return [
        *("run", "hf", "repos", "create", repo, "--type", kind),
        *("--private", "--exist-ok", *extra),
    ]


def info(kind: str, repo: str, expand: str = "private") -> list[str]:
    command = ["run", "hf", f"{kind}s", "info", repo]
    return [*command, "--expand", expand, "--format", "json"]


def hub_info(repo: str, files: str | None = None) -> str:
    """The fake's answer to `<type>s info` for a private repo."""
    answer: dict = {"id": repo, "private": True}
    if files is not None:
        answer["siblings"] = [{"rfilename": f} for f in files.split(",")]
    return json.dumps(answer)


def upload(repo: str, local: str, remote: str, kind: str, message: str) -> list[str]:
    return [
        *("run", "hf", "upload", repo, local, remote, "--type", kind, "--private"),
        *("--commit-message", message),
    ]


# The dry run


def test_the_dry_run_lists_every_file(tmp_path, dist):
    result = run(tmp_path, "--dry-run", "--dist", str(dist))
    assert result.returncode == 0, result.stderr
    assert (
        result.stdout
        == f"""\
Hugging Face export {dist} (dry run: nothing is sent)
user: {USER}
commit message: Upload from {hfcommon.REPO_URL} at 3624f3c62c7e

model: {USER}/qwen3-8b-sec-earnings-lora (model, private)
  folder: {dist}/model
        15 B  README.md
     1.5 KiB  adapter_config.json
     2.5 MiB  adapter_model.safetensors
  3 files, 2.5 MiB

int4-awq: {USER}/qwen3-8b-sec-earnings-int4-awq (model, private)
  folder: {dist}/int4-awq
  only README.md is uploaded: the model's files must already be in the repo \
(uploaded from the GPU instance)
        10 B  README.md
  1 file, 10 B

dataset: {USER}/sec-earnings-bench (dataset, private)
  folder: {dist}/dataset
        17 B  README.md
        15 B  data/answers/validation.jsonl
  2 files, 32 B

space: {USER}/sec-earnings-explorer (space, private, sdk gradio)
  folder: {dist}/space
        20 B  README.md
        13 B  app.py
        12 B  data/texts/validation/0000001001-26-000010.md
  3 files, 45 B

Total: 4 repos, 9 files, 2.5 MiB
A real run creates a missing repo private and reads every repo's visibility from \
the Hub first: if one is public, it uploads nothing (unless --allow-public).
Nothing was sent (dry run).
"""
    )
    assert result.stderr == ""
    assert all(c["args"][:2] == ["run", "python"] for c in calls(tmp_path))
    assert hf_calls(tmp_path) == []


def test_the_dry_run_names_the_placeholder_and_whats_skipped(tmp_path):
    dist = build_dist(tmp_path / "dist", hfcommon.PLACEHOLDER_USER)
    manifest = json.loads((dist / "manifest.json").read_text())
    manifest["skipped"] = ["int4-awq card: results/serving_results.json not found"]
    manifest["build"]["dirty"] = True
    (dist / "manifest.json").write_text(json.dumps(manifest))
    result = run(tmp_path, "--dry-run", "--dist", str(dist), HF_TOKEN=TOKEN)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "user: HF_USER, the placeholder: a real run needs a rebuild with" in out
    assert "at 3624f3c62c7e, with uncommitted changes\n" in out
    assert "Skipped in the build: int4-awq card: results/serving_results.json" in out
    assert out.endswith("Nothing was sent (dry run).\n")
    assert TOKEN not in out + result.stderr
    assert hf_calls(tmp_path) == []


def test_the_dry_run_stops_when_a_folder_differs_from_its_manifest(tmp_path, dist):
    (dist / "dataset" / "README.md").unlink()
    (dist / "space" / "app.py").write_text("print('changed')\n")
    (dist / "space" / ".DS_Store").write_bytes(b"\0")
    (dist / "int4-awq" / "model.safetensors").write_bytes(b"\0")  # not uploaded
    result = run(tmp_path, "--dry-run", "--dist", str(dist))
    assert result.returncode == 1
    assert result.stderr.splitlines() == [
        f"error: dataset/README.md: in the manifest, not in {dist}",
        "error: space/app.py: 17 bytes, the manifest says 13",
        "error: space/.DS_Store: not in the manifest, and hf upload would send it",
        (
            f"error: {dist} doesn't match its manifest: rebuild it "
            "(uv run python -m secbench.hfexport --hf-user NAME)"
        ),
    ]
    assert result.stdout.endswith("Nothing was sent (dry run).\n")
    assert hf_calls(tmp_path) == []


# Manifests it refuses: (a change, the refusal's message).
UNFOLLOWED = {
    "another id": (
        lambda m: m["repos"]["model"].update(id="else/qwen3-8b-sec-earnings-lora"),
        "id 'else/qwen3-8b-sec-earnings-lora', expected 'someone/",
    ),
    "public": (lambda m: m["repos"]["dataset"].update(private=False), "isn't private"),
    "folder": (
        lambda m: m["repos"]["space"].update(folder="../space"),
        "isn't a folder's",
    ),
    "file": (
        lambda m: m["files"].append({"path": "../x", "size": 1}),
        "isn't a manifest",
    ),
}


def test_a_manifest_it_wont_follow(tmp_path):
    """Each case in its own folder, run at once (each run imports secbench.hfcommon,
    about a second)."""
    folders = {}
    for case, (change, _) in UNFOLLOWED.items():
        dist = build_dist(tmp_path / case / "dist")
        manifest = json.loads((dist / "manifest.json").read_text())
        change(manifest)
        (dist / "manifest.json").write_text(json.dumps(manifest))
        folders[case] = (tmp_path / case, dist)
    with ThreadPoolExecutor() as pool:
        results = {
            case: pool.submit(run, folder, "--dry-run", "--dist", str(dist))
            for case, (folder, dist) in folders.items()
        }
        for case, (_, message) in UNFOLLOWED.items():
            result = results[case].result()
            assert result.returncode == 1, case
            assert message in result.stderr, case
            assert hf_calls(tmp_path / case) == [], case


# Refusals before anything is sent


@pytest.mark.parametrize("token", [None, "", " ", " \t\n"])
def test_without_a_token_it_refuses(tmp_path, dist, token):
    """A blank token too: huggingface_hub would strip it to nothing and fall back to
    a stored login."""
    env = {} if token is None else {"HF_TOKEN": token}
    result = run(tmp_path, "--dist", str(dist), **env)
    assert result.returncode != 0
    assert "HF_TOKEN isn't set, or is blank" in result.stderr
    assert calls(tmp_path) == []  # not even the manifest is read


def test_the_placeholder_user_is_refused(tmp_path):
    dist = build_dist(tmp_path / "dist", hfcommon.PLACEHOLDER_USER)
    result = run(tmp_path, "--dist", str(dist), HF_TOKEN=TOKEN)
    assert result.returncode != 0
    assert "placeholder user HF_USER" in result.stderr
    assert "--hf-user NAME" in result.stderr
    assert [c["args"][:3] for c in calls(tmp_path)] == [["run", "python", "-"]]
    assert TOKEN not in result.stdout + result.stderr


def test_a_failed_verify_sends_nothing(tmp_path, dist):
    result = run(
        tmp_path, "--dist", str(dist), HF_TOKEN=TOKEN, FAKE_HF="1", FAKE_VERIFY_EXIT="1"
    )
    assert result.returncode != 0
    assert "secbench.hfexport --verify failed: nothing was sent" in result.stderr
    assert hf_calls(tmp_path) == []


def test_a_token_of_another_user_is_refused(tmp_path, dist):
    result = run(
        tmp_path, "--dist", str(dist), HF_TOKEN=TOKEN, FAKE_HF="1", FAKE_HF_USER="other"
    )
    assert result.returncode != 0
    assert "the token is other's and the export is for someone" in result.stderr
    assert hf_calls(tmp_path) == [["auth", "whoami", "--format", "json"]]


@pytest.mark.parametrize("arguments", [["--bogus"], ["--dist"]])
def test_bad_arguments(tmp_path, arguments):
    result = run(tmp_path, *arguments)
    assert result.returncode != 0
    assert "usage: bash scripts/hf_publish.sh" in result.stderr
    assert calls(tmp_path) == []


def test_a_missing_export(tmp_path):
    result = run(tmp_path, "--dry-run", "--dist", str(tmp_path / "nothing"))
    assert result.returncode != 0
    assert "manifest.json not found: build it with" in result.stderr


# The upload


def test_an_upload_verifies_then_checks_every_repo_then_uploads(tmp_path, dist):
    """A first run: the adapter's, the dataset's and the Space's repos don't exist
    yet, so they are created private; the int4-awq one exists, with its model."""
    model, int4, dataset, space = (
        hfcommon.repo_id(USER, k) for k in ("model", "int4-awq", "dataset", "space")
    )
    result = run(
        tmp_path,
        "--dist",
        str(dist),
        HF_TOKEN=TOKEN,
        HF_OIDC_RESOURCE="someone/x",  # would take precedence over HF_TOKEN
        FAKE_HF="1",
        FAKE_HF_USER=USER,
        FAKE_HF_MISSING=f"{model},{dataset},{space}",
    )
    assert result.returncode == 0, result.stderr
    message = f"Upload from {hfcommon.REPO_URL} at 3624f3c62c7e"
    account = json.dumps({"user": USER, "orgs": None, "endpoint": None})
    recorded = calls(tmp_path)
    assert [c["args"] for c in recorded] == [
        script(dist, "plan"),
        ["run", "python", "-m", "secbench.hfexport", "--verify", "--out", str(dist)],
        ["run", "hf", "auth", "whoami", "--format", "json"],
        script(dist, "whoami", USER, account),
        create(model, "model"),
        info("model", model),
        script(dist, "repo", model, "0", hub_info(model)),
        info("model", int4, "private,siblings"),  # never created: its model is there
        script(dist, "repo", int4, "1", hub_info(int4, INT4_FILES)),
        create(dataset, "dataset"),
        info("dataset", dataset),
        script(dist, "repo", dataset, "0", hub_info(dataset)),
        create(space, "space", "--space-sdk", "gradio"),
        info("space", space),
        script(dist, "repo", space, "0", hub_info(space)),
        upload(model, f"{dist}/model", ".", "model", message),
        upload(int4, f"{dist}/int4-awq/README.md", "README.md", "model", message),
        upload(dataset, f"{dist}/dataset", ".", "dataset", message),
        upload(space, f"{dist}/space", ".", "space", message),
    ]
    # hf saw the token from HF_TOKEN only: no OIDC exchange, and its stored login's
    # file in an empty folder of the script's own, removed at the end. The token is
    # never an argument or printed.
    hf = [c["env"] for c in recorded if c["args"][:2] == ["run", "hf"]]
    assert all(
        (e["HF_TOKEN"], e["HF_OIDC_RESOURCE"], e["token_file"], e["token_folder"])
        == (True, False, False, True)
        for e in hf
    )
    assert len({e["token_path"] for e in hf}) == 1
    assert not Path(hf[0]["token_path"]).parent.exists()
    assert TOKEN not in (tmp_path / "uv.log").read_text()
    assert TOKEN not in result.stdout + result.stderr
    assert "Logged in as someone\n" in result.stdout
    for kind in ("model", "int4-awq", "dataset", "space"):
        assert f"uploaded: {hfcommon.hub_url(USER, kind)}\n" in result.stdout
    assert result.stdout.endswith(
        "== Done: each repo's visibility, as the Hub reported it before the upload\n"
        + "".join(f"{repo}: private\n" for repo in (model, int4, dataset, space))
        + "Look at each private repo on the Hub, then make it public in its settings.\n"
    )


def test_a_public_repo_stops_the_run_before_any_upload(tmp_path, dist):
    """An existing repo keeps its visibility (--private only applies to a new one):
    every repo is checked, then nothing is uploaded."""
    dataset, space = (hfcommon.repo_id(USER, k) for k in ("dataset", "space"))
    result = run(
        tmp_path,
        "--dist",
        str(dist),
        HF_TOKEN=TOKEN,
        FAKE_HF="1",
        FAKE_HF_USER=USER,
        FAKE_HF_PUBLIC=f"{dataset},{space}",
    )
    assert result.returncode != 0
    assert f"dataset: {dataset} (dataset): public\n" in result.stdout
    assert result.stderr == (
        f"error: public on the Hub: {dataset} {space} (an upload there is published "
        "at once): make each one private in its settings and run this again, or run "
        "with --allow-public\n" + NOTHING_UPLOADED
    )
    assert info("space", space)[2:] in hf_calls(tmp_path)
    assert uploads(tmp_path) == []


def test_allow_public_uploads_and_says_which_repos_are_public(tmp_path, dist):
    model, dataset = (hfcommon.repo_id(USER, k) for k in ("model", "dataset"))
    result = run(
        tmp_path,
        "--dist",
        str(dist),
        "--allow-public",
        HF_TOKEN=TOKEN,
        FAKE_HF="1",
        FAKE_HF_USER=USER,
        FAKE_HF_PUBLIC=dataset,
    )
    assert result.returncode == 0, result.stderr
    assert len(uploads(tmp_path)) == 4
    done = result.stdout.split("== Done")[1]
    assert f"{model}: private\n{hfcommon.repo_id(USER, 'int4-awq')}: private\n" in done
    assert f"\n{dataset}: public\n" in done
    assert f"Public (--allow-public): {dataset}: their files are published.\n" in done


UPLOAD_INT4_FIRST = (
    "upload the int4-awq model there from the GPU instance first (this script "
    "uploads only its card)"
)


@pytest.mark.parametrize(
    "hub, message",
    [
        ({"FAKE_HF_MISSING": "INT4"}, "isn't on the Hub, or the token can't see it"),
        (  # weights in a folder aren't at the top, where vLLM reads them
            {"FAKE_HF_FILES": ".gitattributes,README.md,weights/model.safetensors"},
            "has no config.json and no *.safetensors weights on the Hub",
        ),
    ],
    ids=["no repo", "no model"],
)
def test_the_int4_card_goes_only_where_its_model_is(tmp_path, dist, hub, message):
    int4 = hfcommon.repo_id(USER, "int4-awq")
    message = f"{message}: {UPLOAD_INT4_FIRST}"
    hub = {k: v.replace("INT4", int4) for k, v in hub.items()}
    result = run(
        tmp_path,
        "--dist",
        str(dist),
        HF_TOKEN=TOKEN,
        FAKE_HF="1",
        FAKE_HF_USER=USER,
        **hub,
    )
    assert result.returncode != 0
    assert f"error: {int4} {message}" in result.stderr
    assert result.stderr.endswith(NOTHING_UPLOADED)
    assert ["repos", "create", int4] not in [c[:3] for c in hf_calls(tmp_path)]
    assert uploads(tmp_path) == []


def test_a_repo_hf_cant_read_stops_the_run(tmp_path, dist):
    space = hfcommon.repo_id(USER, "space")
    result = run(
        tmp_path,
        "--dist",
        str(dist),
        HF_TOKEN=TOKEN,
        FAKE_HF="1",
        FAKE_HF_USER=USER,
        FAKE_HF_UNREADABLE=space,  # created, yet not found
    )
    assert result.returncode != 0
    assert result.stderr.endswith(
        f"error: hf couldn't read {space} on the Hub\n" + NOTHING_UPLOADED
    )
    assert uploads(tmp_path) == []


def test_an_organizations_export_takes_a_members_token(tmp_path):
    dist = build_dist(tmp_path / "dist", "an-org")
    result = run(
        tmp_path,
        "--dist",
        str(dist),
        HF_TOKEN=TOKEN,
        FAKE_HF="1",
        FAKE_HF_USER="member",
        FAKE_HF_ORGS="other-org,an-org",
    )
    assert result.returncode == 0, result.stderr
    assert "Logged in as member, a member of an-org\n" in result.stdout
    created = [c[2] for c in hf_calls(tmp_path) if c[:2] == ["repos", "create"]]
    assert created == [
        hfcommon.repo_id("an-org", k) for k in ("model", "dataset", "space")
    ]
    uploaded = [c[1] for c in uploads(tmp_path)]
    assert uploaded == [
        hfcommon.repo_id("an-org", k) for k in ("model", "int4-awq", "dataset", "space")
    ]


def test_without_an_int4_folder_there_is_no_int4_repo(tmp_path):
    files = {k: v for k, v in FILES.items() if not k.startswith("int4-awq/")}
    dist = build_dist(tmp_path / "dist", files=files)
    result = run(tmp_path, "--dry-run", "--dist", str(dist))
    assert result.returncode == 0, result.stderr
    assert "int4-awq" not in result.stdout
    assert "Total: 3 repos, 8 files, 2.5 MiB\n" in result.stdout


# The script itself


def test_the_script_never_traces_or_names_a_token_file():
    """The token stays in the environment, and the only token file the script names
    is the one in its own empty folder (see the upload test)."""
    text = SCRIPT.read_text()
    assert "set -euo pipefail\n" in text
    assert "set -x" not in text and "xtrace" not in text
    assert "--token" not in text
    assert ".cache/huggingface" not in text
    assert 'export HF_TOKEN_PATH="$NO_LOGIN/token"\n' in text


def test_help(tmp_path):
    result = run(tmp_path, "--help")
    assert result.returncode == 0
    assert result.stdout.startswith("Publishes the Hugging Face export")
    assert (
        "Usage: bash scripts/hf_publish.sh [--dry-run] [--dist dist/hf] "
        "[--allow-public]\n" in result.stdout
    )
    assert calls(tmp_path) == []
