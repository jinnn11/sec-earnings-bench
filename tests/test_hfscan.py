"""Tests for secbench.hfscan, the privacy scan of the Hugging Face export: each kind
of secret found and masked, the release-text allowance for press contacts, URL paths
left alone, PNG text chunks, safetensors headers, file names, unknown file types, git's
e-mail addresses and the command's exit codes; then the mini repository's runs/*/raw/
files (tests/hf_fixture.py) and, without the allowance, real files (skipped without
them: runs/, training/ and data/ aren't in the repository).

Every fake secret is joined from pieces at run time, so no whole token sits in this
file for a secret scanner to flag."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import hf_fixture
import pytest

from secbench import hfcommon, hfscan, inputs
from secbench.hfscan import Finding

REPO = Path(__file__).resolve().parents[1]
KNOWN = "owner.name@example.org"  # a made-up known address (known_emails' place)
PRESS = hf_fixture.RELEASE_EMAIL  # a press contact, printed in release texts
HOME = "/Users/" + "testowner"


def joined(*parts: str) -> str:
    return "".join(parts)


# (kind, the secret in a line of text, what the excerpt starts with)
SECRETS = [
    (hfscan.KNOWN_EMAIL, KNOWN, "ow…@ex…"),
    (hfscan.KNOWN_EMAIL, KNOWN.upper(), "OW…@EX…"),
    (hfscan.EMAIL, joined("jane.doe", "@", "corp.example.com"), "ja…@co…"),
    (hfscan.HOME_PATH, HOME + "/code/sec-earnings-bench", "/Users/te…"),
    (hfscan.HOME_PATH, joined("/Users/", "alice", "/Downloads"), "/Users/al…"),
    (hfscan.HOME_PATH, joined("/home/", "bob", "/.cache"), "/home/b…"),
    (hfscan.HOME_PATH, joined("file:///home/", "carol", "/x.png"), "/home/ca…"),
    (hfscan.HOME_PATH, joined("/root/", ".cache/huggingface"), "/root/.c…"),
    (hfscan.HOME_PATH, joined("C:\\Users\\", "dave", "\\repo"), "C:\\Users\\da…"),
    (hfscan.HOME_PATH, joined("c:/Users/", "hank", "/x"), "c:/Users/ha…"),  # once
    (hfscan.INSTANCE_PATH, hf_fixture.INSTANCE_PATH, "/workspace/s…"),
    (hfscan.TEMP_PATH, joined("/tmp/", "pytest-of-runner/x"), "/tmp/py…"),
    (hfscan.TEMP_PATH, joined("/private/var/", "folders/cf/x"), "/private/var/fo…"),
    (hfscan.TEMP_PATH, joined("/var/folders/", "cf/2b/T/x"), "/var/folders/c…"),
    (hfscan.TEMP_PATH, joined("/private/tmp/", "build-1/x"), "/private/tmp/bu…"),
    (hfscan.API_KEY, joined("hf", "_", "Fk3QwErTy" * 4), "hf_Fk…"),
    (hfscan.API_KEY, joined("sk", "-proj-", "Ab3dE" * 6), "sk-proj-Ab…"),
    (hfscan.API_KEY, joined("sk", "-ant-api03-", "Xy9Zw" * 8), "sk-ant-api03-Xy…"),
    (hfscan.API_KEY, joined("gh", "p_", "Qr7St" * 7), "ghp_Qr…"),
    (hfscan.API_KEY, joined("github", "_pat_", "11AB", "cd3E" * 6), "github_pat_11…"),
    (hfscan.API_KEY, joined("AK", "IA", "TESTFAKE", "12345678"), "AKIATE…"),
    (hfscan.API_KEY, joined("xo", "xb-", "1234-5678-", "AbCdEf"), "xoxb-12…"),
    (hfscan.API_KEY, joined("Bearer ", "eyFake", "Tok3n" * 4), "Bearer ey…"),
    (hfscan.ACCOUNT_ID, joined("org-", "FakeOrg", "IdAbCd" * 3), "org-Fa…"),
    (hfscan.ACCOUNT_ID, joined("proj", "_", "Ab12" * 6), "proj_Ab…"),
    (hfscan.ACCOUNT_ID, joined("user", "-", "Cd34" * 6), "user-Cd…"),
    (hfscan.ACCOUNT_ID, joined("req", "_", "011CUa" * 4), "req_01…"),
    (hfscan.ACCOUNT_ID, joined("msg", "_", "01Xf" * 6), "msg_01…"),
    (hfscan.ACCOUNT_ID, joined("chat", "cmpl-", "abc123" * 3), "chatcmpl-ab…"),
]
PRIVATE_KEY_BODY = joined("MIIEvQIBADANBgkqhkiG9w0BAQEFAASC", "Fake" * 8)
PRIVATE_KEY = joined(
    "-----BEGIN ", "PRIVATE KEY-----\n", PRIVATE_KEY_BODY, "\n-----END PRIVATE KEY-----"
)


def write(root: Path, relative: str, content: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def scan(root: Path) -> list[Finding]:
    return hfscan.scan(root, emails=[KNOWN], home=HOME)


def summary(findings: list[Finding]) -> list[tuple[str, str, str]]:
    return [(f.path, f.where, f.kind) for f in findings]


def assert_masked(findings: list[Finding], *secrets: str) -> None:
    shown = "\n".join(str(f) for f in findings)
    for secret in secrets:
        assert secret not in shown


# Each kind


@pytest.mark.parametrize("kind, secret, excerpt", SECRETS, ids=lambda x: str(x)[:24])
def test_each_kind_is_found_and_masked(tmp_path, kind, secret, excerpt):
    write(tmp_path, "model/README.md", f"# Card\n\nSee {secret} here.\n")
    findings = scan(tmp_path)
    assert summary(findings) == [("model/README.md", "line 3", kind)]
    assert findings[0].excerpt.startswith(excerpt)
    assert findings[0].excerpt.endswith(("…", "characters)"))
    assert_masked(findings, secret)


def test_a_private_key_shows_its_header_only(tmp_path):
    write(tmp_path, "space/app.py", f'KEY = """{PRIVATE_KEY}"""\n')
    findings = scan(tmp_path)
    assert [(f.where, f.kind, f.excerpt) for f in findings] == [
        ("line 1", hfscan.PRIVATE_KEY, "-----BEGIN PRIVATE KEY-----")
    ]
    assert_masked(findings, PRIVATE_KEY_BODY[:12])


@pytest.mark.parametrize(
    "text",
    [
        "user-friendly-interfaces-for-everyone",  # IDs need a digit or both cases
        "risk-adjusted-performance-measures-and-more",  # "sk-" inside a word
        "the task-Q3Revenue2026Breakdown table",  # random-looking, inside a word
        "org-wide-initiatives-across-the-company",
        "an org-internationalizationprograms effort",  # lowercase: not random
        "the hf_fixture_module and gh_pages_branch",
        "Bearer bonds outstanding-at-year-end",
        "matrix a@b.T and decorators @demo.load",  # no e-mail address
    ],
)
def test_prose_is_not_a_token(tmp_path, text):
    write(tmp_path, "dataset/README.md", text + "\n")
    assert scan(tmp_path) == []


@pytest.mark.parametrize(
    "text",
    [
        "https://ir.example.com/home/default.aspx",
        "http://investors.example.com/Users/guide",
        "www.example.com/root/index.html and example.com/home/x",
        "https://example.com/tmp/report.pdf and https://example.com/private/var/x",
        "https://example.com/workspace/team and http://localhost:8000/home/app",
        "https://github.com/example/repo/blob/main/workspace/x",
    ],
)
def test_url_paths_are_not_home_paths(tmp_path, text):
    write(tmp_path, "space/data/meta.json", json.dumps({"links": [text]}))
    write(tmp_path, "dataset/README.md", f"[link]({text})\n")
    assert scan(tmp_path) == []


def test_the_home_folder_is_found_without_its_leading_separator(tmp_path):
    write(tmp_path, "a.txt", "relative Users/testowner/code and Users\\testowner\n")
    findings = scan(tmp_path)
    assert [(f.kind, f.excerpt) for f in findings] == [
        (hfscan.HOME_PATH, "Users/te…"),
        (hfscan.HOME_PATH, "Users\\te…"),
    ]


# Backslash escapes as code writes them (the text of a .py or .md file): the escape's
# last character would join a path to a URL or a token to a word. (the escape and
# what comes before it, the secret after it, its kind, its excerpt)
ESCAPED = [
    ("a\\n", "/workspace/sec-earnings-bench/x", hfscan.INSTANCE_PATH, "/workspace/s…"),
    ("\\t", "/home/bob/x", hfscan.HOME_PATH, "/home/b…"),
    ("a \\u2014", "/root/.cache", hfscan.HOME_PATH, "/root/.c…"),
    ("a\\r", "/Users/alice/x", hfscan.HOME_PATH, "/Users/al…"),
    ("a\\012", "/tmp/build-1/x", hfscan.TEMP_PATH, "/tmp/bu…"),
    ("token:\\n", joined("hf", "_", "Fk3QwErTy" * 4), hfscan.API_KEY, "hf_Fk…"),
    ("\\x20", joined("sk", "-proj-", "Ab3dE" * 6), hfscan.API_KEY, "sk-proj-Ab…"),
    (
        "id\\u00a0",
        joined("org-", "FakeOrg", "IdAbCd" * 3),
        hfscan.ACCOUNT_ID,
        "org-Fa…",
    ),
    ("Bearer\\t", joined("eyFake", "Tok3n" * 4), hfscan.API_KEY, "Bearer\\tey…"),
]


@pytest.mark.parametrize("name", ["space/app.py", "model/README.md"])
@pytest.mark.parametrize(
    "escape, secret, kind, excerpt", ESCAPED, ids=lambda x: str(x)[:16]
)
def test_a_secret_right_after_a_backslash_escape_is_found(
    tmp_path, name, escape, secret, kind, excerpt
):
    write(tmp_path, name, f'x = 1\nNOTE = "{escape}{secret}"\n')
    findings = scan(tmp_path)
    assert summary(findings) == [(name, "line 2", kind)]
    assert findings[0].excerpt.startswith(excerpt)
    assert findings[0].excerpt.endswith(("…", "characters)"))
    assert_masked(findings, secret)


def test_escapes_in_a_json_string_that_holds_code(tmp_path):
    """JSON decodes its own escapes; a string that holds code keeps the code's."""
    code = 'X = "data:\\n/home/ubuntu/data"'
    write(tmp_path, "space/data/meta.json", json.dumps({"snippet": code}))
    findings = scan(tmp_path)
    assert [(f.where, f.kind, f.excerpt) for f in findings] == [
        ("$.snippet", hfscan.HOME_PATH, "/home/ub…")
    ]


@pytest.mark.parametrize(
    "text",
    [
        'print("a\\nb\\tc\\u2014d")',  # escapes alone
        'url = "\\nhttps://ir.example.com/home/default.aspx"',  # still a URL's path
        "risk\\n-adjusted \\tuser-friendly-interfaces-for-everyone \\x20hf_fixture",
        "markdown escapes \\_ \\* \\# and a literal backslash \\\\n",
    ],
)
def test_escapes_alone_are_not_findings(tmp_path, text):
    write(tmp_path, "space/app.py", text + "\n")
    assert scan(tmp_path) == []


def test_a_windows_path_with_escape_like_names_is_found_once(tmp_path):
    write(tmp_path, "a.md", "C:\\Users\\nancy\\temp\\x\n")
    findings = scan(tmp_path)
    assert [(f.kind, f.excerpt) for f in findings] == [
        (hfscan.HOME_PATH, "C:\\Users\\na…")
    ]


def test_blank_escapes_keeps_offsets():
    escapes = "\\n\\t\\x41\\u2014\\U0001F600\\u{1F600}\\N{EM DASH}\\012\\0"
    text = f"a{escapes}b \\_ \\\\"
    assert hfscan.blank_escapes(text) == "a" + " " * len(escapes) + "b \\_ \\\\"


# JSON, JSONL and the release-text allowance


def test_json_strings_and_keys_are_located_by_path(tmp_path):
    token = joined("hf", "_", "Fk3QwErTy" * 4)
    value = {
        "rows": [{"evidence": "fine"}, {"evidence": f"see {token}"}],
        KNOWN: "a key holding a known address",
        "note": f"also {HOME}/x",
    }
    write(tmp_path, "space/data/errors.json", json.dumps(value))
    findings = scan(tmp_path)
    assert [(f.where, f.kind) for f in findings] == [
        ("$.rows[1].evidence", hfscan.API_KEY),
        ("$: key", hfscan.KNOWN_EMAIL),
        ("$.note", hfscan.HOME_PATH),
    ]
    assert_masked(findings, token, KNOWN, "testowner")


def test_a_secret_in_a_key_is_masked_in_the_path(tmp_path):
    write(tmp_path, "x.json", json.dumps({KNOWN: {"inner": "/home/" + "erin"}}))
    findings = scan(tmp_path)
    assert [f.where for f in findings] == ["$: key", '$["ow…@ex…"].inner']
    assert_masked(findings, KNOWN)


def test_json_escapes_hide_nothing(tmp_path):
    escaped = KNOWN.replace("@", "\\u0040").replace(".", "\\u002e")
    write(tmp_path, "x.json", '{"contact": "' + escaped + '"}')
    assert [f.kind for f in scan(tmp_path)] == [hfscan.KNOWN_EMAIL]


def release_line(**fields: str) -> str:
    return json.dumps(fields) + "\n"


def message_line(system: str, user: str, assistant: str) -> str:
    roles = (("system", system), ("user", user), ("assistant", assistant))
    return json.dumps({"messages": [{"role": r, "content": c} for r, c in roles]})


def test_press_emails_are_allowed_in_release_texts_only(tmp_path):
    contact = f"Investor contact: {PRESS}"
    write(
        tmp_path,
        "dataset/data/releases/validation.jsonl",
        release_line(ticker="AAA", text=contact)
        + release_line(ticker=PRESS, text="no contact")  # not the text field
        + release_line(ticker="BBB", text=f"{contact}, or {KNOWN}"),  # known: never
    )
    write(
        tmp_path,
        "dataset/data/train/train.jsonl",
        message_line(f"system {PRESS}", contact, f"assistant {PRESS}") + "\n",
    )
    write(tmp_path, "space/data/texts/time_test/0000001005-26-000031.md", contact)
    write(tmp_path, "space/data/texts/validation/x.md", f"{contact}\n\n{KNOWN}\n")
    write(tmp_path, "space/data/meta.md", contact)  # not under texts/
    write(tmp_path, "dataset/README.md", f"Questions: {PRESS}\n")
    findings = scan(tmp_path)
    assert [(f.path, f.where, f.kind) for f in findings] == [
        ("dataset/README.md", "line 1", hfscan.EMAIL),
        ("dataset/data/releases/validation.jsonl", "line 2, $.ticker", hfscan.EMAIL),
        (
            "dataset/data/releases/validation.jsonl",
            "line 3, $.text",
            hfscan.KNOWN_EMAIL,
        ),
        (
            "dataset/data/train/train.jsonl",
            "line 1, $.messages[0].content",
            hfscan.EMAIL,
        ),
        (
            "dataset/data/train/train.jsonl",
            "line 1, $.messages[2].content",
            hfscan.EMAIL,
        ),
        ("space/data/meta.md", "line 1", hfscan.EMAIL),
        ("space/data/texts/validation/x.md", "line 3", hfscan.KNOWN_EMAIL),
    ]
    assert_masked(findings, PRESS, KNOWN)


def test_keys_are_never_release_text(tmp_path, monkeypatch):
    whole_file = (hfscan.ReleaseText("data/*.jsonl"),)
    monkeypatch.setattr(hfscan, "RELEASE_TEXTS", whole_file)
    write(tmp_path, "data/x.jsonl", json.dumps({PRESS: f"contact {PRESS}"}) + "\n")
    assert summary(scan(tmp_path)) == [("data/x.jsonl", "line 1, $: key", hfscan.EMAIL)]


def test_release_text_rules():
    rules = hfscan.release_rules("dataset/data/train/train.jsonl")
    assert hfscan.is_release_text(rules, ("messages", 1, "content"), {"role": "user"})
    assert not hfscan.is_release_text(
        rules, ("messages", 1, "content"), {"role": "assistant"}
    )
    assert not hfscan.is_release_text(rules, ("messages", "x", "content"), {})
    assert hfscan.release_rules("space/data/texts/a/b/c.md")
    assert not hfscan.release_rules("space/data/texts.md")
    assert not hfscan.release_rules("dataset/data/releases/x/y.jsonl")
    assert hfscan.glob_match("a/**/b", "a/b") and hfscan.glob_match("a/**/b", "a/x/y/b")


def test_broken_json_is_a_finding_and_still_scanned(tmp_path):
    write(tmp_path, "x.json", '{"contact": "' + KNOWN)
    write(tmp_path, "y.jsonl", '{"ok": 1}\n{"broken": "/home/' + 'frank"\n')
    findings = scan(tmp_path)
    assert summary(findings) == [
        ("x.json", "file", hfscan.UNREADABLE),
        ("x.json", "line 1", hfscan.KNOWN_EMAIL),
        ("y.jsonl", "line 2", hfscan.UNREADABLE),
        ("y.jsonl", "line 2", hfscan.HOME_PATH),
    ]


def test_text_that_isnt_utf8_is_unreadable(tmp_path):
    write(tmp_path, "notes.md", b"caf\xe9")
    write(tmp_path, "rows.jsonl", b'{"a": 1}\n"\xff"\n')
    assert summary(scan(tmp_path)) == [
        ("notes.md", "file", hfscan.UNREADABLE),
        ("rows.jsonl", "line 2", hfscan.UNREADABLE),
    ]


# PNG and safetensors


def chunk(kind: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(kind + body)
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)


def png(*chunks: bytes) -> bytes:
    header = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
    pixels = chunk(b"IDAT", zlib.compress(b"\x00\x00"))
    return (
        hfscan.PNG_SIGNATURE + header + b"".join(chunks) + pixels + chunk(b"IEND", b"")
    )


def test_png_text_chunks_are_scanned(tmp_path):
    token = joined("hf", "_", "Fk3QwErTy" * 4)
    write(
        tmp_path,
        "space/data/chart.png",
        png(
            chunk(
                b"tEXt", b"Software\0Matplotlib version3.11.2, https://matplotlib.org/"
            ),
            chunk(b"tEXt", b"Author\0" + KNOWN.encode()),
            chunk(b"zTXt", b"Comment\0\0" + zlib.compress(f"{HOME}/x".encode())),
            chunk(b"iTXt", b"Description\0\0\0en\0\0" + f"key {token}".encode()),
            chunk(b"iTXt", b"Source\0\1\0\0\0" + zlib.compress(PRESS.encode())),
            chunk(b"tEXt", KNOWN.encode() + b"\0a keyword that is an address"),
            chunk(b"eXIf", b"MM\0*\0\0\0\x08 Artist " + PRESS.encode()),
        ),
    )
    findings = scan(tmp_path)
    assert [(f.where, f.kind) for f in findings] == [
        ("PNG tEXt chunk 2 (Author)", hfscan.KNOWN_EMAIL),
        ("PNG zTXt chunk 3 (Comment)", hfscan.HOME_PATH),
        ("PNG iTXt chunk 4 (Description)", hfscan.API_KEY),
        ("PNG iTXt chunk 5 (Source)", hfscan.EMAIL),
        ("PNG tEXt chunk 6", hfscan.KNOWN_EMAIL),
        ("PNG eXIf chunk 7", hfscan.EMAIL),
    ]
    assert_masked(findings, KNOWN, token, PRESS, "testowner")


def test_the_repositorys_charts_are_clean(tmp_path):
    for name in ("accuracy_vs_cost.png", "test_accuracy_vs_cost.png"):
        shutil.copyfile(REPO / "results" / name, tmp_path / name)
    assert scan(tmp_path) == []


def test_a_broken_png_is_unreadable(tmp_path):
    whole = png(chunk(b"tEXt", b"Author\0nobody"))
    write(tmp_path, "cut.png", whole[:40])
    write(tmp_path, "fake.png", b"GIF89a")
    assert summary(scan(tmp_path)) == [
        ("cut.png", "file", hfscan.UNREADABLE),
        ("fake.png", "file", hfscan.UNREADABLE),
    ]


def safetensors(header: dict) -> bytes:
    data = json.dumps(header).encode()
    return struct.pack("<Q", len(data)) + data + b"\0" * 8


def test_safetensors_headers_are_scanned(tmp_path):
    write(tmp_path, "model/adapter_model.safetensors", hf_fixture.safetensors_bytes())
    write(
        tmp_path,
        "merged/model.safetensors",
        safetensors(
            {
                "__metadata__": {"base_model": hf_fixture.INSTANCE_PATH},
                "w": {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, 8]},
            }
        ),
    )
    write(tmp_path, "short/x.safetensors", struct.pack("<Q", 10**6) + b"{}")
    findings = scan(tmp_path)
    assert summary(findings) == [
        (
            "merged/model.safetensors",
            "safetensors header, $.__metadata__.base_model",
            hfscan.INSTANCE_PATH,
        ),
        ("short/x.safetensors", "file", hfscan.UNREADABLE),
    ]


# Names, types and links


def test_file_and_folder_names_are_scanned_and_masked(tmp_path):
    token = joined("hf", "_", "Fk3QwErTy" * 4)
    write(tmp_path, f"dataset/{KNOWN}/{token}.txt", f"{HOME}\n")
    findings = scan(tmp_path)
    folder, file = "dataset/ow…@ex…", f"dataset/ow…@ex…/{token[:5]}…(39 characters).txt"
    assert summary(findings) == [
        (folder, "folder name", hfscan.KNOWN_EMAIL),
        (file, "file name", hfscan.API_KEY),
        (file, "line 1", hfscan.HOME_PATH),
    ]
    assert_masked(findings, KNOWN, token)


@pytest.mark.parametrize(
    "name, excerpt",
    [
        ("model.bin", ".bin: not read"),
        ("data.parquet", ".parquet: not read"),
        ("LICENSE", "no suffix: not read"),
        (".DS_Store", "no suffix: not read"),
        (".gitattributes", "no suffix: not read"),
    ],
)
def test_other_file_types_are_findings(tmp_path, name, excerpt):
    write(tmp_path, f"space/{name}", "plain")
    findings = scan(tmp_path)
    assert [(f.where, f.kind, f.excerpt) for f in findings] == [
        ("file", hfscan.UNSCANNED, excerpt)
    ]


def test_suffixes_are_read_whatever_their_case(tmp_path):
    write(tmp_path, "CARD.MD", "/home/" + "gina\n")
    assert [f.kind for f in scan(tmp_path)] == [hfscan.HOME_PATH]


@pytest.mark.parametrize("name", [".env", ".env.local", ".env.json", "openai.env"])
def test_env_files_are_never_opened(tmp_path, name):
    path = write(
        tmp_path, f"dataset/{name}", f"MODEL_API_KEY={'sk' + '-' + 'Ab3' * 10}"
    )
    path.chmod(0)  # opening it would be a finding too (unreadable, or the key)
    try:
        findings = scan(tmp_path)
    finally:
        path.chmod(0o600)
    assert [(f.where, f.kind) for f in findings] == [("file", hfscan.ENV_FILE)]


def test_symbolic_links_are_not_followed(tmp_path):
    outside = write(tmp_path / "outside", "secret.md", KNOWN)
    root = tmp_path / "export"
    root.mkdir()
    (root / "card.md").symlink_to(outside)
    (root / "folder").symlink_to(outside.parent, target_is_directory=True)
    findings = scan(root)
    assert [(f.path, f.where, f.kind) for f in findings] == [
        ("folder", "folder", hfscan.UNSCANNED),
        ("card.md", "file", hfscan.UNSCANNED),
    ]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads anything")
def test_folders_and_files_it_cant_open_are_findings(tmp_path):
    locked = tmp_path / "space" / "locked"
    write(locked, "inside.md", KNOWN)
    secret = write(tmp_path, "dataset/README.md", KNOWN)
    locked.chmod(0)
    secret.chmod(0)
    try:
        findings = scan(tmp_path)
    finally:
        locked.chmod(0o700)
        secret.chmod(0o600)
    assert [(f.path, f.where, f.kind) for f in findings] == [
        ("dataset/README.md", "file", hfscan.UNREADABLE),
        ("space/locked", "folder", hfscan.UNREADABLE),
    ]


def test_a_missing_folder_stops_the_scan(tmp_path):
    with pytest.raises(hfscan.ScanError, match="isn't a folder"):
        scan(tmp_path / "missing")


def test_scans_are_deterministic(tmp_path):
    write(tmp_path, "b/x.md", f"{KNOWN}\n{HOME}\n")
    write(tmp_path, "a/y.json", json.dumps({"k": PRESS}))
    first = scan(tmp_path)
    assert first == scan(tmp_path)
    assert [f.path for f in first] == ["a/y.json", "b/x.md", "b/x.md"]


# known_emails and the command


def git(repo: Path, *args: str, **env: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={**os.environ, **env},
    )


@pytest.fixture
def git_repo(tmp_path, monkeypatch) -> Path:
    """A repository whose addresses each come from one place: user.email (local
    config only: no commit uses it), authors (one through .mailmap), a committer,
    a commit on another branch, and pyproject.toml's authors and maintainers."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    for name in ("AUTHOR", "COMMITTER"):
        for part in ("NAME", "EMAIL"):
            monkeypatch.delenv(f"GIT_{name}_{part}", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "Dev")
    git(repo, "config", "user.email", "Dev@Example.org")
    committer = {"GIT_COMMITTER_NAME": "Cy", "GIT_COMMITTER_EMAIL": "cy@example.net"}
    commit = ["commit", "-q", "--allow-empty", "-m"]
    git(repo, *commit, "one", "--author=Ann <ann@example.net>", **committer)
    git(repo, "checkout", "-q", "-b", "side")
    git(repo, *commit, "two", "--author=S <s@example.com>", **committer)
    git(repo, "checkout", "-q", "main")
    (repo / ".mailmap").write_text("Ann <ann@mapped.example.net> <ann@example.net>\n")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "x"\n'
        'authors = [{ name = "O", email = "owner@example.com" }]\n'
        'maintainers = [{ name = "M", email = "maint@example.com" }, { name = "N" }]\n'
    )
    return repo


def test_known_emails_read_git_and_pyproject(git_repo):
    assert hfscan.known_emails(git_repo) == {
        "dev@example.org",  # user.email
        "ann@example.net",  # an author, as recorded
        "ann@mapped.example.net",  # and through .mailmap
        "cy@example.net",  # the committer
        "s@example.com",  # an author on another branch
        "owner@example.com",
        "maint@example.com",
    }


def test_known_emails_need_a_git_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(hfscan.ScanError, match="isn't in a git repository"):
        hfscan.known_emails(plain)


def test_main_exit_codes(git_repo, tmp_path, capsys):
    clean = tmp_path / "clean"
    write(clean, "dataset/README.md", "# A card\n")
    assert hfscan.main([str(clean), "--repo", str(git_repo)]) == 0
    out = capsys.readouterr().out
    assert "Looked for 7 known e-mail addresses" in out
    assert out.endswith("Clean: no findings in 1 file and 1 folder.\n")

    dirty = tmp_path / "dirty"
    write(dirty, "model/README.md", "Built by ann@example.net\n")
    assert hfscan.main([str(dirty), "--repo", str(git_repo)]) == 1
    out = capsys.readouterr().out
    assert "model/README.md: line 1: known e-mail: a…@ex…\n" in out
    assert "1 finding (known e-mail 1) in 1 file or folder, of 1 file" in out
    assert "ann@example.net" not in out

    assert hfscan.main([str(tmp_path / "missing"), "--repo", str(git_repo)]) == 2
    assert hfscan.main([str(clean), "--repo", str(tmp_path)]) == 2
    assert "isn't in a git repository" in capsys.readouterr().err


# The mini repository


def test_the_runs_raw_files_are_caught(hf_repo, tmp_path):
    export = tmp_path / "export"
    for raw in sorted((hf_repo.root / "runs").glob("*/raw")):
        shutil.copytree(raw, export / "runs" / raw.parent.name / "raw")
    findings = hfscan.scan(export)
    raw = sorted(
        p.relative_to(export).as_posix() for p in export.glob("runs/*/raw/*.json")
    )
    assert len(raw) > 10 and sorted({f.path for f in findings}) == raw
    first = [(f.where, f.kind) for f in findings if f.path == raw[0]]
    assert first == [
        ("$.response.id", hfscan.ACCOUNT_ID),
        ("$.response.organization", hfscan.ACCOUNT_ID),
        ("$.response.billing_email", hfscan.EMAIL),
    ]
    assert_masked(findings, hf_fixture.SECRET_ORG, hf_fixture.SECRET_EMAIL)


def test_the_mini_repos_release_texts(hf_context, tmp_path):
    """In the release texts' places, their press contacts are fine; anywhere else
    they are the only findings."""
    allowed, elsewhere = tmp_path / "allowed", tmp_path / "elsewhere"
    for split, rows in hf_context.releases.items():
        hfcommon.write_jsonl(allowed / "dataset/data/releases" / f"{split}.jsonl", rows)
        hfcommon.write_jsonl(elsewhere / "dataset/data" / f"{split}.jsonl", rows)
        for row in rows:
            name = f"{row.release_accession}.md"
            write(allowed / "space/data/texts" / split, name, row.text)
    assert hfscan.scan(allowed, emails=[KNOWN], home=HOME) == []
    findings = hfscan.scan(elsewhere, emails=[KNOWN], home=HOME)
    assert {(f.where.split(", ")[1], f.kind, f.excerpt) for f in findings} == {
        ("$.text", hfscan.EMAIL, "in…@aa…")
    }
    assert len(findings) == sum(len(rows) for rows in hf_context.releases.values())


# The real data (skipped without it)

REAL = hfcommon.Sources(REPO)
needs_real_data = pytest.mark.skipif(
    not (REAL.checkpoint.is_dir() and REAL.splits_dir.is_dir()),
    reason="needs the real checkpoint and splits (not in the repository)",
)


@needs_real_data
def test_real_checkpoint_names_the_instance(tmp_path):
    """The checkpoint's own adapter_config.json and README.md name the GPU
    instance's snapshot path (why the model folder rewrites the config); its
    weights' header and state are clean. Scanned in place, read only."""
    findings = hfscan.scan(REAL.checkpoint, home=str(Path.home()))
    assert {f.kind for f in findings} == {hfscan.INSTANCE_PATH}
    assert {f.path for f in findings} == {"README.md", "adapter_config.json"}
    assert ("adapter_config.json", "$.base_model_name_or_path") in {
        (f.path, f.where) for f in findings
    }
    config = json.loads((REAL.checkpoint / "adapter_config.json").read_text())
    config["base_model_name_or_path"] = "Qwen/Qwen3-8B"
    write(tmp_path, "adapter_config.json", json.dumps(config, indent=2))
    assert hfscan.scan(tmp_path, home=str(Path.home())) == []


@needs_real_data
def test_real_release_texts(tmp_path):
    """Every scored release's text (563, rebuilt as the export holds them): clean in
    the release texts' place; anywhere else, press contacts are the only findings
    (no path, key or ID is read into a release's prose or links)."""
    records = {
        **inputs.release_records(REAL.path("data", "releases")),
        **inputs.release_records(REAL.stage2 / "releases"),
    }
    allowed, elsewhere = tmp_path / "allowed", tmp_path / "elsewhere"
    count = 0
    for split in hfcommon.SPLIT_NAMES:
        for row in hfcommon.read_csv(REAL.splits_dir / split / "tokens.csv"):
            accession = row["release_accession"]
            text = inputs.build_input(records[accession]).text
            write(allowed / "space/data/texts" / split, f"{accession}.md", text)
            write(elsewhere / "space/data" / split, f"{accession}.md", text)
            count += 1
    assert count == 563
    emails, home = hfscan.known_emails(REPO), str(Path.home())
    assert hfscan.scan(allowed, emails=emails, home=home) == []
    findings = hfscan.scan(elsewhere, emails=emails, home=home)
    assert {f.kind for f in findings} == {hfscan.EMAIL}
