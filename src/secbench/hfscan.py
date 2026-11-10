"""The privacy scan of the Hugging Face export (secbench.hfexport writes dist/hf/):
every file and folder name, and every file's content, checked for what must never
be published. Why: the export is built next to private files (runs/*/raw/ holds API
responses with account details; the adapter's own config named the GPU instance's
paths), and nothing can be taken back once a repository is public.

Findings, by kind:
- known e-mail: the repository owner's addresses (git's user.email, every author
  and committer in the history, pyproject.toml's authors and maintainers), anywhere,
  release texts included. known_emails reads them when the scan runs, so they are
  never written into code;
- e-mail: any other address, except inside the release texts (RELEASE_TEXTS), whose
  press releases print press and investor contacts;
- home path: the home folder, /Users/<name>, /home/<name>, /root/, C:\\Users\\<name>;
  instance path: /workspace (a GPU instance's); temporary path: /tmp/, /private/tmp/,
  /var/folders/, /private/var/ (a build's or a test's temporary folder). A path in a
  URL (https://ir.example.com/home/default.aspx) is none of these;
- API key or token (hf_, sk-, sk-ant-, ghp_ and GitHub's other prefixes,
  github_pat_, AKIA and ASIA, xoxb- and Slack's others, Bearer), private key (a PEM
  block), account or request ID (org-, proj_, user-, req_, msg_, chatcmpl-). Their
  variable part must look random (a digit, or upper and lower case), so prose such as
  "user-friendly" isn't one;
- env file (.env, .env.*, *.env: never opened), unscanned file type (any type not
  read below, a symbolic link included) and unreadable file (not UTF-8, invalid
  JSON, a broken PNG or safetensors header, or a file or folder the scan can't
  open): nothing is published unread.

What is read: text files (TEXT_SUFFIXES) as UTF-8; .json and .jsonl files string by
string, keys included, each with its JSON path (so a JSON escape can't hide
anything); a PNG's text chunks (tEXt, zTXt, iTXt, and eXIf); a .safetensors file's
JSON header. A string that holds a backslash escape as code writes one (\\n, \\t,
\\u2014: ESCAPE) is read twice, as it is and with each escape as spaces of the same
length, so a path or token right after an escape ("a\\n/workspace/..." in a .py
file, or in a JSON string that holds code) is found as it would be after a space.
Each finding's excerpt is masked (a fixed prefix and a character or two, then "…"),
and so is any secret in the file names and JSON keys it names: the scan never prints
a whole secret.

Usage: uv run python -m secbench.hfscan DIR [--repo .]
       (prints each finding and a summary; exit 1 on any finding, 0 when clean)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import stat
import struct
import subprocess
import sys
import tomllib
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The file types read (lowercase suffixes); any other file is a finding.
TEXT_SUFFIXES = frozenset(
    {
        ".md",
        ".json",
        ".jsonl",
        ".txt",
        ".py",
        ".csv",
        ".toml",
        ".yaml",
        ".yml",
        ".sh",
        ".html",
        ".css",
        ".js",
    }
)
PNG_SUFFIX, SAFETENSORS_SUFFIX = ".png", ".safetensors"
# Settings files, which hold secrets: never opened, a finding wherever they are.
ENV_FILES = (".env", ".env.*", "*.env")


@dataclass(frozen=True)
class ReleaseText:
    """Where release texts are: files (a glob relative to the scanned folder; "*" any
    name, "**" any folders), and the JSON strings in them (a path of keys, "*" any
    list item; None: the whole file), only in an object that also holds `when`
    (key, value)."""

    files: str
    field: tuple[str, ...] | None = None
    when: tuple[str, str] | None = None


# The press releases as filed, which print press and investor contacts: other
# e-mail addresses are allowed there and nowhere else (a known e-mail never is).
RELEASE_TEXTS = (
    ReleaseText("dataset/data/releases/*.jsonl", ("text",)),
    ReleaseText(
        "dataset/data/train/*.jsonl", ("messages", "*", "content"), ("role", "user")
    ),
    ReleaseText("space/data/texts/**/*.md"),
    # The explorer's page for each release prints its text (and lines of it); the
    # rest of the page comes from data/'s JSON, which hfspace checks holds none.
    ReleaseText("space/release/*.html"),
)

KNOWN_EMAIL = "known e-mail"
EMAIL = "e-mail"
HOME_PATH = "home path"
INSTANCE_PATH = "instance path"
TEMP_PATH = "temporary path"
API_KEY = "API key or token"
PRIVATE_KEY = "private key"
ACCOUNT_ID = "account or request ID"
ENV_FILE = "env file"
UNSCANNED = "unscanned file type"
UNREADABLE = "unreadable file"
KINDS = (
    KNOWN_EMAIL,
    EMAIL,
    HOME_PATH,
    INSTANCE_PATH,
    TEMP_PATH,
    API_KEY,
    PRIVATE_KEY,
    ACCOUNT_ID,
    ENV_FILE,
    UNSCANNED,
    UNREADABLE,
)

# How an excerpt is masked: a token shows its prefix, a character or two and its
# length; a path its prefix and a character or two of the name; a PEM block its
# header line only (no key material).
TOKEN, PATH, HEADER = "token", "path", "header"
# The characters of a URL, a host name or a longer path: a path right after one of
# them is part of a URL (https://ir.example.com/home/...), not a path on a machine.
URL_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~/%-"
)
WORD_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
)
LOCAL_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._%+-"
)
DOMAIN = re.compile(r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
MAX_LOCAL = 64  # an address's local part, at most (RFC 5321)
# A backslash escape as code (Python, JavaScript, JSON in a text file) writes one:
# \n and the other single letters, octal, \x41, \u2014, \U0001F600, \u{1F600},
# \N{EM DASH}. It stands for one character, often a separator, while its last
# character would join a path to a URL or a token to a word.
ESCAPE = re.compile(
    r"\\(?:[abefnrtv]|[0-7]{1,3}|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}"
    r"|u\{[0-9A-Fa-f]{1,6}\}|N\{[A-Za-z0-9 -]{1,80}\})"
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_TEXT_CHUNKS = (b"tEXt", b"zTXt", b"iTXt", b"eXIf")
# PNG's registered keywords, shown in a finding's location (any other keyword could
# be what was found, and isn't shown).
PNG_KEYWORDS = frozenset(
    {
        "Title",
        "Author",
        "Description",
        "Copyright",
        "Creation Time",
        "Software",
        "Disclaimer",
        "Warning",
        "Source",
        "Comment",
    }
)
MAX_INFLATED = 1 << 24  # a compressed PNG text chunk, at most (16 MiB)
MAX_HEADER = 100_000_000  # a safetensors header, at most (the format's limit)
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ScanError(Exception):
    """The scan can't run; the message says why."""


@dataclass(frozen=True)
class Finding:
    """Something that must not be published: the file or folder (relative to the
    scanned folder, any secret in its names masked), where in it, what, and a
    masked excerpt."""

    path: str
    where: str  # "file name", "line 12", "line 3, $.text", "PNG tEXt chunk 1", ...
    kind: str
    excerpt: str  # never the whole secret

    def __str__(self) -> str:
        return f"{self.path}: {self.where}: {self.kind}: {self.excerpt}"


@dataclass(frozen=True)
class Detector:
    """One kind of secret: a pattern with a "prefix" group, shown in the excerpt,
    and a "secret" group, masked; its anchors are fixed texts every match holds (a
    string holding none of them is skipped quickly)."""

    kind: str
    pattern: re.Pattern[str]
    anchors: tuple[str, ...]
    style: str = TOKEN
    after_word: bool = False  # may follow a letter, digit or "_" (else: never)
    outside_urls: bool = False  # paths: never part of a URL or a longer path
    random: bool = True  # tokens and IDs: the secret has a digit, or both cases


def prefixed(prefix: str, secret: str = "[A-Za-z0-9]{20,}") -> re.Pattern[str]:
    return re.compile(f"(?P<prefix>{prefix})(?P<secret>{secret})")


NAME = "[A-Za-z0-9._-]+"  # a user's or a folder's name


def machine_path(kind: str, folder: str, name: str = NAME) -> Detector:
    """A path on a machine: a fixed folder, then a name (masked)."""
    return Detector(
        kind,
        prefixed(re.escape(folder), name),
        (folder,),
        style=PATH,
        after_word=True,
        outside_urls=True,
        random=False,
    )


DETECTORS = (
    Detector(API_KEY, prefixed("hf_(?:oauth_)?", "[A-Za-z0-9]{30,}"), ("hf_",)),
    Detector(
        API_KEY,
        prefixed(
            r"sk-(?:ant-(?:api\d\d-|admin\d\d-)?|proj-|svcacct-|admin-)?",
            "[A-Za-z0-9_-]{20,}",
        ),
        ("sk-",),
    ),
    Detector(API_KEY, prefixed("gh[pousr]_", "[A-Za-z0-9]{30,}"), ("gh",)),
    Detector(API_KEY, prefixed("github_pat_", "[A-Za-z0-9_]{20,}"), ("github_pat_",)),
    Detector(
        API_KEY,
        prefixed("AKIA|ASIA", "[0-9A-Z]{16}(?![0-9A-Za-z])"),
        ("AKIA", "ASIA"),
        random=False,  # their own pattern: uppercase and digits, exactly 16
    ),
    Detector(API_KEY, prefixed("xox[abposr]-", "[A-Za-z0-9-]{10,}"), ("xox",)),
    Detector(API_KEY, prefixed(r"[Bb]earer\s+", "[A-Za-z0-9._~+/=-]{16,}"), ("earer",)),
    Detector(
        PRIVATE_KEY,
        re.compile("(?P<prefix>-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----)"),
        ("PRIVATE KEY",),
        style=HEADER,
        after_word=True,
        random=False,
    ),
    Detector(ACCOUNT_ID, prefixed("org-"), ("org-",)),
    Detector(ACCOUNT_ID, prefixed("proj_"), ("proj_",)),
    Detector(ACCOUNT_ID, prefixed("user-"), ("user-",)),
    Detector(ACCOUNT_ID, prefixed("req_"), ("req_",)),
    Detector(ACCOUNT_ID, prefixed("msg_"), ("msg_",)),
    Detector(ACCOUNT_ID, prefixed("chatcmpl-", "[A-Za-z0-9_-]{8,}"), ("chatcmpl-",)),
    machine_path(HOME_PATH, "/Users/"),
    machine_path(HOME_PATH, "/home/"),
    machine_path(HOME_PATH, "/root/", "[A-Za-z0-9._-]*"),
    Detector(
        HOME_PATH,
        prefixed(r"[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}", "[A-Za-z0-9._ -]+"),
        ("Users",),
        style=PATH,
        random=False,
    ),
    machine_path(INSTANCE_PATH, "/workspace", "(?:/[A-Za-z0-9._-]*)?(?![\\w.-])"),
    machine_path(TEMP_PATH, "/tmp/"),
    machine_path(TEMP_PATH, "/private/tmp/"),
    machine_path(TEMP_PATH, "/var/folders/"),
    machine_path(TEMP_PATH, "/private/var/"),
)


@dataclass(frozen=True)
class Hit:
    """A secret found in a string: where it starts and ends, its kind and its
    masked excerpt."""

    start: int
    end: int
    kind: str
    excerpt: str


# Masking


def keep(text: str, most: int = 2) -> str:
    """At most `most` leading characters of text, and less than half of it."""
    return text[: min(most, len(text) // 2)]


def mask_token(prefix: str, secret: str) -> str:
    whole = len(prefix) + len(secret)
    return f"{prefix}{secret[: min(2, len(secret) // 4)]}…({whole} characters)"


def mask_path(prefix: str, name: str) -> str:
    return f"{prefix}{keep(name)}…" if name else prefix


def mask_email(address: str) -> str:
    local, _, domain = address.partition("@")
    return f"{keep(local)}…@{keep(domain)}…"


def part(text: str, match: re.Match[str], group: str) -> str:
    """A group of a match found in text, or in a copy of it with the same offsets
    (blank_escapes), as text has it."""
    return text[match.start(group) : match.end(group)]


def masked(detector: Detector, text: str, match: re.Match[str]) -> str:
    prefix = re.sub(r"\s+", " ", part(text, match, "prefix"))
    if detector.style == HEADER:
        return prefix
    if detector.style == PATH:
        return mask_path(prefix, part(text, match, "secret"))
    return mask_token(prefix, part(text, match, "secret"))


def blank_escapes(text: str) -> str:
    """text with each backslash escape (ESCAPE) replaced by as many spaces: the same
    offsets, so line numbers and excerpts don't move, and a boundary where each
    escape was."""
    return ESCAPE.sub(lambda m: " " * len(m[0]), text)


def looks_random(text: str) -> bool:
    return any(c.isdigit() for c in text) or (
        any(c.isupper() for c in text) and any(c.islower() for c in text)
    )


def in_url(text: str, start: int) -> bool:
    """Whether a path found at start is part of a URL, a host name or a longer path
    (the path of a file:// URL is a machine's)."""
    if text[max(0, start - 7) : start].lower() == "file://":
        return False
    return start > 0 and text[start - 1] in URL_CHARS


# Where release texts are


def glob_match(pattern: str, relative: str) -> bool:
    """Whether a relative POSIX path matches a glob ("*" within a name, "**" any
    number of folders)."""

    def match(parts: list[str], names: list[str]) -> bool:
        if not parts:
            return not names
        if parts[0] == "**":
            return any(match(parts[1:], names[i:]) for i in range(len(names) + 1))
        return (
            bool(names)
            and fnmatch.fnmatchcase(names[0], parts[0])
            and match(parts[1:], names[1:])
        )

    return match(pattern.split("/"), relative.split("/"))


def release_rules(relative: str) -> list[ReleaseText]:
    return [rule for rule in RELEASE_TEXTS if glob_match(rule.files, relative)]


def is_release_text(
    rules: Iterable[ReleaseText], at: tuple[str | int, ...], parent: Any
) -> bool:
    """Whether the JSON string at path `at`, held by `parent`, is a release text by
    one of its file's rules."""
    for rule in rules:
        if rule.field is None:
            return True
        if len(rule.field) != len(at) or not all(
            isinstance(part, int) if want == "*" else part == want
            for want, part in zip(rule.field, at, strict=True)
        ):
            continue
        if rule.when is None or (
            isinstance(parent, dict) and parent.get(rule.when[0]) == rule.when[1]
        ):
            return True
    return False


# Reading files


def walk_json(
    value: Any, at: tuple[str | int, ...] = (), parent: Any = None
) -> Iterator[tuple[tuple[str | int, ...], str, Any, bool]]:
    """Every string in a JSON value, keys included: (its path, the string, the
    object or list holding it, whether it is a key; a key's path is its object's)."""
    if isinstance(value, str):
        yield at, value, parent, False
    elif isinstance(value, dict):
        for key, item in value.items():
            yield at, key, value, True
            yield from walk_json(item, (*at, key), value)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from walk_json(item, (*at, i), value)


def inflate(data: bytes) -> bytes:
    inflater = zlib.decompressobj()
    out = inflater.decompress(data, MAX_INFLATED)
    if inflater.unconsumed_tail:
        raise ValueError(f"a compressed text chunk over {MAX_INFLATED} bytes")
    return out


def png_texts(data: bytes) -> Iterator[tuple[str, str, str]]:
    """(chunk type, keyword, text) of each text chunk of a PNG file, in order
    (eXIf: no keyword, its bytes as Latin-1)."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file")
    at = len(PNG_SIGNATURE)
    while at + 12 <= len(data):
        length, kind = struct.unpack(">I4s", data[at : at + 8])
        if at + 12 + length > len(data):
            break
        body = data[at + 8 : at + 8 + length]
        at += 12 + length
        if kind == b"IEND":
            return
        if kind not in PNG_TEXT_CHUNKS:
            continue
        name = kind.decode("ascii")
        if kind == b"eXIf":
            yield name, "", body.decode("latin-1")
            continue
        keyword, _, rest = body.partition(b"\0")
        if kind == b"tEXt":
            text = rest.decode("latin-1")
        elif kind == b"zTXt":  # a compression method byte, then zlib
            text = inflate(rest[1:]).decode("latin-1")
        else:  # iTXt: compression flag and method, language, translated keyword
            if len(rest) < 2:
                raise ValueError("a truncated iTXt chunk")
            compressed, rest = rest[0], rest[2:]
            language, _, rest = rest.partition(b"\0")
            translated, _, value = rest.partition(b"\0")
            if compressed:
                value = inflate(value)
            text = "\n".join(
                (language.decode("latin-1"), translated.decode(), value.decode())
            )
        yield name, keyword.decode("latin-1"), text
    raise ValueError("truncated: no IEND chunk")


def safetensors_header(file: Path) -> Any:
    """A .safetensors file's header: an 8-byte little-endian length, then that many
    bytes of JSON."""
    size = file.stat().st_size
    with file.open("rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise ValueError("shorter than the 8 bytes of its header's length")
        (length,) = struct.unpack("<Q", head)
        if length > min(size - 8, MAX_HEADER):
            raise ValueError(f"header length {length}, the file has {size - 8} bytes")
        return json.loads(f.read(length).decode("utf-8"))


def is_env_file(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in ENV_FILES)


def plural(n: int, word: str, many: str = "") -> str:
    return f"{n} {word}" if n == 1 else f"{n} {many or word + 's'}"


# The scan


class Scanner:
    """One folder's scan: run() returns the findings and counts the files and
    folders it read."""

    def __init__(self, root: Path, emails: Iterable[str], home: str | None) -> None:
        self.root = root
        self.emails = {e.strip().lower() for e in emails if "@" in e}
        self.known = (
            re.compile("|".join(map(re.escape, sorted(self.emails))), re.IGNORECASE)
            if self.emails
            else None
        )
        self.home, self.home_name = self.home_pattern(home)
        self.findings: list[Finding] = []
        self.seen: set[Finding] = set()
        self.files = 0
        self.folders = 0

    @staticmethod
    def home_pattern(home: str | None) -> tuple[re.Pattern[str] | None, str]:
        """The home folder as a pattern, with or without its leading separator and
        with either separator (/Users/name, Users\\name), and its last name (a string
        without it is skipped quickly)."""
        names = [n for n in re.split(r"[/\\]+", home or "") if n]
        if len(names) < 2:  # "/" or "/root": the generic patterns find these
            return None, ""
        head = "".join(re.escape(n) + r"[/\\]+" for n in names[:-1])
        pattern = re.compile(
            rf"(?<![\w.-])(?P<prefix>[/\\]?{head})"
            rf"(?P<secret>{re.escape(names[-1])})(?![\w.-])"
        )
        return pattern, names[-1]

    # Strings

    def hits(self, text: str, release_text: bool = False) -> list[Hit]:
        """What text holds, in order. Other e-mail addresses than the known ones
        are allowed in a release text. A text with a backslash escape is read
        twice: as it is, and with its escapes as spaces (blank_escapes), so a
        secret right after an escape is found too."""
        found = self.find(text, text, release_text)
        if "\\" in text:
            view = blank_escapes(text)
            if view != text:
                found += self.find(view, text, release_text)
        # In order; a hit inside an earlier one of its kind is the same secret
        # (c:/Users/name is also /Users/name; a secret both readings find).
        kept, reach = [], {}
        for hit in sorted(found, key=lambda h: (h.start, -h.end, h.kind, h.excerpt)):
            if hit.end > reach.get(hit.kind, -1):
                kept.append(hit)
                reach[hit.kind] = hit.end
        return kept

    def find(self, view: str, text: str, release_text: bool) -> list[Hit]:
        """The hits in view, which is text or a copy of it with the same offsets
        (blank_escapes); the excerpts are text's."""
        found = []
        for detector in DETECTORS:
            if not any(anchor in view for anchor in detector.anchors):
                continue
            for match in detector.pattern.finditer(view):
                start = match.start()
                if not detector.after_word and start and view[start - 1] in WORD_CHARS:
                    continue
                if detector.outside_urls and in_url(view, start):
                    continue
                if detector.random and not looks_random(part(text, match, "secret")):
                    continue
                excerpt = masked(detector, text, match)
                found.append(Hit(start, match.end(), detector.kind, excerpt))
        if self.home is not None and self.home_name in view:
            for match in self.home.finditer(view):
                prefix, name = part(text, match, "prefix"), part(text, match, "secret")
                excerpt = mask_path(prefix, name)
                found.append(Hit(match.start(), match.end(), HOME_PATH, excerpt))
        if "@" in view:
            found += self.addresses(view, text, release_text)
        return found

    def addresses(self, view: str, text: str, release_text: bool) -> Iterator[Hit]:
        """The known e-mail addresses in view (text, or a copy with the same
        offsets), then (outside a release text) the others: each "@" with a local
        part before it and a domain after it. Addresses are shown as text has them."""
        if self.known is not None:
            lower = view.lower()
            if any(email in lower for email in self.emails):
                for match in self.known.finditer(view):
                    excerpt = mask_email(text[match.start() : match.end()])
                    yield Hit(match.start(), match.end(), KNOWN_EMAIL, excerpt)
        if release_text:
            return
        at = view.find("@")
        while at != -1:
            start = at
            while start and at - start < MAX_LOCAL and view[start - 1] in LOCAL_CHARS:
                start -= 1
            domain = DOMAIN.match(view, at + 1)
            if domain and start < at:
                address = text[start : domain.end()]
                if address.lower() not in self.emails:
                    yield Hit(start, domain.end(), EMAIL, mask_email(address))
            at = view.find("@", at + 1)

    def redact(self, text: str) -> str:
        """text with each secret in it replaced by its masked excerpt."""
        out, at = [], 0
        for hit in self.hits(text):
            if hit.start >= at:
                out += [text[at : hit.start], hit.excerpt]
                at = hit.end
        return "".join(out) + text[at:]

    # Findings

    def add(self, relative: str, where: str, kind: str, excerpt: str) -> None:
        finding = Finding(relative, where, kind, excerpt)
        if finding not in self.seen:
            self.seen.add(finding)
            self.findings.append(finding)

    def add_hits(
        self, relative: str, where: str, text: str, release_text: bool = False
    ) -> None:
        for hit in self.hits(text, release_text):
            self.add(relative, where, hit.kind, hit.excerpt)

    def add_text(self, relative: str, text: str, release_text: bool = False) -> None:
        """Hits in a text file's text, each located by its line."""
        for hit in self.hits(text, release_text):
            where = f"line {text.count(chr(10), 0, hit.start) + 1}"
            self.add(relative, where, hit.kind, hit.excerpt)

    def add_value(
        self, relative: str, where: str, value: Any, rules: list[ReleaseText]
    ) -> None:
        """Hits in every string of a JSON value, each located by its JSON path."""
        for at, text, parent, is_key in walk_json(value):
            release_text = not is_key and is_release_text(rules, at, parent)
            hits = self.hits(text, release_text)
            if hits:
                location = where + self.json_path(at) + (": key" if is_key else "")
                for hit in hits:
                    self.add(relative, location, hit.kind, hit.excerpt)

    def json_path(self, at: tuple[str | int, ...]) -> str:
        out = "$"
        for part in at:
            if isinstance(part, int):
                out += f"[{part}]"
                continue
            shown = self.redact(part)
            if IDENTIFIER.fullmatch(shown):
                out += f".{shown}"
            else:
                out += f"[{json.dumps(shown, ensure_ascii=False)}]"
        return out

    # Folders and files

    def run(self) -> list[Finding]:
        if not self.root.is_dir():
            raise ScanError(f"{self.root} isn't a folder")
        for top, folders, files in os.walk(self.root, onerror=self.unlisted):
            folders.sort()
            files.sort()
            for name in list(folders):
                entry = Path(top, name)
                relative = self.shown(entry)
                self.add_hits(relative, "folder name", name)
                if entry.is_symlink():  # os.walk doesn't enter it
                    folders.remove(name)
                    self.add(relative, "folder", UNSCANNED, "symbolic link: not read")
                else:
                    self.folders += 1
            for name in files:
                entry = Path(top, name)
                relative = self.shown(entry)
                self.files += 1
                self.add_hits(relative, "file name", name)
                self.scan_file(entry, relative)
        return self.findings

    def unlisted(self, error: OSError) -> None:
        """A folder os.walk can't list (its files can't be scanned)."""
        folder = Path(error.filename or self.root)
        relative = self.shown(folder) if folder != self.root else "."
        self.add(relative, "folder", UNREADABLE, f"can't be listed ({error.strerror})")

    def shown(self, entry: Path) -> str:
        """A path relative to the scanned folder, as findings show it: any secret
        in its names masked."""
        return "/".join(self.redact(n) for n in entry.relative_to(self.root).parts)

    def scan_file(self, file: Path, relative: str) -> None:
        try:
            self.read_file(file, relative)
        except OSError as exc:
            self.add(relative, "file", UNREADABLE, f"can't be read ({exc.strerror})")

    def read_file(self, file: Path, relative: str) -> None:
        mode = os.lstat(file).st_mode
        suffix = file.suffix.lower()
        if stat.S_ISLNK(mode):
            self.add(relative, "file", UNSCANNED, "symbolic link: not read")
        elif not stat.S_ISREG(mode):
            self.add(relative, "file", UNSCANNED, "not a regular file: not read")
        elif is_env_file(file.name):
            self.add(relative, "file", ENV_FILE, "never opened")
        elif suffix == ".jsonl":
            self.scan_jsonl(file, relative)
        elif suffix == ".json":
            self.scan_json(file, relative)
        elif suffix in TEXT_SUFFIXES:
            self.scan_text(file, relative)
        elif suffix == PNG_SUFFIX:
            self.scan_png(file, relative)
        elif suffix == SAFETENSORS_SUFFIX:
            self.scan_safetensors(file, relative)
        else:
            self.add(relative, "file", UNSCANNED, f"{suffix or 'no suffix'}: not read")

    def decode(self, data: bytes, relative: str, where: str) -> str | None:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            self.add(relative, where, UNREADABLE, f"not UTF-8 (byte {exc.start})")
            return None

    def scan_text(self, file: Path, relative: str) -> None:
        text = self.decode(file.read_bytes(), relative, "file")
        if text is not None:
            whole = any(rule.field is None for rule in release_rules(relative))
            self.add_text(relative, text, release_text=whole)

    def scan_json(self, file: Path, relative: str) -> None:
        text = self.decode(file.read_bytes(), relative, "file")
        if text is None:
            return
        try:
            value = json.loads(text)
        except ValueError as exc:
            self.add(relative, "file", UNREADABLE, f"invalid JSON ({exc})")
            self.add_text(relative, text)
            return
        self.add_value(relative, "", value, release_rules(relative))

    def scan_jsonl(self, file: Path, relative: str) -> None:
        rules = release_rules(relative)
        with file.open("rb") as f:
            for number, data in enumerate(f, 1):
                where = f"line {number}"
                line = self.decode(data, relative, where)
                if line is None or not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except ValueError as exc:
                    self.add(relative, where, UNREADABLE, f"invalid JSON ({exc})")
                    self.add_hits(relative, where, line)
                    continue
                self.add_value(relative, f"{where}, ", value, rules)

    def scan_png(self, file: Path, relative: str) -> None:
        try:
            chunks = list(png_texts(file.read_bytes()))
        except (ValueError, UnicodeDecodeError, zlib.error) as exc:
            self.add(relative, "file", UNREADABLE, f"PNG: {exc}")
            return
        for number, (kind, keyword, text) in enumerate(chunks, 1):
            where = f"PNG {kind} chunk {number}"
            if keyword in PNG_KEYWORDS:
                where += f" ({keyword})"
            self.add_hits(relative, where, f"{keyword}\n{text}")

    def scan_safetensors(self, file: Path, relative: str) -> None:
        try:
            header = safetensors_header(file)
        except (ValueError, UnicodeDecodeError) as exc:
            self.add(relative, "file", UNREADABLE, f"safetensors header: {exc}")
            return
        self.add_value(relative, "safetensors header, ", header, [])


def scan(
    root: Path, *, emails: Iterable[str] = (), home: str | None = None
) -> list[Finding]:
    """Everything under root (the export's folder, dist/hf) that must not be
    published, in a fixed order: folders top-down, names sorted. emails: the known
    addresses (known_emails); home: the home folder (str(Path.home()))."""
    return Scanner(root, emails, home).run()


def known_emails(repo: Path) -> set[str]:
    """The repository owner's e-mail addresses, lowercase, read now (never written
    into code): git's user.email, every author's and committer's in the history
    (every ref, as recorded and through .mailmap) and pyproject.toml's authors' and
    maintainers'."""

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise ScanError(f"git can't run: {exc}") from None

    if git("rev-parse", "--is-inside-work-tree").returncode != 0:
        raise ScanError(f"{repo} isn't in a git repository")
    found = git("config", "user.email").stdout.split()  # exit 1 when not set
    log = git("log", "--all", "--format=%ae%n%aE%n%ce%n%cE")
    if log.returncode != 0:
        raise ScanError(f"git log failed in {repo}: {log.stderr.strip()}")
    found += log.stdout.split()
    pyproject = repo / "pyproject.toml"
    if pyproject.is_file():
        project = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project")
        project = project or {}
        for person in [*project.get("authors", []), *project.get("maintainers", [])]:
            found.append(str(person.get("email", "")))
    return {email.strip().lower() for email in found if "@" in email}


def check(root: Path, repo: Path = Path(".")) -> list[Finding]:
    """The scan as main runs it: repo's known e-mail addresses and this machine's
    home folder."""
    return scan(root, emails=known_emails(repo), home=str(Path.home()))


def summary(findings: list[Finding], files: int, folders: int) -> str:
    scanned = f"{plural(files, 'file')} and {plural(folders, 'folder')}"
    if not findings:
        return f"Clean: no findings in {scanned}."
    kinds = Counter(f.kind for f in findings)
    by_kind = ", ".join(f"{kind} {kinds[kind]}" for kind in KINDS if kinds[kind])
    places = len({f.path for f in findings})
    return (
        f"{plural(len(findings), 'finding')} ({by_kind}) in "
        f"{plural(places, 'file or folder', 'files or folders')}, of {scanned}: "
        "nothing may be published until they are gone."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m secbench.hfscan",
        description="Scan a folder (the Hugging Face export, dist/hf) for what must "
        "not be published: e-mail addresses, home and instance paths, keys, tokens "
        "and account IDs. Exit 1 on any finding.",
    )
    parser.add_argument("folder", type=Path, metavar="DIR")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("."),
        help="the git repository whose owner's e-mail addresses are looked for "
        "(default: the current folder)",
    )
    args = parser.parse_args(argv)
    try:
        emails = known_emails(args.repo)
        scanner = Scanner(args.folder, emails, str(Path.home()))
        findings = scanner.run()
    except ScanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for finding in findings:
        print(finding)
    print(
        f"Looked for {plural(len(emails), 'known e-mail address', 'known e-mail addresses')} (git and "
        "pyproject.toml), the home folder, other e-mail addresses outside release "
        "texts, home and instance paths, keys, tokens and account IDs."
    )
    print(summary(findings, scanner.files, scanner.folders))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
