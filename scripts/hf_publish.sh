#!/usr/bin/env bash
# Publishes the Hugging Face export (dist/hf/, built by secbench.hfexport) to the Hub:
# the LoRA adapter's model repo, the int4-awq model's card (README.md only: the model's
# own files must already be in its repo, uploaded from the GPU instance), the dataset
# and the results explorer Space. Run it from the repository root, after reading the
# files and a dry run:
#
#   uv run python -m secbench.hfexport --hf-user NAME   # builds dist/hf/
#   bash scripts/hf_publish.sh --dry-run                # what would be sent: no network
#   HF_TOKEN=... bash scripts/hf_publish.sh             # creates the repos and uploads
#
# The dry run needs no token and sends nothing. For each repo in dist/hf/manifest.json
# it prints the repo's id, type and visibility, its folder, every file it would upload
# with its size, and totals. It stops (exit 1) when a folder doesn't match the
# manifest: a file missing, another size, or a file the manifest doesn't list (hf
# upload sends a whole folder, so that file would go up too).
#
# Without --dry-run it refuses unless HF_TOKEN is set and not blank (the token is read
# from that variable only and never printed; huggingface_hub's stored login is pointed
# at a file that doesn't exist, so a blank token can't fall back to it) and the export
# names a real user, not the placeholder HF_USER (rebuild with --hf-user NAME). Then,
# in order: secbench.hfexport --verify (every file's size and sha256 as the manifest
# lists them, none missing or extra, and the privacy scan clean); `hf auth whoami`
# must name the manifest's user as the token's user or one of its organizations; each
# repo that doesn't exist is created private, except the int4-awq one, which must
# exist already and hold its model's config.json and *.safetensors; then every repo's
# visibility is read from the Hub. An existing repo keeps its visibility, so if one is
# public the script stops before any upload (its files would be published at once),
# unless --allow-public is given. Then each folder is uploaded, with a commit message
# naming the build's commit, and the end lists each repo's visibility as the Hub
# reported it. Running it again is safe: an upload of unchanged files makes no
# commit. The script never makes a repo public: do that on the Hub, in each repo's
# settings, after a look at it there.
#
# Usage: bash scripts/hf_publish.sh [--dry-run] [--dist dist/hf] [--allow-public]
set -euo pipefail
unset CDPATH  # so cd prints nothing

fail() { echo "error: $*" >&2; exit 1; }
step() { printf '\n== %s\n' "$*"; }
USAGE="usage: bash scripts/hf_publish.sh [--dry-run] [--dist dist/hf] [--allow-public]"

DRY_RUN=0
ALLOW_PUBLIC=0
DIST_ARG=dist/hf
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --allow-public) ALLOW_PUBLIC=1; shift ;;
        --dist)
            [ $# -ge 2 ] && [ -n "$2" ] || fail "--dist needs a folder ($USAGE)"
            DIST_ARG=$2
            shift 2
            ;;
        -h | --help)
            sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;s/^#$//;p;}' "$0"
            exit 0
            ;;
        *) fail "unknown argument $1 ($USAGE)" ;;
    esac
done
[ -f "$DIST_ARG/manifest.json" ] ||
    fail "$DIST_ARG/manifest.json not found: build it with uv run python -m secbench.hfexport --hf-user NAME"
DIST=$(cd "$DIST_ARG" && pwd)
cd "$(dirname "$0")/.."  # uv runs the repository's project

# The manifest, read by Python (no jq). `manifest show`: the dry run's listing, with
# the folders checked against the manifest; `manifest plan`: one tab-separated line
# for the user, one for the commit message and one per repo; `manifest whoami USER
# JSON`: whether `hf auth whoami --format json` names the manifest's user; `manifest
# repo ID CARD_ONLY JSON`: the repo's visibility from `hf <type>s info --format json`.
manifest() {
    uv run python - "$DIST" "$DIST_ARG" "$@" <<'EOF'
import json
import sys
from pathlib import Path, PurePosixPath

dist, shown, command, *args = sys.argv[1:]
dist = Path(dist)
UNITS = (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10))


def fail(message):
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def whoami(user, answer):
    """Whether hf auth whoami's answer names the user, or an organization of it."""
    try:
        account = json.loads(answer)
        name = account["user"]
    except (ValueError, KeyError, TypeError):
        fail("hf auth whoami gave no user name")
    orgs = account.get("orgs") or []
    if isinstance(orgs, str):
        orgs = orgs.split(",")
    if user != name and user not in orgs:
        fail(
            f"the token is {name}'s and the export is for {user}: use a token of "
            f"{user}, or rebuild with --hf-user {name}"
        )
    print(f"Logged in as {name}" + ("" if user == name else f", a member of {user}"))
    return 0


def repo(repo_id, card_only, answer):
    """The repo's visibility as the Hub reports it, "private" or "public". A card-only
    repo (int4-awq) must already hold its model, as secbench.hfcards requires before
    it writes the card: config.json and *.safetensors at the top, where vLLM reads
    them."""
    try:
        info = json.loads(answer)
        private = info["private"]
        files = [sibling["rfilename"] for sibling in info.get("siblings") or []]
    except (ValueError, KeyError, TypeError, AttributeError):
        fail(f"hf gave no visibility for {repo_id}")
    if not isinstance(private, bool):
        fail(f"hf gave no visibility for {repo_id}")
    if card_only == "1":
        missing = []
        if "config.json" not in files:
            missing.append("config.json")
        if not any(f.endswith(".safetensors") and "/" not in f for f in files):
            missing.append("*.safetensors weights")
        if missing:
            fail(
                f"{repo_id} has no {' and no '.join(missing)} on the Hub: upload the "
                "int4-awq model there from the GPU instance first (this script "
                "uploads only its card)"
            )
    print("private" if private else "public")
    return 0


if command == "whoami":
    raise SystemExit(whoami(*args))
if command == "repo":
    raise SystemExit(repo(*args))

from secbench import hfcommon  # after whoami and repo, which need none of it


def human(size):
    for unit, scale in UNITS:
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{size} B"


def count(files):
    noun = "file" if len(files) == 1 else "files"
    return f"{len(files)} {noun}, {human(sum(f['size'] for f in files))}"


def load():
    path = dist / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        user, repos, files = data["hf_user"], data["repos"], data["files"]
        hfcommon.check_user(user)
        for f in files:
            name = PurePosixPath(f["path"])
            if name.is_absolute() or ".." in name.parts or not isinstance(f["size"], int):
                raise ValueError(f"file {f['path']!r}")
    except (OSError, ValueError, KeyError, TypeError, hfcommon.ExportError) as exc:
        fail(f"{path} isn't a manifest secbench.hfexport wrote ({exc})")
    for key, repo in repos.items():
        where = f"{path}: repo {key!r}"
        if key not in hfcommon.REPO_NAMES:
            fail(f"{where} isn't one of {', '.join(hfcommon.REPO_NAMES)}")
        expected = {"id": hfcommon.repo_id(user, key), "type": hfcommon.REPO_TYPES[key]}
        for field, value in expected.items():
            if repo.get(field) != value:
                fail(f"{where}: {field} {repo.get(field)!r}, expected {value!r}")
        folder = repo.get("folder")
        if not isinstance(folder, str) or folder in ("", ".", "..") or "/" in folder:
            fail(f"{where}: folder {folder!r} isn't a folder's name")
        if repo.get("private") is not True:
            fail(f"{where} isn't private: this script creates private repos only")
        if repo["type"] == "space" and not repo.get("sdk"):
            fail(f"{where}: a Space needs its sdk")
    return data, user, repos, files


def uploads(repo, files):
    """The manifest's files that go up to the repo: its folder's, or its card."""
    folder = repo["folder"]
    if repo.get("card_only"):
        card = [f for f in files if f["path"] == f"{folder}/README.md"]
        if not card:
            fail(f"repo {repo['id']}: {folder}/README.md isn't in the manifest")
        return card
    return [f for f in files if f["path"].split("/")[0] == folder]


def problems(repos, files):
    """Where the folders differ from the manifest (hf upload sends what is there)."""
    found = []
    for f in files:
        path = dist / f["path"]
        if not path.is_file():
            found.append(f"{f['path']}: in the manifest, not in {shown}")
        elif path.stat().st_size != f["size"]:
            size = path.stat().st_size
            found.append(f"{f['path']}: {size} bytes, the manifest says {f['size']}")
    listed = {f["path"] for f in files}
    for repo in repos.values():
        if repo.get("card_only"):
            continue
        for path in sorted((dist / repo["folder"]).rglob("*")):
            name = path.relative_to(dist).as_posix()
            if not path.is_dir() and name not in listed:
                found.append(f"{name}: not in the manifest, and hf upload would send it")
    return found


def commit_message(data):
    build = data.get("build") or {}
    commit = str(build.get("commit") or "")[:12]
    message = f"Upload from {hfcommon.REPO_URL}"
    if commit:
        message += f" at {commit}"
    if build.get("dirty"):
        message += ", with uncommitted changes"
    return message


def show(data, user, repos, files):
    print(f"Hugging Face export {shown} (dry run: nothing is sent)")
    if user == hfcommon.PLACEHOLDER_USER:
        print(
            f"user: {user}, the placeholder: a real run needs a rebuild with "
            "uv run python -m secbench.hfexport --hf-user NAME"
        )
    else:
        print(f"user: {user}")
    print(f"commit message: {commit_message(data)}")
    sent = []
    for key, repo in repos.items():
        extra = f", sdk {repo['sdk']}" if repo["type"] == "space" else ""
        print(f"\n{key}: {repo['id']} ({repo['type']}, private{extra})")
        print(f"  folder: {shown}/{repo['folder']}")
        if repo.get("card_only"):
            print(
                "  only README.md is uploaded: the model's files must already be in "
                "the repo (uploaded from the GPU instance)"
            )
        mine = uploads(repo, files)
        for f in mine:
            name = f["path"].split("/", 1)[1]
            print(f"  {human(f['size']):>10}  {name}")
        print(f"  {count(mine)}")
        sent += mine
    sent_paths = {f["path"] for f in sent}
    others = [f for f in files if f["path"] not in sent_paths]
    if others:
        print("\nNot uploaded (in no repo's upload):")
        for f in others:
            print(f"  {human(f['size']):>10}  {f['path']}")
    repo_count = f"{len(repos)} repo" + ("" if len(repos) == 1 else "s")
    print(f"\nTotal: {repo_count}, {count(sent)}")
    for item in data.get("skipped") or []:
        print(f"Skipped in the build: {item}")
    found = problems(repos, files)
    for problem in found:
        print(f"error: {problem}", file=sys.stderr)
    if found:
        print(
            f"error: {shown} doesn't match its manifest: rebuild it "
            "(uv run python -m secbench.hfexport --hf-user NAME)",
            file=sys.stderr,
        )
    print(
        "A real run creates a missing repo private and reads every repo's visibility "
        "from the Hub first: if one is public, it uploads nothing (unless "
        "--allow-public)."
    )
    print("Nothing was sent (dry run).")
    return 1 if found else 0


def plan(data, user, repos, files):
    if user == hfcommon.PLACEHOLDER_USER:
        fail(
            f"{shown} was built for the placeholder user {user}: rebuild it for "
            "your Hugging Face user or organization with "
            "uv run python -m secbench.hfexport --hf-user NAME"
        )
    print(f"user\t{user}")
    print(f"message\t{commit_message(data)}")
    for key, repo in repos.items():
        uploads(repo, files)  # a card-only repo's README.md is listed
        fields = (
            "repo",
            key,
            repo["id"],
            repo["type"],
            repo["folder"],
            "1" if repo.get("card_only") else "0",
            repo.get("sdk") or "-",
            hfcommon.hub_url(user, key),
        )
        print("\t".join(fields))
    return 0


commands = {"show": show, "plan": plan}
raise SystemExit(commands[command](*load()))
EOF
}

if [ "$DRY_RUN" = 1 ]; then
    manifest show
    exit 0
fi

case "${HF_TOKEN:-}" in
    *[![:space:]]*) ;;
    *) fail "HF_TOKEN isn't set, or is blank: set it to a Hugging Face access token with write access (the only place this script reads a token from), or run with --dry-run" ;;
esac
# The token is HF_TOKEN's: no OIDC token exchange, no older variable, no staging Hub
# (whose token file is fixed), and no stored login: huggingface_hub falls back to its
# token file (HF_TOKEN_PATH) when HF_TOKEN strips to nothing, and here that file is in
# a new, empty folder, so it doesn't exist.
unset HF_OIDC_RESOURCE HUGGING_FACE_HUB_TOKEN HUGGINGFACE_CO_STAGING
NO_LOGIN=$(mktemp -d)
trap 'rm -rf "$NO_LOGIN"' EXIT
export HF_TOKEN_PATH="$NO_LOGIN/token"
export HF_HUB_DISABLE_TELEMETRY=1
plan=$(manifest plan) || exit 1
TAB=$(printf '\t')
NL='
'
user=$(printf '%s\n' "$plan" | sed -n "s/^user$TAB//p")
message=$(printf '%s\n' "$plan" | sed -n "s/^message$TAB//p")
NOTHING_UPLOADED="no file was uploaded (a repo this run created is private and empty)"

step "Checking $DIST_ARG: its files against the manifest, and the privacy scan"
uv run python -m secbench.hfexport --verify --out "$DIST" </dev/null ||
    fail "secbench.hfexport --verify failed: nothing was sent"

step "Hugging Face account"
account=$(uv run hf auth whoami --format json </dev/null) ||
    fail "hf auth whoami failed: is HF_TOKEN a valid token? Nothing was sent"
manifest whoami "$user" "$account"

# Before any upload: each repo created private if it doesn't exist (not the int4-awq
# one, whose model must be there already), then its visibility read from the Hub.
step "Repositories (a missing one is created private; nothing is uploaded yet)"
states=""
public=""
private=0
while IFS="$TAB" read -r kind key id type folder card_only sdk url; do
    if [ "$kind" != repo ]; then
        continue
    fi
    expand=private
    if [ "$card_only" = 1 ]; then
        expand=private,siblings
    elif [ "$type" = space ]; then
        uv run hf repos create "$id" --type "$type" --private --exist-ok \
            --space-sdk "$sdk" </dev/null >/dev/null
    else
        uv run hf repos create "$id" --type "$type" --private --exist-ok \
            </dev/null >/dev/null
    fi
    if ! info=$(uv run hf "${type}s" info "$id" --expand "$expand" --format json </dev/null); then
        if [ "$card_only" = 1 ]; then
            echo "error: $id isn't on the Hub, or the token can't see it: upload the int4-awq model there from the GPU instance first (this script uploads only its card)" >&2
        else
            echo "error: hf couldn't read $id on the Hub" >&2
        fi
        fail "$NOTHING_UPLOADED"
    fi
    state=$(manifest repo "$id" "$card_only" "$info") || fail "$NOTHING_UPLOADED"
    echo "$key: $id ($type): $state"
    states="$states$id: $state$NL"
    if [ "$state" = public ]; then
        public="$public $id"
    else
        private=$((private + 1))
    fi
done <<EOF
$plan
EOF
if [ -n "$public" ] && [ "$ALLOW_PUBLIC" != 1 ]; then
    echo "error: public on the Hub:$public (an upload there is published at once): make each one private in its settings and run this again, or run with --allow-public" >&2
    fail "$NOTHING_UPLOADED"
fi

while IFS="$TAB" read -r kind key id type folder card_only sdk url; do
    if [ "$kind" != repo ]; then
        continue
    fi
    step "$key: uploading to $id"
    if [ "$card_only" = 1 ]; then
        uv run hf upload "$id" "$DIST/$folder/README.md" README.md --type "$type" \
            --private --commit-message "$message" </dev/null
    else
        uv run hf upload "$id" "$DIST/$folder" . --type "$type" --private \
            --commit-message "$message" </dev/null
    fi
    echo "uploaded: $url"
done <<EOF
$plan
EOF

step "Done: each repo's visibility, as the Hub reported it before the upload"
printf '%s' "$states"
if [ -n "$public" ]; then
    echo "Public (--allow-public):$public: their files are published."
fi
if [ "$private" -gt 0 ]; then
    echo "Look at each private repo on the Hub, then make it public in its settings."
fi
