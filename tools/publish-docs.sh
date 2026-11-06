#!/usr/bin/env bash
#
# Publish one built docs site into a subtree of the shared GitHub Pages branch.
#
# The fastfields documentation is one site assembled from four repositories:
#
#     /          fastfields.github.io   (the landing page + the shared theme)
#     /python/   fastfields
#     /dlpack/   fastfields-dlpack
#     /lib/      fastfields-lib
#     /whl/      whl -- NOT here; its own project Pages site, independent
#
# All but `whl` publish into the `gh-pages` branch of fastfields.github.io,
# which is what Pages serves. The one invariant that makes that safe:
#
#     *** A publisher writes ONLY its own subtree. ***
#
# Get that wrong with a full checkout-and-replace and each deploy silently
# deletes the other three sections. So the sync below is scoped: a subdirectory
# publisher rsyncs inside its own directory and can reach nothing else, and the
# root publisher excludes every subtree owned by someone else. An excluded path
# is also protected from --delete by rsync, which is the property being relied
# on here.
#
# Concurrency: `concurrency:` groups in GitHub Actions are per-repository, so
# they cannot serialize publishers that live in different repos. Two repos can
# and will deploy at the same time. That is handled the only way it can be --
# optimistically: fetch, re-apply onto the current tip, push, and on a
# non-fast-forward rejection start over against the new tip. The sync is
# idempotent, so a retry is always safe.
#
# Usage:  publish-docs.sh --subdir <name|""> [--site site]
# Needs:  GH_TOKEN with `contents: write` on the target repo.

set -euo pipefail

SUBDIR=""
SITE="site"
AUTH=""
TARGET_REPO="${TARGET_REPO:-fastfields/fastfields.github.io}"
BRANCH="${BRANCH:-gh-pages}"
ATTEMPTS="${ATTEMPTS:-5}"

# Subtrees owned by other publishers. Only consulted for the root publisher.
OTHERS=(python dlpack lib)

while [ $# -gt 0 ]; do
  case "$1" in
    --subdir) SUBDIR="${2:-}"; shift 2 ;;
    --site)   SITE="${2:-}";   shift 2 ;;
    --auth)   AUTH="${2:-}";   shift 2 ;;
    *) echo "publish-docs.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

[ -d "$SITE" ] || { echo "publish-docs.sh: '$SITE' is not a directory -- did the build run?" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Credentials.
#
# Checked BEFORE any work, and the failure is one loud line that names the
# secret and where it has to be visible. The failure mode this project has
# already lived through is a credential that quietly stopped working and took
# CI down at checkout for weeks because nobody read the error. A deploy key
# cannot expire, but it can be revoked or scoped to the wrong repositories.
#
#   --auth ssh    the three publisher repos. DEPLOY_KEY is the private half of
#                 the SSH deploy key registered on the target repo with write
#                 access, supplied from the DOCS_DEPLOY_KEY org secret.
#   --auth token  the target repo publishing its own site root, where the
#                 default GITHUB_TOKEN already has contents: write.
#
# PUBLISH_REMOTE overrides the remote entirely, so the scoping and the retry
# can be exercised against a scratch repo or a fork.
# ---------------------------------------------------------------------------
SSH_KEY_FILE=""

case "$AUTH" in
  ssh)
    if [ -z "${DEPLOY_KEY:-}" ]; then
      echo "publish-docs.sh: DEPLOY_KEY is empty -- the DOCS_DEPLOY_KEY organization secret is missing, revoked, or not scoped to this repository." >&2
      echo "publish-docs.sh: it must be visible to fastfields, fastfields-lib and fastfields-dlpack, and be the private half of a deploy key with write access on ${TARGET_REPO}." >&2
      exit 1
    fi
    SSH_KEY_FILE="$(mktemp)"
    chmod 600 "$SSH_KEY_FILE"
    printf '%s\n' "$DEPLOY_KEY" > "$SSH_KEY_FILE"
    export GIT_SSH_COMMAND="ssh -i $SSH_KEY_FILE -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
    DEFAULT_REMOTE="git@github.com:${TARGET_REPO}.git"
    ;;
  token)
    if [ -z "${GH_TOKEN:-}" ]; then
      echo "publish-docs.sh: GH_TOKEN is empty -- expected the workflow's GITHUB_TOKEN with contents: write on ${TARGET_REPO}." >&2
      exit 1
    fi
    DEFAULT_REMOTE="https://x-access-token:${GH_TOKEN}@github.com/${TARGET_REPO}.git"
    ;;
  *)
    if [ -n "${PUBLISH_REMOTE:-}" ]; then
      DEFAULT_REMOTE=""
    else
      echo "publish-docs.sh: --auth must be 'ssh' or 'token' (or set PUBLISH_REMOTE)." >&2
      exit 2
    fi
    ;;
esac

# Reject anything that could escape the intended subtree.
case "$SUBDIR" in
  */*|.|..|/*|*..*) echo "publish-docs.sh: refusing unsafe --subdir '$SUBDIR'" >&2; exit 2 ;;
esac

if [ -z "$SUBDIR" ]; then
  echo "publish-docs.sh: publishing the SITE ROOT of $TARGET_REPO@$BRANCH"
else
  echo "publish-docs.sh: publishing /$SUBDIR/ of $TARGET_REPO@$BRANCH"
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK" "${SSH_KEY_FILE:-}"' EXIT
REMOTE="${PUBLISH_REMOTE:-$DEFAULT_REMOTE}"

for attempt in $(seq 1 "$ATTEMPTS"); do
  CLONE="$WORK/try$attempt"

  # A fresh clone each attempt is what makes the retry correct: the re-apply
  # happens onto whatever tip won the race, not onto a stale index.
  if git clone --quiet --depth 1 --branch "$BRANCH" "$REMOTE" "$CLONE" 2>/dev/null; then
    :
  else
    echo "  branch '$BRANCH' does not exist yet; creating it"
    git clone --quiet --depth 1 "$REMOTE" "$CLONE"
    git -C "$CLONE" checkout --quiet --orphan "$BRANCH"
    git -C "$CLONE" rm -rq --cached . 2>/dev/null || true
    find "$CLONE" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
  fi

  if [ -z "$SUBDIR" ]; then
    # --delete at the clone root would otherwise delete .git itself, since it
    # is not part of the built site. A subdirectory publisher never hits this
    # because it syncs into $CLONE/$SUBDIR/.
    EXCLUDES=( "--exclude=/.git" )
    for d in "${OTHERS[@]}"; do EXCLUDES+=( "--exclude=/$d" ); done
    rsync -a --delete "${EXCLUDES[@]}" "$SITE/" "$CLONE/"
  else
    mkdir -p "$CLONE/$SUBDIR"
    rsync -a --delete "$SITE/" "$CLONE/$SUBDIR/"
  fi

  git -C "$CLONE" add -A
  if git -C "$CLONE" diff --cached --quiet; then
    echo "publish-docs.sh: no change to publish."
    exit 0
  fi

  git \
    -C "$CLONE" \
    -c user.name="fastfields-docs-bot" \
    -c user.email="noreply@github.com" \
    commit --quiet \
    -m "docs: publish ${SUBDIR:-/} from ${GITHUB_REPOSITORY:-local}@${GITHUB_SHA:-manual}"

  if git -C "$CLONE" push --quiet origin "$BRANCH"; then
    echo "publish-docs.sh: published ${SUBDIR:-/} (attempt $attempt)."
    exit 0
  fi

  echo "  push rejected (another publisher won the race); retrying against the new tip"
  sleep $(( attempt * 5 ))
done

echo "publish-docs.sh: giving up after $ATTEMPTS attempts." >&2
exit 1
