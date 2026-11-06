#!/usr/bin/env python3
"""Fail the docs build when the rendered API reference has gone empty.

`docs/api/index.md` is a heading and a single `::: fastfields.dlpack`, so
every function on the published API page is produced by mkdocstrings out
of the *built* extension: `fastfields/dlpack/__init__.py` is 58
`name = _core.name` aliases and the docstrings live in `src/ext.cpp`.
That makes the page's content entirely dependent on the handler being
able to resolve `_core` -- and when it cannot, mkdocstrings does not
fail, or warn, or render a placeholder. It renders the module heading
and its module docstring, drops all 58 functions, and the build reports
`No issues found`.

That is not hypothetical. It is what shipped: `zensical.toml` set
`paths = ["."]` without `force_inspection`, griffe resolved
`fastfields.dlpack` by static analysis of the source tree where
`_core`'s attributes are unresolvable, and
https://fastfields.github.io/dlpack/api/ went live with one documented
object -- the module -- and a 170-byte (i.e. empty) `objects.inv`. The
equivalent page in the `fastfields` repo carried 74. Nothing in the
build, the nav check or the test suite noticed, because nothing looked
at the rendered HTML.

This script looks at the rendered HTML. It is the only check in this
repo downstream of the renderer, so it is deliberately blunt: it counts
the documented objects mkdocstrings emitted and fails if there are
fewer than `FLOOR`. It is not a style or completeness check -- an op
missing from the *binding* is caught by `tests/test_api_coverage.py`
against the hub headers, which is a sharper check and a different
question. This one answers only "did the renderer produce an API
reference at all", which is the question that was going unasked.

Marker
------
mkdocstrings-python wraps each rendered object in

    <div class="doc doc-object doc-function">     (also doc-class, doc-module)

and gives each one a sibling `<div class="doc doc-contents ...">`. The
two counts agree, and `doc-object` is the more specific of the pair --
it also says what *kind* of object it was, which is what makes the
failure legible: the broken page has one `doc-module` and zero
`doc-function`, so the report can name the thing that went missing
rather than quoting a number. Note that `doc-object-name` (the
`<code>` holding the signature) is a *different* class and must not be
counted; the pattern below requires the exact `doc doc-object doc-KIND`
triple.

Usage:  check-api-page.py --check [site/api/index.html ...]
        check-api-page.py --selftest
"""

from __future__ import annotations

import os
import re
import sys

#: The rendered page this guard is about, relative to the site root.
DEFAULT_PAGE = os.path.join("site", "api", "index.html")

#: Minimum number of documented objects on the API page.
#:
#: `src/ext.cpp` binds 58 functions and
#: `fastfields/dlpack/__init__.py` re-exports all 58, so a healthy page
#: renders the module, one entry per function, and `__version__`: 60
#: objects, as measured. The floor is set at 50 -- comfortably under
#: that, so deliberately privatising or dropping a handful of ops is a
#: code review rather than an edit to this script, and far above the
#: single-digit range that means the handler stopped resolving `_core`
#: at all. The failure this exists to catch took the page from 60 to 1;
#: it is not a near-miss, and the floor does not need to be tight to
#: catch it.
#:
#: Raise it if the binding grows a lot. Do NOT lower it to make a red
#: check go green -- a drop of that size is the renderer breaking, not
#: the API shrinking.
FLOOR = 50

#: One rendered object. `doc doc-object doc-object-name` does not match:
#: `object` must be followed by a space.
_OBJECT_RE = re.compile(r'class="doc doc-object doc-([a-z]+)"')


def count_objects(html: str) -> dict[str, int]:
    """Return ``{kind: n}`` for the documented objects in a rendered page."""
    counts: dict[str, int] = {}
    for kind in _OBJECT_RE.findall(html):
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _describe(counts: dict[str, int]) -> str:
    """Render ``{kind: n}`` as ``"58 function, 1 module"``."""
    if not counts:
        return "nothing"
    return ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))


def check_page(path: str) -> list[str]:
    """Problems with one rendered API page; empty means it is fine."""
    if not os.path.exists(path):
        return [
            f"{path}: not found. Build the site first "
            "(`zensical build --clean`), or pass the page's path."
        ]
    try:
        html = open(path, encoding="utf-8").read()
    except OSError as error:  # pragma: no cover - unreadable build output
        return [f"{path}: cannot be read: {error}"]

    counts = count_objects(html)
    total = sum(counts.values())
    if total >= FLOOR:
        return []

    problems = [
        f"{path}: {total} documented object(s) ({_describe(counts)}); "
        f"at least {FLOOR} expected."
    ]
    if not counts.get("function"):
        problems.append(
            f"{path}: not one function was rendered. The handler did not "
            "resolve `fastfields.dlpack._core` -- see HELP below."
        )
    return problems


# (name, html, expected-to-pass)
#
# The point of these cases is that the guard must reject the *exact*
# shape of the page that shipped: a module heading, its docstring, and
# no functions. "Fails on an empty page" is the whole contract, so it is
# the first case, built from the markup mkdocstrings really emits.
_MODULE = '<div class="doc doc-object doc-module">'
_NAME = '<code class="doc doc-object-name doc-module-name">x</code>'
_CONTENTS = '<div class="doc doc-contents first">'
_FUNC = '<div class="doc doc-object doc-function">'

SELFTEST_CASES = [
    (
        "the page that shipped: module only, no functions",
        f"<html>{_MODULE}{_NAME}{_CONTENTS}<p>docstring</p></html>",
        False,
    ),
    ("no mkdocstrings output at all", "<html><h1>API</h1></html>", False),
    (
        "a healthy page: module + FLOOR functions",
        "<html>" + _MODULE + _FUNC * FLOOR + "</html>",
        True,
    ),
    (
        "exactly at the floor",
        "<html>" + _FUNC * FLOOR + "</html>",
        True,
    ),
    (
        "one object short of the floor",
        "<html>" + _FUNC * (FLOOR - 1) + "</html>",
        False,
    ),
    (
        "the real page: module + 58 functions + __version__",
        "<html>"
        + _MODULE
        + _FUNC * 58
        + '<div class="doc doc-object doc-attribute">'
        + "</html>",
        True,
    ),
    # `doc-object-name` is the signature's <code>, not an object. A guard
    # that counted it would see the broken page as having two objects --
    # and, worse, would scale with the page rather than with the number
    # of documented things.
    (
        "doc-object-name is not counted as an object",
        "<html>" + _MODULE + _NAME * 200 + "</html>",
        False,
    ),
    (
        "classes and modules count towards the total",
        "<html>"
        + _MODULE
        + '<div class="doc doc-object doc-class">' * (FLOOR - 1)
        + "</html>",
        True,
    ),
]


def selftest() -> int:
    """Check the checker: every case must land on its expected verdict."""
    import tempfile

    failures = 0
    for name, html, should_pass in SELFTEST_CASES:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.html")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(html)
            problems = check_page(path)
        passed = not problems
        if passed != should_pass:
            failures += 1
            verdict = "passed" if passed else "failed"
            want = "pass" if should_pass else "fail"
            print(
                f"  selftest {name!r}: {verdict}, expected it to {want}",
                file=sys.stderr,
            )
        else:
            got = "passes" if passed else "rejected"
            print(f"  selftest {name!r}: ok ({got})")

    # A missing page must be an error, not a pass: a guard that is happy
    # when it cannot find the page would go green on a build that
    # produced nothing at all.
    if not check_page(os.path.join("does", "not", "exist.html")):
        failures += 1
        print(
            "  selftest 'missing page': passed, expected it to fail",
            file=sys.stderr,
        )
    else:
        print("  selftest 'missing page': ok (rejected)")

    if failures:
        print(
            f"check-api-page.py: {failures} selftest case(s) wrong.",
            file=sys.stderr,
        )
        return 1
    n = len(SELFTEST_CASES) + 1
    print(f"check-api-page.py: all {n} selftest cases behave.")
    return 0


HELP = (
    "\n  The API page is rendered from the BUILT extension: the\n"
    "  docstrings live in src/ext.cpp, and\n"
    "  fastfields/dlpack/__init__.py only aliases _core's attributes.\n"
    "  For mkdocstrings to see them, the handler must INSPECT the\n"
    "  module rather than parse its source -- `force_inspection` in\n"
    "  zensical.toml -- and the package must be importable, which is\n"
    "  what `pip install .` is for in .github/workflows/docs.yaml.\n"
    "  Check, in this order:\n"
    "    1. `pip install .` ran and `python -c \"import\n"
    "       fastfields.dlpack._core\"` works;\n"
    "    2. zensical.toml still sets force_inspection = true under\n"
    "       [project.plugins.mkdocstrings.handlers.python.options];\n"
    "    3. the handler's `paths` does not shadow the installed\n"
    "       package with the source tree (which has no built _core).\n"
    "  A silently empty page is the failure mode this check exists\n"
    "  for; do not lower FLOOR to clear it."
)


def main(argv: list[str]) -> int:
    args = [a for a in argv if a != "--check"]
    if "--selftest" in args:
        if len(args) != 1:
            print(
                "check-api-page.py: --selftest takes no other arguments.",
                file=sys.stderr,
            )
            return 2
        return selftest()

    paths = args or [DEFAULT_PAGE]
    problems: list[str] = []
    for path in paths:
        problems.extend(check_page(path))

    if problems:
        print(
            "check-api-page.py: the rendered API reference is empty or "
            "nearly so.\n",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print(HELP, file=sys.stderr)
        return 1

    for path in paths:
        counts = count_objects(open(path, encoding="utf-8").read())
        print(
            f"check-api-page.py: {path} documents "
            f"{sum(counts.values())} objects ({_describe(counts)})."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
