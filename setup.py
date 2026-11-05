"""Manual (no-CMake) build for fastfields_bind.

Custom build_ext that:
  0. makes sure the fastfields-lib *source* tree is present at
     ./_fastfields_lib, fetching the pinned commit recorded in
     ``fastfields-lib.sha`` if it is not (see ``_ensure_source_tree``),
  1. builds the fastfields-lib shared libraries (via its Makefile) if missing
     -- or, if ``FASTFIELDS_PREBUILT_LIB`` is set, copies them in from there
     instead of invoking ``make`` at all (see ``PREBUILT_LIB_DIR`` below),
  2. copies libfastfields.so + libfastfields-cpu.so into fastfields_bind/lib/,
  3. compiles the nanobind extension (src/ext.cpp + nb_combined.cpp) against
     ./fastfields, linking -lfastfields with an $ORIGIN/lib rpath,
  4. ships the .so libraries as package data.

./_fastfields_lib is treated as the fastfields-lib source tree. It used to be a
git submodule; it is now a *recorded commit SHA* (``fastfields-lib.sha``) that
step 0 fetches. The reason is sdists: git does not put submodule contents in a
source distribution, so ``pip install`` from an sdist had no C++ tree and could
not build. A SHA in a tracked file travels in the sdist and can be fetched at
build time.

Where the source tree comes from, in order of precedence:

  1. ``$FASTFIELDS_DIR`` (and/or ``$FASTFIELDS_INCLUDE``) -- an explicit local
     checkout. Nothing is fetched, nothing is reset: a contributor iterating on
     the C++ side points the build at their own working tree and the build
     leaves it exactly as it is.
  2. an existing ./_fastfields_lib directory. Likewise used as-is. If its
     checked-out commit is not the pinned one, the build *warns loudly and
     carries on* -- resetting somebody's working tree would be worse than
     building something they can see is unpinned.
  3. otherwise, a shallow fetch of the pinned SHA from ``$FASTFIELDS_URL``
     (default: the public fastfields-lib on GitHub). This is the only step that
     needs the network; with no network and no local checkout the build fails
     with a message naming the SHA, the URL, and ``FASTFIELDS_DIR``.

Note that the extension itself (step 3) compiles against the C++ *headers*
under the source tree, so that tree must be present even when the compiled .so
files come from ``FASTFIELDS_PREBUILT_LIB`` -- only the expensive templated
``make`` build of the .so files is skipped.

The Python package built here is the PEP 420 namespace subpackage
``fastfields.dlpack`` (the compiled extension imports as
``fastfields.dlpack._core``); no ``fastfields/__init__.py`` is created, so
other distributions can merge into the same ``fastfields`` namespace.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.abspath(__file__))

# The pinned fastfields-lib commit, and where to fetch it from. The SHA lives
# in its own tracked file so that there is exactly one place to look and one
# place to edit; that file also carries the bump instructions.
PIN_FILE = os.path.join(HERE, "fastfields-lib.sha")
DEFAULT_URL = "https://github.com/fastfields/fastfields-lib.git"
FASTFIELDS_URL = os.environ.get("FASTFIELDS_URL") or DEFAULT_URL

# ``$FASTFIELDS_DIR`` / ``$FASTFIELDS_INCLUDE`` override where the C++ source
# tree and its public headers are found. Both are opt-in; unset, the build uses
# the in-tree ./_fastfields_lib that step 0 populates.
DEFAULT_FASTFIELDS_DIR = os.path.join(HERE, "_fastfields_lib")
FASTFIELDS_DIR_ENV = os.environ.get("FASTFIELDS_DIR")
FASTFIELDS_DIR = FASTFIELDS_DIR_ENV or DEFAULT_FASTFIELDS_DIR
FASTFIELDS_BUILD = os.path.join(FASTFIELDS_DIR, "build")
# Public headers moved under <tree>/include/ when the six fastfields repos were
# consolidated into fastfields-lib; the hub's own headers are now reached as
# "fastfields/api/<name>.h" (see src/ext.cpp).
FASTFIELDS_INCLUDE_ENV = os.environ.get("FASTFIELDS_INCLUDE")
FASTFIELDS_INCLUDE = FASTFIELDS_INCLUDE_ENV or os.path.join(
    FASTFIELDS_DIR, "include"
)
PKG_DIR = os.path.join(HERE, "fastfields", "dlpack")
PKG_LIB_DIR = os.path.join(PKG_DIR, "lib")

# The fastfields-lib Makefile names its shared libraries per platform, exactly
# as the loader expects: .so on Linux, .dylib on macOS, .dll on Windows. Match
# that here (a hard-coded ".so" silently broke the macOS/Windows CI legs). The
# extension is found next to its shipped libs via an rpath that also differs by
# platform ($ORIGIN on ELF, @loader_path on Mach-O); Windows has no rpath and
# instead relies on os.add_dll_directory at import time (see __init__.py).
if sys.platform == "darwin":
    LIBEXT = "dylib"
    RPATH_FLAG = "-Wl,-rpath,@loader_path/lib"
elif sys.platform == "win32":
    LIBEXT = "dll"
    RPATH_FLAG = None
else:
    LIBEXT = "so"
    RPATH_FLAG = "-Wl,-rpath,$ORIGIN/lib"

# Shared libraries produced by the fastfields-lib Makefile.
MAIN_LIB = "libfastfields." + LIBEXT
CPU_LIB = "libfastfields-cpu." + LIBEXT
MAIN_LIB_PATH = os.path.join(FASTFIELDS_BUILD, MAIN_LIB)
CPU_LIB_PATH = os.path.join(FASTFIELDS_BUILD, "lib", CPU_LIB)

CXX = os.environ.get("CXX", "clang++")

# Opt-in fast path for CI: if set, points at a directory already containing
# MAIN_LIB/CPU_LIB (built once, out of band -- e.g. the wheel-build workflow's
# `build-native` job, shared across every Python-version job in its matrix
# instead of each of them re-running `make`). When unset, behaviour is exactly
# the from-source build below, so local/dev installs and this repo's own
# non-release test CI are unaffected.
PREBUILT_LIB_DIR = os.environ.get("FASTFIELDS_PREBUILT_LIB")

_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z")


def _read_pin() -> str:
    """Return the pinned fastfields-lib commit SHA from ``PIN_FILE``.

    The file is comments (``#``) plus one full 40-character lower-case SHA.
    Anything else is an error: a half-written pin must not be guessed at.
    """
    try:
        with open(PIN_FILE, encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as exc:
        raise RuntimeError(
            f"cannot read the fastfields-lib pin at {PIN_FILE!r}: {exc}. "
            "It records the C++ commit this extension builds against and "
            "must ship in the sdist (see MANIFEST.in)."
        ) from None
    tokens = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(tokens) != 1 or not _SHA_RE.match(tokens[0]):
        raise RuntimeError(
            f"{PIN_FILE!r} must contain exactly one full 40-character "
            f"lower-case commit SHA (plus '#' comments); found {tokens!r}. "
            "A tag or a branch name is deliberately not accepted: it can be "
            "moved, and a moved pin changes what a pinned build compiles."
        )
    return tokens[0]


def _git_head(directory: str) -> str | None:
    """Return the HEAD commit of ``directory``, or None if it is not a repo."""
    try:
        out = subprocess.run(
            ["git", "-C", directory, "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.decode().strip()


def _warn_existing_tree(directory: str, pinned: str, named: bool) -> None:
    """Report which tree is being used, and whether it matches the pin.

    A mismatch is a warning, never an error and never a reset: this is how a
    contributor builds against their own work-in-progress C++ tree.
    """
    why = (
        "the tree named by $FASTFIELDS_DIR/$FASTFIELDS_INCLUDE,"
        if named
        else "the existing checkout at"
    )
    head = _git_head(directory)
    if head == pinned:
        print(
            f"[fastfields_bind] using {why} {directory} "
            f"(at the pinned commit {pinned[:12]})"
        )
        return
    remedy = (
        "unset it to build against the pin."
        if named
        else "remove it to build against the pin."
    )
    found = head[:12] if head else "not a git checkout"
    print(
        f"\n[fastfields_bind] WARNING: building against {why} "
        f"{directory},\n"
        f"[fastfields_bind]          which is at {found}, not the pinned "
        f"commit {pinned[:12]}\n"
        f"[fastfields_bind]          recorded in {os.path.basename(PIN_FILE)}."
        f" Using it anyway, untouched:\n"
        f"[fastfields_bind]          {remedy}\n",
        file=sys.stderr,
    )


def _fetch_pinned_tree(destination: str, pinned: str) -> None:
    """Shallow-fetch exactly ``pinned`` from ``FASTFIELDS_URL``.

    ``git clone --depth 1 --branch <sha>`` does not work -- ``--branch`` takes
    refs, not commit ids -- so this is the init/remote/fetch/checkout form.
    GitHub serves an arbitrary reachable commit this way
    (``uploadpack.allowReachableSHA1InWant``).

    The fetch lands in a sibling temporary directory and is moved into place
    only once it has succeeded, so an interrupted build never leaves a
    half-populated tree that the next build would mistake for a checkout. If
    ``destination`` already exists without being a checkout -- a restored CI
    cache of ``_fastfields_lib/build/`` is the case that matters -- the tree is
    merged around it rather than replacing it.
    """
    staging = destination + ".fetching"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)
    steps = [
        ["git", "init", "-q", staging],
        ["git", "-C", staging, "remote", "add", "origin", FASTFIELDS_URL],
        ["git", "-C", staging, "fetch", "--depth", "1", "origin", pinned],
        ["git", "-C", staging, "checkout", "-q", "--detach", "FETCH_HEAD"],
    ]
    print(
        f"[fastfields_bind] fetching pinned fastfields-lib {pinned[:12]} "
        f"from {FASTFIELDS_URL}"
    )
    try:
        for step in steps:
            subprocess.run(
                step,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
    except (OSError, subprocess.CalledProcessError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(_fetch_failure_message(pinned, exc)) from None
    head = _git_head(staging)
    if head != pinned:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f"fetched fastfields-lib is at {head!r}, not the pinned "
            f"{pinned!r} -- refusing to build against it"
        )
    _move_into_place(staging, destination)
    print(f"[fastfields_bind] fastfields-lib {pinned[:12]} -> {destination}")


def _move_into_place(staging: str, destination: str) -> None:
    """Move the fetched tree to ``destination``, merging around what is there.

    ``destination`` can already exist without being a checkout: the obvious
    case is a CI cache of the expensive ``make`` output, restored to
    ``_fastfields_lib/build/`` before anything has fetched the sources it was
    built from. Merging keeps that usable. Nothing is ever overwritten -- a
    name that already exists is an error, not a silent replacement.
    """
    if not os.path.exists(destination):
        os.replace(staging, destination)
        return
    entries = sorted(os.listdir(staging))
    clashes = [
        e for e in entries if os.path.exists(os.path.join(destination, e))
    ]
    if clashes:
        # Check every name before moving any, so a refusal leaves both the
        # destination and the staging area exactly as they were.
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f"refusing to overwrite {clashes} in the existing "
            f"{destination!r} while unpacking the pinned fastfields-lib. "
            "Remove that directory (or point $FASTFIELDS_DIR at a real "
            "checkout) and build again."
        )
    for entry in entries:
        os.replace(
            os.path.join(staging, entry), os.path.join(destination, entry)
        )
    os.rmdir(staging)


def _fetch_failure_message(pinned: str, exc: BaseException) -> str:
    """Explain an unreachable pin: what, from where, and how to supply it."""
    if isinstance(exc, subprocess.CalledProcessError):
        output = (exc.output or b"").decode(errors="replace").strip()
        detail = "git said:\n    " + "\n    ".join(
            output.splitlines()[-6:] or ["(no output)"]
        )
    else:
        detail = f"could not run git: {exc}"
    return (
        "\n"
        "could not obtain the fastfields-lib C++ source tree.\n"
        "\n"
        f"  wanted commit : {pinned}\n"
        f"  from          : {FASTFIELDS_URL}\n"
        f"  into          : {DEFAULT_FASTFIELDS_DIR}\n"
        f"  pinned by     : {PIN_FILE}\n"
        "\n"
        "Building this package needs that tree: src/ext.cpp compiles against\n"
        "its headers. It is fetched at build time (it is not a submodule, so\n"
        "that an sdist can carry the pin), which means this step needs\n"
        "network access to the URL above.\n"
        "\n"
        "To build without the network, supply the tree locally and point the\n"
        "build at it -- it is then used as-is and never modified:\n"
        "\n"
        "    git clone https://github.com/fastfields/fastfields-lib.git\n"
        f"    git -C fastfields-lib checkout {pinned}\n"
        "    FASTFIELDS_DIR=$PWD/fastfields-lib pip install .\n"
        "\n"
        "(Or unpack it to ./_fastfields_lib, which is used without\n"
        "FASTFIELDS_DIR being set. A tree at a different commit is accepted\n"
        "with a warning.)\n"
        "\n" + detail
    )


def _nanobind_paths():
    import nanobind

    nb_root = os.path.dirname(nanobind.__file__)
    return {
        "include": nanobind.include_dir(),
        "robin_map": os.path.join(nb_root, "ext", "robin_map", "include"),
        "combined_src": os.path.join(nb_root, "src", "nb_combined.cpp"),
    }


class BuildExt(build_ext):
    def run(self):
        self._ensure_source_tree()
        self._ensure_fastfields_libs()
        self._copy_libs_into_package()
        super().run()

    # -- step 0: make sure the C++ source tree is there ----------------------
    def _ensure_source_tree(self):
        """Resolve the C++ source tree: env var, then existing dir, then fetch.

        Only the last of the three touches the network or the filesystem; the
        first two use what is already there, unchanged.
        """
        pinned = _read_pin()
        named = bool(FASTFIELDS_DIR_ENV or FASTFIELDS_INCLUDE_ENV)
        if os.path.isdir(FASTFIELDS_INCLUDE):
            # Headers already available: case 1 or case 2. Which of the two
            # only changes the wording of the message.
            _warn_existing_tree(FASTFIELDS_DIR, pinned, named)
            return
        if named:
            # An explicit instruction that does not resolve. Fetching
            # somewhere else instead would silently ignore it.
            raise RuntimeError(
                f"$FASTFIELDS_DIR / $FASTFIELDS_INCLUDE point at "
                f"{FASTFIELDS_INCLUDE!r}, which is not a directory. Set them "
                "to a fastfields-lib checkout (whose public headers are in "
                "<tree>/include/), or unset both to build against the commit "
                f"pinned in {os.path.basename(PIN_FILE)}."
            )
        _fetch_pinned_tree(FASTFIELDS_DIR, pinned)

    # -- step 1: build the C++ libraries if they are not there yet -----------
    def _ensure_fastfields_libs(self):
        if os.path.exists(MAIN_LIB_PATH) and os.path.exists(CPU_LIB_PATH):
            return
        if PREBUILT_LIB_DIR:
            self._use_prebuilt_libs()
            return
        print(f"[fastfields_bind] building fastfields-lib in {FASTFIELDS_DIR}")
        # No explicit target: fastfields-lib's make/common.mk sets
        # `.DEFAULT_GOAL := all`, and a bare `make` is what builds both the
        # hub and the CPU backend.
        subprocess.check_call(["make", "-C", FASTFIELDS_DIR, f"CXX={CXX}"])
        if not (
            os.path.exists(MAIN_LIB_PATH) and os.path.exists(CPU_LIB_PATH)
        ):
            raise RuntimeError(
                "fastfields-lib build did not produce the expected .so files"
            )

    # -- step 1b: or copy them in from a pre-built directory (CI fast path) --
    def _use_prebuilt_libs(self):
        print(
            f"[fastfields_bind] using prebuilt libs from "
            f"{PREBUILT_LIB_DIR!r} (FASTFIELDS_PREBUILT_LIB); skipping make"
        )
        os.makedirs(os.path.dirname(MAIN_LIB_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(CPU_LIB_PATH), exist_ok=True)
        for name, dst in ((MAIN_LIB, MAIN_LIB_PATH), (CPU_LIB, CPU_LIB_PATH)):
            src = os.path.join(PREBUILT_LIB_DIR, name)
            if not os.path.exists(src):
                raise RuntimeError(
                    f"FASTFIELDS_PREBUILT_LIB={PREBUILT_LIB_DIR!r} is set but "
                    f"{src!r} does not exist"
                )
            shutil.copyfile(src, dst)
            shutil.copymode(src, dst)

    # -- step 2: ship the libraries inside the package -----------------------
    def _copy_libs_into_package(self):
        os.makedirs(PKG_LIB_DIR, exist_ok=True)
        for src in (MAIN_LIB_PATH, CPU_LIB_PATH):
            dst = os.path.join(PKG_LIB_DIR, os.path.basename(src))
            shutil.copyfile(src, dst)
            shutil.copymode(src, dst)
        # Mirror into the build tree so a wheel build also picks them up.
        if getattr(self, "build_lib", None):
            build_pkg_lib = os.path.join(
                self.build_lib, "fastfields", "dlpack", "lib"
            )
            os.makedirs(build_pkg_lib, exist_ok=True)
            for name in (MAIN_LIB, CPU_LIB):
                shutil.copyfile(
                    os.path.join(PKG_LIB_DIR, name),
                    os.path.join(build_pkg_lib, name),
                )

    # -- step 3: compile the nanobind extension ------------------------------
    def build_extension(self, ext):
        nb = _nanobind_paths()

        include_dirs = [
            nb["include"],
            nb["robin_map"],
            sysconfig.get_path("include"),
            FASTFIELDS_INCLUDE,
        ]
        sources = list(ext.sources) + [nb["combined_src"]]

        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)

        cmd = [CXX, "-std=c++17", "-O2", "-shared", "-fvisibility=hidden"]
        # Position-independent code is POSIX-only (default on Windows PE/COFF).
        if sys.platform != "win32":
            cmd.append("-fPIC")
        # A CPython extension resolves the interpreter's C-API symbols (_Py*)
        # at load time, so they must be left undefined in the module. ELF
        # permits undefined symbols in a shared object by default; Mach-O does
        # not, so ld errors ("Undefined symbols ... _PyWeakref_NewRef") unless
        # we defer them explicitly.
        if sys.platform == "darwin":
            cmd += ["-undefined", "dynamic_lookup"]
        for inc in include_dirs:
            cmd += ["-I", inc]
        cmd += sources
        if sys.platform == "win32":
            # No import library is produced for libfastfields.dll, so link the
            # extension directly against the DLL by path; the DLL is then
            # located at import time via os.add_dll_directory.
            cmd += [MAIN_LIB_PATH]
        else:
            cmd += ["-L", FASTFIELDS_BUILD, "-lfastfields"]
        if RPATH_FLAG:
            cmd += [RPATH_FLAG]
        cmd += ["-o", ext_path]
        print("[fastfields_bind] " + " ".join(cmd))
        subprocess.check_call(cmd)

        # Ensure the libs sit next to the freshly-built extension too (covers
        # both in-place/editable builds and staged wheel builds).
        ext_lib_dir = os.path.join(os.path.dirname(ext_path), "lib")
        os.makedirs(ext_lib_dir, exist_ok=True)
        for name in (MAIN_LIB, CPU_LIB):
            shutil.copyfile(
                os.path.join(PKG_LIB_DIR, name),
                os.path.join(ext_lib_dir, name),
            )


ext_modules = [
    Extension(
        "fastfields.dlpack._core",
        sources=[os.path.join("src", "ext.cpp")],
    )
]

setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=ext_modules,
)
