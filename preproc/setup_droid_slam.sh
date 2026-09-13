#!/usr/bin/env bash
# Build DROID-SLAM (camera estimation for the preprocessing chain) on a modern
# toolchain. `setup_dependencies.sh` cannot do this on PyTorch >= 2: four separate
# things break, and each one only surfaces after the previous is fixed.
#
#   1. Nested submodules are never initialised. DROID-SLAM needs thirdparty/lietorch
#      and thirdparty/eigen, and lietorch needs its OWN nested eigen -- three levels.
#      A plain `git submodule update --init` stops at the first level and the build
#      fails with "package directory 'thirdparty/lietorch/lietorch' does not exist".
#
#   2. DROID-SLAM's setup.py calls setup() TWICE in one file (droid_backends and
#      lietorch). PEP-517 frontends reject that: "Exactly one .egg-info should have
#      been produced, but found 2". So we build the extensions in place instead of
#      `pip install .`, which is also how the repo's own demo.py expects to run.
#
#   3. tensor.type() no longer converts implicitly to c10::ScalarType in PyTorch 2.x:
#      "no suitable conversion function from const at::DeprecatedTypeProperties".
#      44 call sites across 6 files need .scalar_type(). Applied from patches/, NOT
#      by sed, so the change is reviewable and survives a fresh clone. Note
#      `.device().type()` is a different, still-valid API and must be left alone.
#
#   4. ninja invokes plain `c++`, which does not exist on distros that ship only
#      versioned compilers (Ubuntu 24.04 has g++-13 but no g++). The failure is
#      "FAILED: [code=127]" -- 127 is command-not-found -- but the Python traceback
#      only says "Error compiling objects for extension", so the real cause is
#      invisible unless you read ninja's own output.
#
# Also patches UniDepth, which DROID-SLAM depends on here: launch_slam.py takes its
# intrinsics from `unidepth_intrins`, so a broken UniDepth blocks camera estimation.
# xformers deleted `xformers.components` (NystromAttention with it) after 0.0.22, and
# UniDepth's decoder constructs NystromBlock unconditionally -- so `import unidepth`
# fails outright on any modern install. The patch substitutes exact scaled-dot-product
# attention: Nystrom is a LOW-RANK APPROXIMATION of full attention, so computing it
# exactly is equal or better in accuracy, at O(n^2) rather than O(n * landmarks).
#
# Usage:  bash preproc/setup_droid_slam.sh
# Then :  export PYTHONPATH=$PWD/preproc/DROID-SLAM:$PWD/preproc/DROID-SLAM/thirdparty/lietorch
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/preproc/DROID-SLAM"

echo "== 1/4  nested submodules (recursive) =="
git -C "$D" submodule update --init --recursive

echo "== 2/4  PyTorch 2.x source patches =="
for p in lietorch-torch2:thirdparty/lietorch droid-slam-torch2:.; do
  patch_file="$ROOT/preproc/patches/${p%%:*}.patch"; target="$D/${p##*:}"
  [ -f "$patch_file" ] || { echo "   missing $patch_file"; exit 1; }
  if git -C "$target" apply --check "$patch_file" 2>/dev/null; then
    git -C "$target" apply "$patch_file"; echo "   applied ${p%%:*}"
  else
    echo "   ${p%%:*} already applied (or does not apply) -- skipping"
  fi
done

echo "== 2b/4  UniDepth xformers patch =="
UD="$ROOT/preproc/UniDepth"
UD_PATCH="$ROOT/preproc/patches/unidepth-xformers.patch"
if [ -d "$UD/.git" ] && [ -f "$UD_PATCH" ]; then
  if git -C "$UD" apply --check "$UD_PATCH" 2>/dev/null; then
    git -C "$UD" apply "$UD_PATCH"; echo "   applied unidepth-xformers"
  else
    echo "   unidepth-xformers already applied -- skipping"
  fi
  # The installed copy is what `import unidepth` resolves to, and it is a real install
  # rather than an editable one, so patching only the source tree changes nothing.
  INSTALLED="$(cd "$ROOT" && python -c "import unidepth,os;print(os.path.dirname(unidepth.__file__))" 2>/dev/null || true)"
  if [ -n "$INSTALLED" ] && [ -d "$INSTALLED" ]; then
    cp "$UD/unidepth/layers/nystrom_attention.py" "$INSTALLED/layers/nystrom_attention.py"
    echo "   synced into $INSTALLED"
  fi
fi

echo "== 3/4  toolchain =="
# Prefer an unversioned c++ if the distro has one; otherwise pin the newest g++-N.
if ! command -v c++ >/dev/null 2>&1; then
  CXX_BIN="$(ls /usr/bin/g++-* 2>/dev/null | sort -V | tail -1)"
  CC_BIN="$(ls /usr/bin/gcc-*  2>/dev/null | grep -E 'gcc-[0-9]+$' | sort -V | tail -1)"
  [ -n "$CXX_BIN" ] || { echo "   no C++ compiler found"; exit 1; }
  export CXX="$CXX_BIN" CC="$CC_BIN"
  echo "   no plain c++; using $CXX"
fi
export PATH="$ROOT/.venv/bin:$PATH"          # ninja lives here

echo "== 4/4  build extensions in place =="
(cd "$D/thirdparty/lietorch" && python setup.py build_ext --inplace)
(cd "$D" && python setup.py build_ext --inplace)

echo
echo "done. Verify with:"
echo "  PYTHONPATH=$D:$D/thirdparty/lietorch python -c 'import lietorch, droid_backends; print(\"ok\")'"
