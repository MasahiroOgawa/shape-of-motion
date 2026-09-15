#!/usr/bin/env bash
# Create unversioned c++/cc symlinks for nvcc and ninja. See README.md for why.
set -euo pipefail
BIN="$(cd "$(dirname "$0")" && pwd)/bin"
mkdir -p "$BIN"
CXX=$(command -v c++ || ls /usr/bin/g++-* 2>/dev/null | grep -E 'g\+\+-[0-9]+$' | sort -V | tail -1)
CC=$(command -v cc  || ls /usr/bin/gcc-*  2>/dev/null | grep -E 'gcc-[0-9]+$'  | sort -V | tail -1)
[ -n "$CXX" ] && [ -n "$CC" ] || { echo "no C/C++ compiler found"; exit 1; }
ln -sf "$CXX" "$BIN/c++"; ln -sf "$CXX" "$BIN/g++"
ln -sf "$CC"  "$BIN/cc";  ln -sf "$CC"  "$BIN/gcc"
echo "c++ -> $CXX"; echo "cc  -> $CC"
