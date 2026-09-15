# Compatibility shims for modern toolchains (CUDA 13 / Blackwell)

Put `preproc/compat` on `PYTHONPATH` and `preproc/compat/bin` on `PATH` before running
anything in `preproc/` or `scripts/`. Two unrelated gaps are covered here.

## `torch_scatter.py` — pure-PyTorch stand-in

`torch_scatter` is a compiled extension pinned to an exact torch build, and no wheel
exists for torch>=2.14+cu130, the combination Blackwell (sm_120) requires. DROID-SLAM
imports only `scatter_sum` and `scatter_mean`, and upstream implements both as composite
ops over `Tensor.scatter_add_` rather than as custom kernels — so the shim is the same
computation, not an approximation. Drop it from `PYTHONPATH` if you ever build the real
package.

## `bin/c++`, `bin/cc` — unversioned host-compiler names

nvcc invokes the host compiler as plain `c++`, and ninja as plain `c++`/`cc`. Distros
that ship only versioned compilers (Ubuntu 24.04 has `g++-13`, no `g++`) therefore fail:
nvcc reports `nvcc fatal : Failed to preprocess host compiler properties` and ninja
reports `FAILED: [code=127]` — 127 being command-not-found. Neither message names the
missing binary, so the cause is invisible without knowing to look for it.

Regenerate the symlinks with `bash preproc/compat/make_cxx_shim.sh`.
