#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OUTPUT=${1:-"$ROOT/build/jit_dump/loader.so"}
PYTHON=${PYTHON:-python3}
CXX=${CXX:-c++}
MUSA_HOME=${MUSA_HOME:-/usr/local/musa}

TVM_FFI_INCLUDE=$(
  "$PYTHON" -c 'from tvm_ffi.libinfo import find_include_path; print(find_include_path())'
)
TVM_FFI_LIBRARY=$(
  "$PYTHON" -c 'from tvm_ffi.libinfo import find_libtvm_ffi; print(find_libtvm_ffi())'
)
TVM_FFI_LIBDIR=$(dirname "$TVM_FFI_LIBRARY")

mkdir -p "$(dirname "$OUTPUT")"
"$CXX" \
  -std=c++17 \
  -O2 \
  -fPIC \
  -shared \
  "$ROOT/tools/jit_dump/loader.cpp" \
  -I"$TVM_FFI_INCLUDE" \
  -I"$MUSA_HOME/include" \
  -L"$TVM_FFI_LIBDIR" \
  -ltvm_ffi \
  -L"$MUSA_HOME/lib" \
  -lmusart \
  -lmusa \
  -o "$OUTPUT"

echo "$OUTPUT"
