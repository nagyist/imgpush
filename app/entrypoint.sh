#!/bin/bash

WORKERS="${WORKERS:-$(getconf _NPROCESSORS_ONLN)}"

# Force single-threaded ONNX to prevent process spawning
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting uvicorn with $WORKERS workers"

nginx
uvicorn app:app --uds imgpush.sock --access-log --workers "$WORKERS" --limit-max-requests 200
