#!/bin/sh
# Second seed for the three constrained variants (softplus, ReLU, square), as run_all.sh otherwise.
cd "$(dirname "$0")"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for v in softplus3 relu square; do
    python3 -I net_probe.py "$v" 2 3000 1e-3 >> run_seed2.log 2>&1
done
echo "all done" >> run_seed2.log
