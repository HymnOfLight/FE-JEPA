#!/bin/sh
# Exploratory network probe for C1.3: the five label-free variants and the supervised control, one seed each,
# 3000 steps, learning rate 1e-3, run one after another on one thread.
cd "$(dirname "$0")"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for v in relu softplus3 square identity softplus2 supervised; do
    python3 -I net_probe.py "$v" 1 3000 1e-3 >> run_all.log 2>&1
done
echo "all done" >> run_all.log
