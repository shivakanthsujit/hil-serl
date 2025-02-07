export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd_sim.py "$@" \
    --exp_name=robohive_pick \
    --checkpoint_path=/data/hil/six_run \
    --actor \
