export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \

exp_name=${1:-"bin_sim"}
run_name=${2:-"six_run"}

python ../../train_rlpd_sim.py "$@" \
    --exp_name=${exp_name} \
    --checkpoint_path=/data/hil/${exp_name}/${run_name} \
    --actor \
