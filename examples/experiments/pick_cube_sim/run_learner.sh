export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3

exp_name=${1:-"bin_sim"}
run_name=${2:-"six_run"}
# Directory containing your demo files
DEMO_DIR="demo_data"
DEMO_DIR="${DEMO_DIR}/${exp_name}"

# Initialize the demo paths variable
DEMO_PATHS=""

# Iterate over each pickle file in the demo_data directory
for demo_file in "$DEMO_DIR"/*.pkl; do
    # Add the current demo file as a --demo_path argument
    DEMO_PATHS+=" --demo_path=$demo_file"
done

python ../../train_rlpd_sim.py "$@" \
    --exp_name=${exp_name} \
    --checkpoint_path=/data/hil/${exp_name}/${run_name} \
    $DEMO_PATHS\
    --learner \