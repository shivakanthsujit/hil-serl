export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \

# Directory containing your demo files
DEMO_DIR="demo_data"

# Initialize the demo paths variable
DEMO_PATHS=""

# Iterate over each pickle file in the demo_data directory
for demo_file in "$DEMO_DIR"/*.pkl; do
    # Add the current demo file as a --demo_path argument
    DEMO_PATHS+=" --demo_path=$demo_file"
done

python ../../train_rlpd_sim.py "$@" \
    --exp_name=robohive_pick \
    --checkpoint_path=/data/hil/six_run \
    $DEMO_PATHS\
    --learner \