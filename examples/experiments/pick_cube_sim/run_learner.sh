export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
# python ../../train_rlpd_sim.py "$@" \
#     --exp_name=pick_cube_sim \
#     --checkpoint_path=six_run \
#     --demo_path=demo_data/pick_cube_sim_10_demos_2025-01-31_13-42-32.pkl\
#     --learner \

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
    --exp_name=pick_cube_sim \
    --checkpoint_path=six_run \
    $DEMO_PATHS\
    --learner \