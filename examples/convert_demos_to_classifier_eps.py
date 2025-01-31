import datetime
import os
import numpy as np
import pickle

def get_shapes(obj):
    if isinstance(obj, list):
        return [get_shapes(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: get_shapes(v) for k, v in obj.items()}
    else:
        return np.array(obj).shape
    
exp_name = "pick_cube_sim"
success_needed = 10 

fname = "../demo_data/pick_cube_sim_10_demos_2025-01-31_13-42-32.pkl"
with open(fname, "rb") as f:
    x = pickle.load(f)

rews = [t["rewards"] for t in x]
rews = np.array(rews)
success_indexes = np.arange(len(rews))[rews == 1]
fail_indexes = np.arange(len(rews))[rews == 0]

success = [x[s] for s in success_indexes]
fail = [x[s] for s in fail_indexes]

if not os.path.exists("./classifier_data"):
    os.makedirs("./classifier_data")

uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"./classifier_data/{exp_name}_{success_needed}_success_images_{uuid}.pkl"
with open(file_name, "wb") as f:
    pickle.dump(success, f)
    print(f"saved {success_needed} successful transitions to {file_name}")

file_name = f"./classifier_data/{exp_name}_failure_images_{uuid}.pkl"
with open(file_name, "wb") as f:
    pickle.dump(fail, f)
    print(f"saved {len(fail)} failure transitions to {file_name}")
    