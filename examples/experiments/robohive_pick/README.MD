
```sh
cd examples/experiments/robohive_pick

python ../../record_success_fail_sim.py --exp_name robohive_pick --successes_needed 200

python ../../train_reward_classifier.py --exp_name robohive_pick

python ../../record_demos_sim.py --exp_name robohive_pick --successes_needed 5

bash run_actor.sh
bash run_learner.sh
```

- Match env class structure
- Add wrapper for ee_pose control
- Add wrapper for delta ee_pose control



## Changes

- [x] Move robot and table forward so you dont go near singularities
- [x] Make block a cube so you can pick it up even if it falls
- [x] Keep bin lower so that you dont have to raise the block too much (causes singularities)
- [x] Have a pretraining phase of about 5 to 10k steps so the policy isnt completely random as we start. Can avoid having to give so much data in the beginning
- [x] Save weights and buffer to /data
- [x] It seems to log all the keys in info, prune out stuff we dont want, and add stuff we do
- [ ] Train across the servers??
- [ ] Add option to prematurely end episodes to save time
- [ ] Loggin extras
  - [ ] Total data in buffer
  - [x] Intervention
  - [ ] Demo data
  - [x] Policy data
  - [ ] intervation steps/intervention count
- [ ] Lower box so that the block is never occluded by it
- [ ] Make the box spawning area tighter so that weird robot positions are avoided (too close to the robot base for example)
- [ ] Implement opspace control in 