# Eval last checkpoint of pure PPO
python pendulum_ppo.py --notrain --loadstep 500000 --console

# Eval checkpoint at step 200k of pure PPO
python pendulum_ppo.py --notrain --loadstep 200000 --console

# Eval last checkpoint of pure PPO + CALF wrapper
python pendulum_ppo_eval_calf_wrapper.py --notrain --loadstep 500000 --console

# Eval checkpoint at step 200k of pure PPO + CALF wrapper
python pendulum_ppo_eval_calf_wrapper.py --notrain --loadstep 200000 --console

# Get data of EnergyBasedController as a reference
python pendulum_controller.py --console
