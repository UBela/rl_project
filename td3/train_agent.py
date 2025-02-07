import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from td3.td3_agent import TD3Agent
from td3.trainer import TD3Trainer
import torch
from hockey import hockey_env as h_env
import optparse
from td3.utils import RandomAgent
import sys
import time



optParser = optparse.OptionParser()

# Training parameters
optParser.add_option("--episodes_on_random", type=int, default=5000)
optParser.add_option("--episodes_on_weak", type=int, default=10000)
optParser.add_option("--max_episodes", type=int, default=25000)
optParser.add_option("--max_timesteps", type=int, default=1000)
optParser.add_option("--iter_fit", type=int, default=32)
optParser.add_option("--log_interval", type=int, default=20)
optParser.add_option("--random_seed", type=int, default=None)
optParser.add_option("--render", action="store_true", default=False)
optParser.add_option("--use_hard_opp", action="store_true", default=True)
optParser.add_option("--use_curr_learning", action="store_true", default=False)
optParser.add_option("--evaluate_every", type=int, default=2000)
optParser.add_option("--results_folder", type=str, default="/home/stud311/work/rl_project/td3/results")

# agent parameters
optParser.add_option("--actor_lr", type=float, default=0.0001)
optParser.add_option("--critic_lr", type=float, default=0.0001)
optParser.add_option("--tau", type=float, default=0.005)
optParser.add_option("--gamma", type=float, default=0.99)
optParser.add_option("--noise_std", type=float, default=0.2)
optParser.add_option("--noise_clip", type=float, default=0.5)
optParser.add_option("--exploration_steps", type=int, default=2000)
optParser.add_option("--batch_size", type=int, default=128)
optParser.add_option("--buffer_size", type=int, default=int(1e6))
optParser.add_option("--policy_update_freq", type=int, default=2)


opts, _ = optParser.parse_args()

if __name__ == '__main__':
    start_prep_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")   
    env = h_env.HockeyEnv(h_env.Mode.NORMAL)
    random_opponent = RandomAgent(env.observation_space, env.action_space)
    weak_opponent = h_env.BasicOpponent(weak=True)
    strong_opponent = h_env.BasicOpponent(weak=False) if opts.use_hard_opp else weak_opponent
    
    opponents = [random_opponent, weak_opponent, strong_opponent]
    
    agent = TD3Agent(env.observation_space, env.action_space, device=device, userconfig=vars(opts))
    print("### Agent created ###")
    trainer = TD3Trainer(vars(opts))
    print("### Trainer created ###")
    print("### Start training..... ###")
    end_prep_time = time.time()
    print(f"Preparation time: {end_prep_time - start_prep_time} seconds.")
    start_time = time.time()
    trainer.train(agent, opponents, env)
    
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")

