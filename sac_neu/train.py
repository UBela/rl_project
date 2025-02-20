import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sac_agent import SACAgent
from trainer import SACTrainer
import torch
from hockey import hockey_env as h_env
import optparse
import time

optParser = optparse.OptionParser()
os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Training parameters
optParser.add_option("--max_episodes", type=int, default=25_000)
optParser.add_option("--max_timesteps", type=int, default=1000)
optParser.add_option("--iter_fit", type=int, default=32)
optParser.add_option("--log_interval", type=int, default=20)
optParser.add_option("--random_seed", type=int, default=None)
optParser.add_option("--render", action="store_true", default=False)
optParser.add_option("--use_hard_opp", action="store_true", default=True)
optParser.add_option("--use_self_play", action="store_true", default=True)
optParser.add_option("--self_play_intervall", type=int, default=50_000)
optParser.add_option("--self_play_start", type=int, default=16_000)
optParser.add_option("--evaluate_every", type=int, default=2000)
optParser.add_option("--results_folder", type=str, default="./results")
optParser.add_option("--use_PER", action="store_true", default=False)
optParser.add_option("--per_alpha", help="Alpha for PER", type=float, default=0.6)
optParser.add_option("--per_beta", help="Beta for PER", type=float, default=0.4)
optParser.add_option("--per_beta_update", help="Beta update for PER", type=float, default=None)

# SAC-specific agent parameters
optParser.add_option("--policy_lr", type=float, default=1e-4)
optParser.add_option("--q_lr", type=float, default=1e-3)
optParser.add_option("--value_lr", type=float, default=1e-3)
optParser.add_option("--tau", type=float, default=0.005)
optParser.add_option("--gamma", type=float, default=0.99)
optParser.add_option("--alpha", type=float, default=0.2)
optParser.add_option("--automatic_entropy_tuning", action="store_true", default=False)
optParser.add_option("--batch_size", type=int, default=128)
optParser.add_option("--buffer_size", type=int, default=int(2**20))

opts, _ = optParser.parse_args()

if __name__ == '__main__':
    print("### Start preparation ###")
    print("### Options ###")
    print(opts)
    start_prep_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")   

    env = h_env.HockeyEnv()
    weak_opponent = h_env.BasicOpponent(weak=True)
    strong_opponent = h_env.BasicOpponent(weak=False) 
    
    opponents = [weak_opponent, strong_opponent]
    
    
    agent = SACAgent(
        state_dim=env.observation_space.shape[0], 
        action_dim=env.action_space.shape[0], 
        hidden_dim=256, 
        gamma=opts.gamma, 
        tau=opts.tau,
        alpha=opts.alpha,
        automatic_entropy_tuning=opts.automatic_entropy_tuning,
        policy_lr=opts.policy_lr,
        q_lr=opts.q_lr,
        value_lr=opts.value_lr,
        device=device
    )
    
    print("### Agent created ###")
    
    trainer = SACTrainer(vars(opts))
    print("### Trainer created ###")
    
    print("### Start training..... ###")
    end_prep_time = time.time()
    print(f"Preparation time: {end_prep_time - start_prep_time} seconds.")
    
    start_time = time.time()

    trainer.train(agent, opponents, env)
    
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")
