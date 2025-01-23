import torch
import gymnasium as gym
import numpy as np
from td3 import TD3Agent
import pickle
import matplotlib.pyplot as plt
from DDPG import DDPGAgent
import optparse
import pickle
def running_mean(x, N):
    # EVERY NONE BECOMES 0
    x = np.array(x)
    x[x == None] = 0
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

class NormalizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32
        )

    def action(self, action):
        low, high = self.env.action_space.low, self.env.action_space.high
        action = (action + 1.0) / 2.0 * (high - low) + low
        return np.clip(action, low, high)

    def reverse_action(self, action):
        low, high = self.env.action_space.low, self.env.action_space.high
        action = 2.0 * (action - low) / (high - low) - 1.0
        return np.clip(action, -1.0, 1.0)

def main():
    rewards = []
    losses = []
    lengths = []
    actions = []
    timestep = 0
    env_name = "Pendulum-v1"
    
    env = gym.make(env_name, render_mode="rgb_array")
    #env = NormalizeActionWrapper(env)

    # normalize action space?
    observation_space = env.observation_space
    action_space = env.action_space
    agent = TD3Agent(observation_space, action_space)
    agent = DDPGAgent(observation_space, action_space)
    max_episodes = 2000
    max_time_steps = 2000
    render = False
    seed = 0  
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    def save_stats():
        with open(f"results_td3_{env_name}_rewards.pkl", "wb") as f:
            pickle.dump({"rewards": rewards, "losses": losses, "lengths": lengths}, f)
            
    for episode in range(1, max_episodes+1):
        state, _ = env.reset()
        agent.reset()
        
        episode_reward = 0
        
        for t in range(max_time_steps):
            timestep += 1
            done = False
            action = agent.get_action(state)
            
            if render: env.render()
            (next_state, reward, done, trunc, _info) = env.step(action)
            #print("next state", next_state)
            agent.store_transition((state, action, reward, next_state, done))
            episode_reward += reward
            state = next_state
            if done or trunc:
                break
            
        losses.extend(agent.train(iter_fit=32))
        rewards.append(episode_reward)
        lengths.append(t)
        actions.append(action)
        # plot rewards every 200 episodes
        if episode % 200 == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(running_mean(rewards, 10))
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"Reward vs Episode")
            plt.savefig(f"plots/results_td3_{env_name}_rewards_episode_{episode}.png")
            plt.close()
            
            plt.figure(figsize=(10, 5))
            # losses are tuples of (critic_loss, actor_loss)
            critic_losses, actor_losses = zip(*losses)
            plt.plot(critic_losses, label="Critic Loss", color="red")
            plt.plot(actor_losses, label="Actor Loss", color="blue")
            plt.legend()
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title(f"Loss vs Iteration")
            plt.savefig(f"plots/results_td3_{env_name}_loss_episode_{episode}.png")
            plt.close()
            
            # Convert to numpy for ease of plotting
            
            # Plot the histogram for each action dimension
            actions_arr = np.array(actions)
            for i in range(actions_arr.shape[1]):
                plt.figure(figsize=(8, 5))
                plt.hist(actions_arr[:, i], bins=50, alpha=0.7, color='blue', edgecolor='black')
                plt.title(f"Action Distribution for Dimension {i+1}")
                plt.xlabel("Action Value")
                plt.ylabel("Frequency")
                plt.savefig(f"plots/results_td3_{env_name}_actions_episode_{episode}_dim_{i}.png")
                plt.close()
        if episode % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            avg_length = np.mean(lengths[-20:])
            
            print(f"Episode: {episode}, Avg Reward: {avg_reward}, Avg Length: {avg_length}")
            
    save_stats()
    
    
def main2():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env',action='store', type='string',
                         dest='env_name',default="Pendulum-v1",
                         help='Environment (default %default)')
    optParser.add_option('-n', '--eps',action='store',  type='float',
                         dest='eps',default=0.1,
                         help='Policy noise (default %default)')
    optParser.add_option('-t', '--train',action='store',  type='int',
                         dest='train',default=32,
                         help='number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr',action='store',  type='float',
                         dest='lr',default=0.0001,
                         help='learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes',action='store',  type='float',
                         dest='max_episodes',default=2000,
                         help='number of episodes (default %default)')
    optParser.add_option('-u', '--update',action='store',  type='float',
                         dest='update_every',default=100,
                         help='number of episodes between target network updates (default %default)')
    optParser.add_option('-s', '--seed',action='store',  type='int',
                         dest='seed',default=None,
                         help='random seed (default %default)')
    opts, args = optParser.parse_args()
    ############## Hyperparameters ##############
    env_name = opts.env_name
    # creating environment
    if env_name == "LunarLander-v2":
        env = gym.make(env_name, continuous = True)
    else:
        env = gym.make(env_name)
    render = False
    log_interval = 20           # print avg reward in the interval
    max_episodes = opts.max_episodes # max training episodes
    max_timesteps = 2000         # max timesteps in one episode

    train_iter = opts.train      # update networks for given batched after every episode
    eps = opts.eps               # noise of DDPG policy
    lr  = opts.lr                # learning rate of DDPG policy
    random_seed = opts.seed
    #############################################


    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    ddpg = DDPGAgent(env.observation_space, env.action_space, eps = eps, learning_rate_actor = lr,
                     update_target_every = opts.update_every)
    
    td3 = TD3Agent(env.observation_space, env.action_space)
    
    
    # logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    def save_statistics():
        with open(f"./results/td3{env_name}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-stat.pkl", 'wb') as f:
            pickle.dump({"rewards" : rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                         "lr": lr, "update_every": opts.update_every, "losses": losses}, f)

    # training loop
    for i_episode in range(1, max_episodes+1):
        ob, _info = env.reset()
        #ddpg.reset()
        total_reward=0
        for t in range(max_timesteps):
            timestep += 1
            done = False
            a = td3.get_action(ob)
            (ob_new, reward, done, trunc, _info) = env.step(a)
            total_reward+= reward
            td3.store_transition((ob, a, reward, ob_new, done))
            ob=ob_new
            if done or trunc: break

        losses.extend(td3.train(train_iter))

        rewards.append(total_reward)
        lengths.append(t)

        # save every 500 episodes
        if i_episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            torch.save(ddpg.state(), f'./results/td3_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}.pth')
            save_statistics()

        # logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
    save_statistics()



if __name__ == "__main__":
    main2()