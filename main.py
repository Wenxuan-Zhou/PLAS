"""
Based on https://github.com/sfujim/BCQ
"""
import argparse
import gym
import numpy as np
import os
import pickle
import utils
import algos
from logger import logger, setup_logger
import d4rl
import torch
import time


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    info = {'AverageReturn': avg_reward}
    print ("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print ("---------------------------------------")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Additional parameters
    parser.add_argument("--ExpID", default=9999, type=int)              # Experiment ID
    parser.add_argument('--log_dir', default='./results/', type=str)    # Logging directory
    parser.add_argument("--load_model", default=None, type=str)         # Load model and optimizer parameters
    parser.add_argument("--save_model", default=True, type=bool)        # Save model and optimizer parameters
    parser.add_argument("--save_freq", default=1e5, type=float)         # How often it saves the model
    parser.add_argument("--env_name", default="walker2d-medium-v0")     # OpenAI gym environment name
    parser.add_argument("--algo_name", default="Latent")                # Algorithm: Latent or LatentPerturbation
    parser.add_argument("--dataset", default=None, type=str)            # path to dataset if not d4rl env
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e3, type=float)         # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=5e5, type=float)     # Max time steps to run environment for
    parser.add_argument('--vae_mode', default='train', type=str)		# VAE mode: train or load from a specific version
    parser.add_argument('--vae_itr', default=500000, type=int)		    # vae training iterations
    parser.add_argument('--vae_hidden_size', default=750, type=int)		# vae training iterations
    parser.add_argument('--max_latent_action', default=2, type=float)   # max action of the latent policy
    parser.add_argument('--phi', default=0., type=float)	            # max perturbation
    parser.add_argument('--batch_size', default=100, type=int)	        # batch size"Latent" is the latent policy and "LatentPerturbation" is the latent policy with the perturbation layer.

    parser.add_argument('--actor_lr', default=1e-4, type=float)	        # policy learning rate
    parser.add_argument('--critic_lr', default=1e-3, type=float)	    # policy learning rate
    parser.add_argument('--tau', default=0.005, type=float)	            # actor network size
    args = parser.parse_args()
    if args.dataset is None:
        args.dataset = args.env_name

    # Setup Logging
    file_name = f"Exp{args.ExpID:04d}_{args.algo_name}_{args.dataset}-{args.seed}"
    folder_name = os.path.join(args.log_dir, file_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    print(folder_name)
    if os.path.exists(os.path.join(folder_name, 'variant.json')):
        raise AssertionError
    variant = vars(args)
    variant.update(node=os.uname()[1])
    setup_logger(os.path.basename(folder_name), variant=variant, log_dir=folder_name)

    # Setup Environment
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load Dataset
    if args.env_name == args.dataset:
        dataset = d4rl.qlearning_dataset(env)  # Load d4rl dataset
    else:
        if args.dataset == 'hopper-medium-expert':
            dataset1 = d4rl.qlearning_dataset(gym.make('hopper-medium-v0'))
            dataset2 = d4rl.qlearning_dataset(gym.make('hopper-expert-v0'))
            dataset = {key:np.concatenate([dataset1[key], dataset2[key]]) for key in dataset1.keys()}
            print("Loaded data from hopper-medium-v0 and hopper-expert-v0")
        else:
            dataset_file = os.path.dirname(os.path.abspath(__file__)) + '/dataset/'+args.dataset + '.pkl'
            dataset = pickle.load(open(dataset_file,'rb'))
            print("Loaded data from "+dataset_file)

    # Train or Load VAE
    latent_dim = action_dim * 2
    vae_trainer = algos.VAEModule(state_dim, action_dim, latent_dim, max_action, vae_lr=args.vae_lr, hidden_size=args.vae_hidden_size)
    if args.vae_mode == 'train':
        # Train VAE
        print(time.ctime(), "Training VAE...")
        logs = vae_trainer.train(dataset, folder_name, iterations=args.vae_itr)
    else:
        # Select vae automatically
        vae_dirname = os.path.dirname(os.path.abspath(__file__)) + '/models/vae_' + args.vae_mode
        vae_filename = args.dataset + '-' + str(args.seed)
        vae_trainer.load(vae_filename, vae_dirname)
        print('Loaded VAE from:' + os.path.join(vae_dirname, vae_filename))

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.load(dataset)

    policy = None
    if args.algo_name == 'Latent':
        policy = algos.Latent(vae_trainer.vae, state_dim, action_dim, latent_dim, max_action,**vars(args))
    elif args.algo_name == 'LatentPerturbation':
        policy = algos.LatentPerturbation(vae_trainer.vae, state_dim, action_dim, latent_dim, max_action,**vars(args))

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0
    while training_iters < args.max_timesteps:
        # Train
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        training_iters += args.eval_freq
        print("Training iterations: " + str(training_iters))
        logger.record_tabular('Training Epochs', int(training_iters // int(args.eval_freq)))

        # Save Model
        if training_iters % args.save_freq == 0 and args.save_model:
            policy.save('model_' + str(training_iters), folder_name)

        # Eval
        info = eval_policy(policy, env)
        evaluations.append(info['AverageReturn'])
        np.save(os.path.join(folder_name, 'eval'), evaluations)

        for k, v in info.items():
            logger.record_tabular(k, v)

        logger.dump_tabular()
