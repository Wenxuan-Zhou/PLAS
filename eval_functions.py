import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reward_to_return(reward_arr, discount=0.99):
    assert type(reward_arr) == np.ndarray
    discount_factor = discount ** np.arange(len(reward_arr))
    return np.sum(reward_arr * discount_factor)


def eval_critic(select_action, critic, eval_env, eval_episodes=10, tf=False):

    Q_list = []
    true_return = []
    sum_reward = []

    max_step = eval_env._max_episode_steps
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        reward_list = []
        step_count = 0
        while not done and step_count < 2 * max_step:
            action = select_action(np.array(state))
            if step_count < max_step:
                if tf:
                    Q_list.append(critic(state, action))
                else:
                    state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
                    action_tensor = torch.FloatTensor(action.reshape(1, -1)).to(device)
                    Q_list.append(critic(state_tensor, action_tensor).cpu().data.numpy().reshape(-1)[0])
            state, reward, done, info = eval_env.env.step(action)  # Bypass Timelimit
            reward_list.append(reward)
            step_count += 1

        reward_list = np.array(reward_list)

        for start_n in range(min(max_step, len(reward_list))):
            end_n = min(start_n + max_step, len(reward_list))
            true_return.append(reward_to_return(reward_list[start_n: end_n]))

        sum_reward.append(np.sum(reward_list[: max_step]))

    eval_dict = dict()
    true_return = np.array(true_return).reshape(-1)
    Q_list = np.array(Q_list).reshape(-1)
    error = Q_list - true_return
    eval_dict['Reward'] = np.average(sum_reward)
    print('Average Reward', np.average(sum_reward))
    eval_dict['Return'] = np.average(true_return)
    eval_dict['Average_Q'] = np.average(Q_list)
    eval_dict['Error Mean'] = np.average(error)
    eval_dict['Error Std'] = np.std(Q_list - true_return)
    eval_dict['MSE'] = np.mean(np.square(np.array(true_return) - np.array(Q_list)))
    eval_dict['Positive Error Percentage'] = np.mean(error > 0)
    if np.sum(error > 0) > 0:
        eval_dict['Positive Error Mean'] = np.mean(error[error > 0])
        eval_dict['Positive Error Std'] = np.std(error[error > 0])
    else:
        eval_dict['Positive Error Mean'] = 0
        eval_dict['Positive Error Std'] = 0

    if np.sum(error < 0) > 0:
        eval_dict['Negative Error Mean'] = np.mean(error[error < 0])
        eval_dict['Negative Error Std'] = np.std(error[error < 0])
    else:
        eval_dict['Negative Error Mean'] = 0
        eval_dict['Negative Error Std'] = 0

    return eval_dict
