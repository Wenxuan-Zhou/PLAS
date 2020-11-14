"""
Based on https://github.com/sfujim/BCQ
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pickle
from logger import logger
from logger import create_stats_ordered_dict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.hidden_size = (400, 300)

        self.l1 = nn.Linear(state_dim, self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class ActorPerturbation(nn.Module):
    def __init__(self, state_dim, action_dim, latent_action_dim, max_action, max_latent_action=2, phi=0.05):
        super(ActorPerturbation, self).__init__()

        self.hidden_size = (400, 300, 400, 300)

        self.l1 = nn.Linear(state_dim, self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], latent_action_dim)

        self.l4 = nn.Linear(state_dim + action_dim, self.hidden_size[2])
        self.l5 = nn.Linear(self.hidden_size[2], self.hidden_size[3])
        self.l6 = nn.Linear(self.hidden_size[3], action_dim)

        self.max_latent_action = max_latent_action
        self.max_action = max_action
        self.phi = phi

    def forward(self, state, decoder):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        latent_action = self.max_latent_action * torch.tanh(self.l3(a))

        mid_action = decoder(state, z=latent_action)

        a = F.relu(self.l4(torch.cat([state, mid_action], 1)))
        a = F.relu(self.l5(a))
        a = self.phi * torch.tanh(self.l6(a))
        final_action = (a + mid_action).clamp(-self.max_action, self.max_action)
        return latent_action, mid_action, final_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.hidden_size = (400, 300)

        self.l1 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], 1)

        self.l4 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        self.l5 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l6 = nn.Linear(self.hidden_size[1], 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden_size=750):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)
        self.d3 = nn.Linear(hidden_size, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None, clip=None, raw=False):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(device)
            if clip is not None:
                z = z.clamp(-clip, clip)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        a = self.d3(a)
        if raw: return a
        return self.max_action * torch.tanh(a)


class VAEModule(object):
    def __init__(self, *args, vae_lr=1e-4, **kwargs):
        self.vae = VAE(*args, **kwargs).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)

    def train(self, dataset, folder_name, batch_size=100, iterations=500000):
        logs = {'vae_loss': [], 'recon_loss': [], 'kl_loss': []}
        for i in range(iterations):
            vae_loss, recon_loss, KL_loss = self.train_step(dataset, batch_size)
            logs['vae_loss'].append(vae_loss)
            logs['recon_loss'].append(recon_loss)
            logs['kl_loss'].append(KL_loss)
            if (i + 1) % 50000 == 0:
                print('Itr ' + str(i+1) + ' Training loss:' + '{:.4}'.format(vae_loss))
                self.save('model_' + str(i+1), folder_name)
                pickle.dump(logs, open(folder_name + "/vae_logs.p", "wb"))

        return logs

    def loss(self, state, action):
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        recon, mean, std = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss
        return vae_loss, recon_loss, KL_loss

    def train_step(self, dataset, batch_size=100):
        dataset_size = len(dataset['observations'])
        ind = np.random.randint(0, dataset_size, size=batch_size)
        state = dataset['observations'][ind]
        action = dataset['actions'][ind]
        vae_loss, recon_loss, KL_loss = self.loss(state, action)
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        return vae_loss.cpu().data.numpy(), recon_loss.cpu().data.numpy(), KL_loss.cpu().data.numpy()

    def save(self, filename, directory):
        torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))

    def load(self, filename, directory):
        self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename), map_location=device))


class Latent(object):
    def __init__(self, vae, state_dim, action_dim, latent_dim, max_action, discount=0.99, tau=0.005,
                 actor_lr=1e-3, critic_lr=1e-3, lmbda=0.75, max_latent_action=2, **kwargs):
        self.actor = Actor(state_dim, latent_dim, max_latent_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.latent_dim = latent_dim
        self.vae = vae
        self.max_action = max_action
        self.max_latent_action = max_latent_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.vae.decode(state, z=self.actor(state))
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # Critic Training
            with torch.no_grad():
                next_latent_action = self.actor_target(next_state)
                next_action = self.vae.decode(next_state, z=next_latent_action)

                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1 - self.lmbda) * torch.max(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor Training
            latent_actions = self.actor(state)
            actions = self.vae.decode(state, z=latent_actions)
            actor_loss = -self.critic.q1(state, actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Logging
        # logger.record_dict(create_stats_ordered_dict('Noise', noise.cpu().data.numpy(),))
        logger.record_dict(create_stats_ordered_dict('Q_target', target_Q.cpu().data.numpy(),))
        logger.record_tabular('Actor Loss', actor_loss.cpu().data.numpy())
        logger.record_tabular('Critic Loss', critic_loss.cpu().data.numpy())
        logger.record_dict(create_stats_ordered_dict('Actions', actions.cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Latent Actions', latent_actions.cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Latent Actions Norm', torch.norm(latent_actions, dim=1).cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Current_Q', current_Q1.cpu().data.numpy()))
        assert (np.abs(np.mean(target_Q.cpu().data.numpy())) < 1e6)

    def save(self, filename, directory):
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))

        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))

    def load(self, filename, directory):
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pth' % (directory, filename)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.actor_target = copy.deepcopy(self.actor)


class LatentPerturbation(object):
    def __init__(self, vae, state_dim, action_dim, latent_dim, max_action, discount=0.99, tau=0.005,
                 actor_lr=1e-3, critic_lr=1e-3, lmbda=0.75, max_latent_action=2, phi=0.05, **kwargs):
        self.actor = ActorPerturbation(state_dim, action_dim, latent_dim, max_action,
                                       max_latent_action=max_latent_action, phi=phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.vae = vae
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            _, _, action = self.actor(state, self.vae.decode)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # Critic Training
            with torch.no_grad():
                _, _, next_action = self.actor_target(next_state, self.vae.decode)
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1 - self.lmbda) * torch.max(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor Training
            latent_actions, mid_actions, actions = self.actor(state, self.vae.decode)
            actor_loss = -self.critic.q1(state, actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Logging
        logger.record_dict(create_stats_ordered_dict('Q_target', target_Q.cpu().data.numpy(),))
        logger.record_tabular('Actor Loss', actor_loss.cpu().data.numpy())
        logger.record_tabular('Critic Loss', critic_loss.cpu().data.numpy())
        logger.record_dict(create_stats_ordered_dict('Actions', actions.cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Mid Actions', mid_actions.cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Latent Actions', latent_actions.cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Latent Actions Norm', torch.norm(latent_actions, dim=1).cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Perturbation Norm', torch.norm(actions-mid_actions, dim=1).cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Current_Q', current_Q1.cpu().data.numpy()))
        assert (np.abs(np.mean(target_Q.cpu().data.numpy())) < 1e6)

    def save(self, filename, directory):
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))

        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))

    def load(self, filename, directory):
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pth' % (directory, filename)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.actor_target = copy.deepcopy(self.actor)
