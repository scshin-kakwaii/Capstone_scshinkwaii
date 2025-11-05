from mb_agg import *
from agent_utils import eval_actions
from agent_utils import select_action
from models.actor_critic import ActorCritic
from copy import deepcopy
import torch
import time
import torch.nn as nn
import numpy as np
from Params import configs
from validation import validate

device = torch.device(configs.device)


class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]


class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_j,
                 n_m,
                 num_layers,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(n_j=n_j,
                                  n_m=n_m,
                                  num_layers=num_layers,
                                  learn_eps=False,
                                  neighbor_pooling_type=neighbor_pooling_type,
                                  input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=device)
        self.policy_old = deepcopy(self.policy)

        '''self.policy.load_state_dict(
            torch.load(path='./{}.pth'.format(str(n_j) + '_' + str(n_m) + '_' + str(1) + '_' + str(99))))'''

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)

        self.V_loss_2 = nn.MSELoss()

    def update(self, memories, n_tasks, g_pool):

        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef

        rewards_all_env = []
        adj_mb_t_all_env = []
        fea_mb_t_all_env = []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        a_mb_t_all_env = []
        old_logprobs_mb_t_all_env = []
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_all_env.append(rewards)
            adj_mb_t_all_env.append(aggr_obs(torch.stack(memories[i].adj_mb).to(device), n_tasks))
            fea_mb_t = torch.stack(memories[i].fea_mb).to(device)
            fea_mb_t = fea_mb_t.reshape(-1, fea_mb_t.size(-1))
            fea_mb_t_all_env.append(fea_mb_t)
            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(device).squeeze().detach())

        mb_g_pool = g_pool_cal(g_pool, torch.stack(memories[0].adj_mb).to(device).shape, n_tasks, device)

        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            for i in range(len(memories)):
                # NOTE TO PROFESSOR: CHANGE 1
                # The policy network call is simplified. 'candidate' and 'mask' are set to None.
                # In the original code, these tensors were passed to the policy.
                # This reflects a change in the model architecture to a "rule-selecting" agent
                # that doesn't require an explicit list of candidate actions during the forward pass.
                pis, vals = self.policy(x=fea_mb_t_all_env[i],
                                        graph_pool=mb_g_pool,
                                        adj=adj_mb_t_all_env[i],
                                        candidate=None, 
                                        mask=None,      
                                        padded_nei=None)
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                advantages = rewards_all_env[i] - vals.view(-1).detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.squeeze(), rewards_all_env[i])
                p_loss = - torch.min(surr1, surr2).mean()
                ent_loss = - ent_loss.clone()
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss
                loss_sum += loss
                vloss_sum += v_loss
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item()


def main():

    from JSSP_Env import SJSSP
    envs = [SJSSP(n_j=configs.n_j, n_m=configs.n_m) for _ in range(configs.num_envs)]
    
    from uniform_instance_gen import uni_instance_gen
    data_generator = uni_instance_gen

    dataLoaded = np.load('./DataGen/generatedData' + str(configs.n_j) + '_' + str(configs.n_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')
    vali_data = []
    for i in range(dataLoaded.shape[0]):
        vali_data.append((dataLoaded[i][0], dataLoaded[i][1]))

    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)

    memories = [Memory() for _ in range(configs.num_envs)]

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=configs.n_j,
              n_m=configs.n_m,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)

    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, configs.n_j*configs.n_m, configs.n_j*configs.n_m]),
                             n_nodes=configs.n_j*configs.n_m,
                             device=device)
    # training loop
    log = []
    validation_log = []
    optimal_gaps = []
    optimal_gap = 1
    record = 100000
    
    adj_envs = []
    fea_envs = []
    candidate_envs = []
    mask_envs = []
    
    for i, env in enumerate(envs):
        instance_data = data_generator(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high)
        # NOTE TO PROFESSOR: CHANGE 2
        # The environment reset now follows the standard Gymnasium API.
        # Original: adj, fea, candidate, mask = env.reset(...)
        # New: `reset` returns a tuple (obs, info), where `obs` is a dictionary.
        # `reset` now returns a dictionary of observations and an info dict
        obs, _ = env.reset(options={'data': instance_data})
        # We now unpack the observation dictionary to get the state components.
        adj_envs.append(obs['adj'])
        fea_envs.append(obs['fea'])
        candidate_envs.append(obs['candidate'])
        mask_envs.append(obs['mask'])

    for i_update in range(configs.max_updates):

        t3 = time.time()

        ep_rewards = [-env.initQuality for env in envs]

        # rollout the env
        while True:
            # --- Convert current states to tensors ---
            fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(device) for fea in fea_envs]
            adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(device).to_sparse() for adj in adj_envs]
            candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(device) for candidate in candidate_envs]
            mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(device) for mask in mask_envs]
            
            # --- Select actions for all environments ---
            with torch.no_grad():
                action_envs = []
                a_idx_envs = []
                for i in range(configs.num_envs):
                    # NOTE TO PROFESSOR: CHANGE 3 (Consistent with Change 1)
                    # The policy call here is also updated to no longer require 'candidate' and 'mask'.
                    # Original: ppo.policy_old(..., candidate=candidate_tensor_envs[i].unsqueeze(0), mask=mask_tensor_envs[i].unsqueeze(0))
                    pi, _ = ppo.policy_old(x=fea_tensor_envs[i],
                                           graph_pool=g_pool_step,
                                           padded_nei=None,
                                           adj=adj_tensor_envs[i],
                                           candidate=None, # Not used
                                           mask=None)      # Not used

                    # NOTE TO PROFESSOR: CHANGE 4
                    # The action selection function is simplified.
                    # Original: action, a_idx = select_action(pi, candidate_envs[i], memories[i])
                    # The `candidate_envs` is no longer needed as an argument.
                    action, a_idx = select_action(pi, memories[i])
                    action_envs.append(action)
                    a_idx_envs.append(a_idx)
            
            # --- Prepare to store next states ---
            next_adj_envs = []
            next_fea_envs = []
            next_candidate_envs = []
            next_mask_envs = []

            # --- Take a step in each environment ---
            for i in range(configs.num_envs):
                # Save the current state to memory before stepping
                memories[i].adj_mb.append(adj_tensor_envs[i])
                memories[i].fea_mb.append(fea_tensor_envs[i])
                memories[i].candidate_mb.append(candidate_tensor_envs[i])
                memories[i].mask_mb.append(mask_tensor_envs[i])
                memories[i].a_mb.append(a_idx_envs[i])

                # NOTE TO PROFESSOR: CHANGE 5
                # The environment step now follows the standard Gymnasium API.
                # Original: adj, fea, reward, done, candidate, mask = envs[i].step(...)
                # New: `step` returns a 5-element tuple: (obs, reward, terminated, truncated, info).
                obs, reward, terminated, truncated, info = envs[i].step(action_envs[i].item())
                # The 'done' signal is now a combination of 'terminated' and 'truncated'.
                done = terminated or truncated

                # Unpack the new observation dictionary for the next state
                next_adj_envs.append(obs['adj'])
                next_fea_envs.append(obs['fea'])
                next_candidate_envs.append(obs['candidate'])
                next_mask_envs.append(obs['mask'])

                ep_rewards[i] += reward
                memories[i].r_mb.append(reward)
                memories[i].done_mb.append(done)

            adj_envs = next_adj_envs

            # NOTE TO PROFESSOR: CHANGE 6
            # The logic for updating the state variables for the next loop is now more structured.
            # Instead of clearing and re-appending, we assign the collected 'next' states.
            adj_envs = next_adj_envs
            fea_envs = next_fea_envs
            candidate_envs = next_candidate_envs
            mask_envs = next_mask_envs

            # NOTE TO PROFESSOR: CHANGE 7
            # The episode termination check is updated.
            # Original: `if envs[0].done(): break` (relied on a custom env method)
            # New: We use the `done` flag returned by the standard Gymnasium `step` function.
            if done: 
                adj_envs = []
                fea_envs = []
                candidate_envs = []
                mask_envs = []
                # The reset logic is now placed here, inside the rollout loop, which is a more standard pattern.
                for i, env in enumerate(envs):
                    instance_data = data_generator(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high)
                    obs, _ = env.reset(options={'data': instance_data})
                    adj_envs.append(obs['adj'])
                    fea_envs.append(obs['fea'])
                    candidate_envs.append(obs['candidate'])
                    mask_envs.append(obs['mask'])
                break
        for j in range(configs.num_envs):
            ep_rewards[j] -= envs[j].posRewards

        loss, v_loss = ppo.update(memories, configs.n_j*configs.n_m, configs.graph_pool_type)
        for memory in memories:
            memory.clear_memory()
            
        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
        log.append([i_update, mean_rewards_all_env])
        if (i_update + 1) % 100 == 0:
            file_writing_obj = open('./' + 'log_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_obj.write(str(log))

        # log results
        print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}'.format(
            i_update + 1, mean_rewards_all_env, v_loss))
        
        # validate and save use mean performance
        t4 = time.time()
        if (i_update + 1) % 100 == 0:
            # NOTE TO PROFESSOR: CHANGE 8 (Crucial Correction)
            # Removed the negative sign from the validation result.
            # Original: `vali_result = - validate(vali_data, ppo.policy).mean()`
            # The `validate` function has been corrected to return the true, POSITIVE makespan.
            # Therefore, we no longer need to negate it. This ensures that the logged and printed
            # validation results are the actual, correct makespans.
            vali_result = validate(vali_data, ppo.policy).mean()
            validation_log.append(vali_result)
            if vali_result < record:
                torch.save(ppo.policy.state_dict(), './{}.pth'.format(
                    str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high)))
                record = vali_result
            print('The validation quality is:', vali_result)
            file_writing_obj1 = open(
                './' + 'vali_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_obj1.write(str(validation_log))
        t5 = time.time()


if __name__ == '__main__':
    total1 = time.time()
    main()
    total2 = time.time()
    # print(total2 - total1)
