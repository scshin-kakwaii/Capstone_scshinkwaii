import torch
import numpy as np
from JSSP_Env import SJSSP
from mb_agg import g_pool_cal
from agent_utils import greedy_select_action  # <<< CORRECTED IMPORT
from Params import configs

device = torch.device(configs.device)

def validate(validation_data, policy):
    """
    Validates the "rule-selecting" policy on a set of problem instances.
    """
    # Extract problem size from the first instance
    N_JOBS = validation_data[0][0].shape[0]
    N_MACHINES = validation_data[0][0].shape[1]

    # Create a single environment for validation
    env = SJSSP(n_j=N_JOBS, n_m=N_MACHINES)
    
    # g_pool_step is not used by the new rule-selecting model's forward pass,
    # but we create it to match the training call signature if needed.
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                             n_nodes=env.number_of_tasks,
                             device=device)
    
    makespans = []
    
    # Rollout each instance in the validation set
    for instance_data in validation_data:
        # --- CRITICAL CHANGE: Use Gymnasium API for reset ---
        obs, _ = env.reset(options={'data': instance_data})
        
        done = False
        while not done:
            # Convert observation dictionary to tensors
            fea_tensor = torch.from_numpy(np.copy(obs['fea'])).to(device)
            adj_tensor = torch.from_numpy(np.copy(obs['adj'])).to(device).to_sparse()
            
            with torch.no_grad():
                # --- CRITICAL CHANGE: Call policy without candidate/mask ---
                pi, _ = policy(x=fea_tensor,
                               graph_pool=g_pool_step,
                               padded_nei=None,
                               adj=adj_tensor,
                               candidate=None,  # Not used by rule-selecting agent
                               mask=None)       # Not used by rule-selecting agent

            # Select the best rule (greedy)
            rule_index = greedy_select_action(pi)
            
            # --- CRITICAL CHANGE: Use Gymnasium API for step ---
            obs, reward, terminated, truncated, info = env.step(rule_index.item())
            done = terminated or truncated
        
        # After the episode is done, record the final makespan
        makespans.append(env.max_endTime)
        
    return np.array(makespans)


if __name__ == '__main__':
    # This block allows you to run validation as a standalone script.
    # It has been updated to be consistent with the changes above.
    
    from uniform_instance_gen import uni_instance_gen
    import argparse
    from PPO_jssp_multiInstances import PPO

    parser = argparse.ArgumentParser(description='Arguments for validating ppo_jssp')
    parser.add_argument('--Pn_j', type=int, default=15, help='Number of jobs of instances to test')
    parser.add_argument('--Pn_m', type=int, default=15, help='Number of machines instances to test')
    parser.add_argument('--Nn_j', type=int, default=15, help='Number of jobs of the trained network')
    parser.add_argument('--Nn_m', type=int, default=15, help='Number of machines of the trained network')
    parser.add_argument('--low', type=int, default=1, help='LB of duration')
    parser.add_argument('--high', type=int, default=99, help='UB of duration')
    parser.add_argument('--seed', type=int, default=200, help='Max seed for validation set generation')
    parser.add_argument('--n_vali', type=int, default=100, help='Validation set size')
    params = parser.parse_args()

    N_JOBS_P = params.Pn_j
    N_MACHINES_P = params.Pn_m
    LOW = params.low
    HIGH = params.high
    N_JOBS_N = params.Nn_j
    N_MACHINES_N = params.Nn_m

    # Initialize a PPO agent to load the policy into
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=N_JOBS_N,  # Size of the trained model
              n_m=N_MACHINES_N,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)

    # Load the saved model weights
    path = './{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))
    ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu'))) # Use map_location for CPU if needed

    print(f"Validating model {path} on {params.Pn_j}x{params.Pn_m} instances.")

    SEEDs = range(0, params.seed, 10)
    all_results = []
    for SEED in SEEDs:
        print(f"Running with random seed: {SEED}")
        np.random.seed(SEED)
        
        # Generate a validation set of instances
        vali_data = [uni_instance_gen(n_j=N_JOBS_P, n_m=N_MACHINES_P, low=LOW, high=HIGH) for _ in range(params.n_vali)]

        # The validate function returns positive makespans
        makespans = validate(vali_data, ppo.policy)
        mean_makespan = makespans.mean()
        all_results.append(mean_makespan)
        print(f"Mean makespan for this seed: {mean_makespan:.2f}")
    
    print("\n-------------------------------------------")
    print(f"Overall Mean Makespan across all seeds: {np.mean(all_results):.2f}")
    print("-------------------------------------------")