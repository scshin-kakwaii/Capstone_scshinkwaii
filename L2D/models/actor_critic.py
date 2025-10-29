import torch.nn as nn
from models.mlp import MLPActor
from models.mlp import MLPCritic
import torch.nn.functional as F
from models.graphcnn_congForSJSSP import GraphCNN
import torch


class ActorCritic(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 # feature extraction net unique attributes:
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 # feature extraction net MLP attributes:
                 num_mlp_layers_feature_extract,
                 # actor net MLP attributes:
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # actor net MLP attributes:
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 # actor/critic/feature_extraction shared attribute
                 device
                 ):
        super(ActorCritic, self).__init__()
        # job size for problems, no business with network
        self.n_j = n_j
        # machine size for problems, no business with network
        self.n_m = n_m
        self.device = device

        # There are 8 dispatching rules in `dispatching_rules.py`
        self.num_rules = 8

        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        
        # --- CRITICAL CHANGE 1: Redefine the Actor ---
        # The actor's input is now just the global feature vector (size: hidden_dim).
        # The actor's output must be a score for each of the 8 rules (size: self.num_rules).
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim, hidden_dim_actor, self.num_rules).to(device)

        # The critic is unchanged. It always operated on the global feature vector.
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj,
                candidate,  # This will be None, no longer used
                mask,       # This will be None, no longer used
                ):

        # 1. Feature extraction is the same. It produces a global representation of the state.
        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)
        
        # --- CRITICAL CHANGE 2: Simplify the Forward Pass ---
        # 2. REMOVED all logic for preparing candidate features.
        #    - No more `torch.gather`.
        #    - No more `torch.cat`.
        
        # 3. The actor now takes the global feature 'h_pooled' directly as input.
        rule_scores = self.actor(h_pooled)

        # 4. REMOVED masking. The policy is a distribution over all 8 rules.
        
        # 5. The policy 'pi' is a probability distribution over the 8 rules.
        pi = F.softmax(rule_scores, dim=-1)

        # 6. The critic value is calculated as before.
        v = self.critic(h_pooled)
        
        return pi, v


if __name__ == '__main__':
    print('This is the ActorCritic model for RULE SELECTION.')