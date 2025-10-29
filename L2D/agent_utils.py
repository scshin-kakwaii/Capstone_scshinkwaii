# agent_utils.py (Corrected for Rule Selection)

import torch
from torch.distributions.categorical import Categorical

def select_action(p, memory):
    """
    Selects a dispatching rule index from the policy distribution.
    """
    dist = Categorical(p.squeeze())
    rule_index = dist.sample()
    if memory is not None:
        memory.logprobs.append(dist.log_prob(rule_index))
    # The action and the index are the same in this design
    return rule_index, rule_index

def eval_actions(p, actions):
    """
    Evaluates actions for the PPO update step.
    """
    softmax_dist = Categorical(p)
    log_probs = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return log_probs, entropy

def greedy_select_action(p):
    """
    Selects the rule with the highest probability for validation.
    """
    # No mask is needed as all 8 rules are always considered.
    _, rule_index = p.squeeze().max(0)
    return rule_index