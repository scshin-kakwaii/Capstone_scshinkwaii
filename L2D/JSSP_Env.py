import gymnasium as gym #--- CHANGE ---: Original Gym API, Switched to Gymnasium (new API)
from gymnasium import spaces #--- CHANGE ---: Explicitly import spaces for defining action/observation spaces
import numpy as np
from gymnasium.utils import EzPickle
from uniform_instance_gen import override
from updateEntTimeLB import calEndTimeLB
from Params import configs
from permissibleLS import permissibleLeftShift
from updateAdjMat import getActionNbghs

# --- CHANGE ---: new import for dispatching rule logic
from dispatching_rules import Rules, apply_dispatching_rule

class SJSSP(gym.Env, EzPickle):
    def __init__(self, n_j, n_m):
        EzPickle.__init__(self)

        # Basic problem parameters
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = n_j * n_m

        # Original code: no action or observation space defined
        # --- CRITICAL CHANGES ---: Gymnasium API Compliance ---
        self.action_space = spaces.Discrete(self.number_of_jobs) # This is like the original code: The action is the index of the operation to schedule from the list of available candidates 
        self.action_space = spaces.Discrete(8) # However, we overwrites above, the action space now corresponds to 8 dispatching rules

        # --- CHANGE ---: Added observation space as Dict type
        # The observation is a dictionary containing all the state information.
        self.observation_space = spaces.Dict({
            "adj": spaces.Box(low=0, high=1, shape=(self.number_of_tasks, self.number_of_tasks), dtype=np.single),
            "fea": spaces.Box(low=-1e6, high=1e6, shape=(self.number_of_tasks, configs.input_dim), dtype=np.single),
            "candidate": spaces.Box(low=0, high=self.number_of_tasks, shape=(self.number_of_jobs,), dtype=np.int64),
            "mask": spaces.Box(low=0, high=1, shape=(self.number_of_jobs,), dtype=bool)
        })
        
        # Static information
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]

        # Utility functions
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs

    # --- CHANGE ---: Added helper method to construct observation dictionary
    def _get_obs(self):
        """
        Helper function to create the observation dictionary from the current state.
        """
        features = np.concatenate(
            (self.LBs.reshape(-1, 1) / configs.et_normalize_coef,
             self.finished_mark.reshape(-1, 1)),
            axis=1
        )
        return {
            "adj": self.adj,
            "fea": features,
            "candidate": self.omega,
            "mask": self.mask
        }

    def done(self):
        """Check if the episode is finished."""
        return len(self.partial_sol_sequeence) == self.number_of_tasks

    @override
    # --- CHANGE ---: # Parameter renamed: environment now receives rule_action (0–7) ( Initial: def step(self, action) )
    def step(self, rule_action): 
        """
        # --- CRITICAL CHANGE ---: environment now receives a *rule index*, not a direct operation.
        # Use the chosen rule to select the best operation from the candidates.
        """
        current_candidates = self.omega[~self.mask]
        
        if len(current_candidates) == 0:
            obs = self._get_obs()
            return obs, 0.0, self.done(), False, {}

        action = apply_dispatching_rule(
            Rules(rule_action),
            current_candidates,
            self
        )
        
        if action not in self.partial_sol_sequeence:
            row = action // self.number_of_machines
            col = action % self.number_of_machines
            self.step_count += 1
            self.finished_mark[row, col] = 1
            dur_a = self.dur[row, col]
            self.partial_sol_sequeence.append(action)
            
            startTime_a, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
            self.flags.append(flag)

            if action not in self.last_col:
                self.omega[action // self.number_of_machines] += 1
            else:
                self.mask[action // self.number_of_machines] = 1

            self.temp1[row, col] = startTime_a + dur_a

            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)
            precd, succd = self.getNghbs(action, self.opIDsOnMchs)
            self.adj[action] = 0
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1
            if flag and precd != action and succd != action:
                self.adj[succd, precd] = 0

        # Calculate reward and create the next observation
        # --- CHANGE ---: Gymnasium format: return (obs, reward, terminated, truncated, info). In original code, return format is: (adj, fea, reward, done, omega, mask)
        obs = self._get_obs() # def _get.obs already defined above.
        reward = - (self.LBs.max() - self.max_endTime)
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.max_endTime = self.LBs.max()

        
        terminated = self.done()
        
        # Return in standard Gymnasium format
        return obs, reward, terminated, False, {}

    @override
    def reset(self, *, seed=None, options=None): # --- CHANGE ---: Gymnasium-compliant reset signature
        """
        Resets the environment to an initial state using instance data from `options`.
        """
        # It's good practice to call the parent's reset method to handle seeding
        super().reset(seed=seed)

        # --- CHANGE ---: Load instance data through 'options' dict instead of direct argument
        if options is None or 'data' not in options:
            raise ValueError("Instance data must be provided via the 'options' dictionary, e.g., env.reset(options={'data': ...})")
        data = options['data']

        # Initialize environment state
        self.step_count = 0
        self.m = data[-1]  # Machine matrix
        self.dur = data[0].astype(np.single)  # Duration matrix
        self.dur_cp = np.copy(self.dur)
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0

        # Initialize adjacency matrix with job-based precedence constraints
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        conj_nei_up_stream[self.first_col] = 0
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.adj = self_as_nei + conj_nei_up_stream

        # Initialize features and other state variables
        self.LBs = np.cumsum(self.dur, axis=1, dtype=np.single)
        self.initQuality = self.LBs.max() if not configs.init_quality_flag else 0
        self.max_endTime = self.initQuality
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)
        
        # Omega is the set of candidate operations (initially the first op of each job)
        self.omega = self.first_col.astype(np.int64)
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        # Machine-specific scheduling info
        self.mchsStartTimes = -configs.high * np.ones_like(self.dur.transpose(), dtype=np.int32)
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.dur.transpose(), dtype=np.int32)
        self.temp1 = np.zeros_like(self.dur, dtype=np.single)

        # --- CHANGE ---: The orginal code return self.adj, fea, self.omega, self.mask
        # Return the initial observation and an empty info dict
        obs = self._get_obs() 
        return obs, {}
