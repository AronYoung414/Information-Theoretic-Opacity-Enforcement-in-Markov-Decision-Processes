import numpy as np
import itertools
import torch


class ProductHMM_Matrices:
    def __init__(self, prod_hmm, num_of_states, num_sensor_queries, num_obs, k_steps=1, device='cpu'):
        """
        Helper class to extract and store Vectorized Matrices from ProductHMM.
        """
        self.device = device
        self.prod_hmm = prod_hmm
        self.num_of_states = num_of_states
        self.num_sensor_queries = num_sensor_queries
        self.num_obs = num_obs
        self.k_step = k_steps

        # 1. Compute Matrices
        # (This is your exact logic moved here to keep the solver clean)
        self.T, self.B, self.I_N, self.mu_0, self.cost_matrix = self._compute_initial_matrices()
        self.V_safe = self._precompute_safety_vector()

    def _compute_initial_matrices(self):
        """
        Compute the initial T, B, I_N, mu_0, and cost matrices for the ProductHMM.
        :return:
        """

        # 1. Transition Matrix T [N, N]
        # We assume T is fixed (Natural Dynamics).
        # We need it in the shape [To, From] for matrix multiplication: b' = T @ b

        if hasattr(self.prod_hmm, 'prod_transition_mat'):
            T_np = self.prod_hmm.prod_transition_mat.T  # Transpose to get [To, From]
        else:
            raise ValueError("Prod_HMM object is missing 'prod_transition_mat'. Check setup.")

        # 2. Emission Matrix B [Action, Obs, State]
        # B_sigma[o, s] = P(o | s, sigma)
        # We assume the setup already built this 3D array.
        B_np = np.zeros((self.num_sensor_queries, self.num_of_states, self.num_obs))
        for state, query, obs in itertools.product(self.prod_hmm.prod_states, self.prod_hmm.hmm.sensing_acts,
                                                   self.prod_hmm.hmm.observations):
            s_idx = self.prod_hmm.prod_states_indx_dict[state]
            q_idx = self.prod_hmm.hmm.sensing_act_indx_dict[query]
            o_idx = self.prod_hmm.hmm.observations_indx_dict[obs]
            prob = self.prod_hmm.prod_emission_prob[state][query][obs]

            B_np[q_idx, s_idx, o_idx] = prob

        # 3. Safety Filter I_N [N, N]
        # Diagonal Matrix: 1.0 if Safe, 0.0 if Failure
        I_N_np = np.eye(self.num_of_states)

        if hasattr(self.prod_hmm, 'failure_states_indx_dict'):
            # Direct lookup from optimized dictionary
            for idx in self.prod_hmm.failure_states_indx_dict.values():
                I_N_np[idx, idx] = 0.0

        # 4. Initial Distribution mu_0 [N]
        mu_0_np = self.prod_hmm.prod_mu_0

        # 5. Cost Matrix C [Action, Action]
        # Rows: Previous Action, Cols: Current Action
        # Construct from dictionary: cost_dict[prev_idx][curr_idx] = cost
        C_np = np.zeros((self.num_sensor_queries, self.num_sensor_queries))

        if hasattr(self.prod_hmm.hmm, 'cost_dict'):
            for prev_query, curr_query in itertools.product(self.prod_hmm.hmm.sensing_acts,
                                                            self.prod_hmm.hmm.sensing_acts):
                prev_idx = self.prod_hmm.hmm.sensing_act_indx_dict[prev_query]
                curr_idx = self.prod_hmm.hmm.sensing_act_indx_dict[curr_query]
                C_np[prev_idx, curr_idx] = self.prod_hmm.hmm.cost_dict[prev_idx][curr_idx]
        else:
            print("Warning: No cost_dict found. Assuming zero costs.")

        # Convert everything to PyTorch Tensors on the GPU
        return (torch.tensor(T_np, dtype=torch.float32, device=self.device),
                torch.tensor(B_np, dtype=torch.float32, device=self.device),
                torch.tensor(I_N_np, dtype=torch.float32, device=self.device),
                torch.tensor(mu_0_np, dtype=torch.float32, device=self.device),
                torch.tensor(C_np, dtype=torch.float32, device=self.device))

    def _precompute_safety_vector(self):
        """
        Computes the vector V_safe where V_safe[s] is the probability
        of staying safe for k steps starting from state s.
        Math: 1^T (I_N T)^k I_N
        """
        # Start with a vector of ones (1^T) representing "Safe"
        # We work backwards: P(Safe at t+k | Safe at t)

        # 1. Initialize v = Diagonal of I_N (1 if safe, 0 if unsafe)
        v = torch.diagonal(self.I_N).clone()

        # 2. Iterate backwards k times
        # v_prev = I_N * T * v_next
        # In code (transposed): v = (T @ v) * mask
        safe_mask = torch.diagonal(self.I_N)

        for _ in range(self.k_step):
            # T is [To, From]. We want to check if we transition TO a safe state.
            # So we sum over rows (transitions to safe states).
            # v = T^T @ v
            v = torch.matmul(self.T.t(), v)

            # Mask unsafe states immediately
            v = v * safe_mask

        return v
