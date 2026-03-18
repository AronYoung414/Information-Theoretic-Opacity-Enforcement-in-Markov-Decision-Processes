import torch
import torch.nn.functional as F
import numpy as np
import time
import pickle
import os
import itertools
import matplotlib.pyplot as plt
from setup_and_solvers.setup_file_for_active_perception_for_predictive_safety import *
from setup_and_solvers.policy import *
from setup_and_solvers.environment_matrices import *

from loguru import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class ObservationPolicyLSTM(torch.nn.Module):
#     def __init__(self, num_obs, num_actions, embedding_dim=32, hidden_dim=64):
#         super(ObservationPolicyLSTM, self).__init__()
#
#         self.hidden_dim = hidden_dim
#
#         # 1. THE TRANSLATOR (Embedding)
#         # Neural networks hate integers (0, 1, 2). They love vectors.
#         # This layer turns obs ID '3' into a vector like [0.1, -0.5, 0.0, ...]
#         self.embedding = torch.nn.Embedding(num_obs, embedding_dim)
#
#         # 2. THE BRAIN (LSTM Cell)
#         # This is the recurrent core.
#         # It takes an input vector (size 32) and the old memory (size 64).
#         # It outputs the new memory (size 64).
#         self.lstm = torch.nn.LSTMCell(embedding_dim, hidden_dim)
#
#         # 3. THE DECIDER (Linear/Fully Connected)
#         # This takes the memory (size 64) and decides the action scores.
#         # Output size = num_actions
#         self.fc = torch.nn.Linear(hidden_dim, num_actions)
#
#     def forward(self, obs_idx, hidden_state):
#         # Step 1: Translate Integer -> Vector
#         # obs_idx shape: [Batch_Size] -> embedded shape: [Batch, 32]
#         embedded = self.embedding(obs_idx)
#
#         # Step 2: Update Memory
#         # Input: (New Vector, Old Memory Tuple)
#         # Output: (New Hidden h_x, New Cell State c_x)
#         h_x, c_x = self.lstm(embedded, hidden_state)
#
#         # Step 3: Decide Action
#         # Input: New Hidden State h_x
#         # Output: Raw scores for actions (Logits)
#         logits = self.fc(h_x)
#
#         # Return the scores AND the new memory (to be used in the next loop iteration)
#         return logits, (h_x, c_x)
#
#     def init_hidden(self, batch_size):
#         return (torch.zeros(batch_size, self.hidden_dim, device=device),
#                 torch.zeros(batch_size, self.hidden_dim, device=device))


class GradientDescent_LSTM_solver:
    def __init__(self, prod_hmm, k_step=1, lr=0.005, alpha_cost=0):
        """
        The gradient based solver with the observation based LSTM policy.
        Args:
             prod_hmm: The ProductHMM object defining the environment.
             k_step: The forward looking window for predictive safety.
             lr: Learning rate.
             alpha_cost: Weight for the sensor cost in the loss function.
        """
        self.prod_hmm = prod_hmm
        self.k_step = k_step
        self.lr = lr
        self.alpha_cost = alpha_cost

        # Dimensions of inputs
        self.num_of_states = len(self.prod_hmm.prod_states)
        self.num_sensor_queries = len(self.prod_hmm.hmm.sensing_acts)
        self.num_obs = len(self.prod_hmm.hmm.observations)

        # Compute the initial Matrices for the Observable Operators
        # self.T, self.B, self.I_N, self.mu_0, self.cost_matrix = self._compute_initial_matrices()
        self.matrices = ProductHMM_Matrices(self.prod_hmm, self.num_of_states, self.num_sensor_queries, self.num_obs,
                                            self.k_step, device)
        self.T = self.matrices.T
        self.B = self.matrices.B
        self.I_N = self.matrices.I_N
        self.mu_0 = self.matrices.mu_0
        self.cost_matrix = self.matrices.cost_matrix
        self.V_safe = self.matrices.V_safe

        # print(self.T)
        # print(self.B)
        # print(self.I_N)

        # Initialize the LSTM policy network
        self.policy_net = ObservationPolicyLSTM(self.num_obs, self.num_sensor_queries).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Logging
        self.history = {'entropy': [], 'cost': []}

    # def _compute_initial_matrices(self):
    #     """
    #     Compute the initial T, B, I_N, mu_0, and cost matrices for the ProductHMM.
    #     :return:
    #     """
    #
    #     # 1. Transition Matrix T [N, N]
    #     # We assume T is fixed (Natural Dynamics).
    #     # We need it in the shape [To, From] for matrix multiplication: b' = T @ b
    #
    #     if hasattr(self.prod_hmm, 'prod_transition_mat'):
    #         T_np = self.prod_hmm.prod_transition_mat.T  # Transpose to get [To, From]
    #     else:
    #         raise ValueError("Prod_HMM object is missing 'prod_transition_mat'. Check setup.")
    #
    #     # 2. Emission Matrix B [Action, Obs, State]
    #     # B_sigma[o, s] = P(o | s, sigma)
    #     # We assume the setup already built this 3D array.
    #     B_np = np.zeros((self.num_sensor_queries, self.num_of_states, self.num_obs))
    #     for state, query, obs in itertools.product(self.prod_hmm.prod_states, self.prod_hmm.hmm.sensing_acts,
    #                                                self.prod_hmm.hmm.observations):
    #         s_idx = self.prod_hmm.prod_states_indx_dict[state]
    #         q_idx = self.prod_hmm.hmm.sensing_act_indx_dict[query]
    #         o_idx = self.prod_hmm.hmm.observations_indx_dict[obs]
    #         prob = self.prod_hmm.prod_emission_prob[state][query][obs]
    #
    #         B_np[q_idx, s_idx, o_idx] = prob
    #
    #     # 3. Safety Filter I_N [N, N]
    #     # Diagonal Matrix: 1.0 if Safe, 0.0 if Failure
    #     I_N_np = np.eye(self.num_of_states)
    #
    #     if hasattr(self.prod_hmm, 'failure_states_indx_dict'):
    #         # Direct lookup from optimized dictionary
    #         for idx in self.prod_hmm.failure_states_indx_dict.values():
    #             I_N_np[idx, idx] = 0.0
    #
    #     # 4. Initial Distribution mu_0 [N]
    #     mu_0_np = self.prod_hmm.prod_mu_0
    #
    #     # 5. Cost Matrix C [Action, Action]
    #     # Rows: Previous Action, Cols: Current Action
    #     # Construct from dictionary: cost_dict[prev_idx][curr_idx] = cost
    #     C_np = np.zeros((self.num_sensor_queries, self.num_sensor_queries))
    #
    #     if hasattr(self.prod_hmm.hmm, 'cost_dict'):
    #         for prev_query, curr_query in itertools.product(self.prod_hmm.hmm.sensing_acts,
    #                                                         self.prod_hmm.hmm.sensing_acts):
    #             prev_idx = self.prod_hmm.hmm.sensing_act_indx_dict[prev_query]
    #             curr_idx = self.prod_hmm.hmm.sensing_act_indx_dict[curr_query]
    #             C_np[prev_idx, curr_idx] = self.prod_hmm.hmm.cost_dict[prev_idx][curr_idx]
    #     else:
    #         print("Warning: No cost_dict found. Assuming zero costs.")
    #
    #     # Convert everything to PyTorch Tensors on the GPU
    #     return (torch.tensor(T_np, dtype=torch.float32, device=device),
    #             torch.tensor(B_np, dtype=torch.float32, device=device),
    #             torch.tensor(I_N_np, dtype=torch.float32, device=device),
    #             torch.tensor(mu_0_np, dtype=torch.float32, device=device),
    #             torch.tensor(C_np, dtype=torch.float32, device=device))

    def sample_data_batch(self, batch_size, horizon):
        """
        Generates a batch of trajectories using the internal policy and environment matrices.
        """
        # 1. Initialize State (s_0)
        # Use self.mu_0 instead of solver.mu_0
        curr_state_idx = torch.multinomial(self.mu_0.unsqueeze(0).expand(batch_size, -1), 1).squeeze()

        # 2. Initialize LSTM
        # Use self.policy_net instead of policy_net
        hidden_state = self.policy_net.init_hidden(batch_size)

        # 3. Initialize First Input (Start Token)
        obs_input = torch.zeros(batch_size, dtype=torch.long, device=device)

        obs_history = []
        act_history = []
        log_prob_history = []

        for t in range(horizon):
            # --- A. DECISION PHASE (Policy) ---
            # Use self.policy_net
            logits, hidden_state = self.policy_net(obs_input, hidden_state)
            probs = F.softmax(logits, dim=1)

            # Create distribution from these probs
            dist = torch.distributions.Categorical(probs)

            # Sample Action sigma_t
            sigma_t = dist.sample()
            log_prob = dist.log_prob(sigma_t)

            # Store Action & LogProb
            act_history.append(sigma_t)
            log_prob_history.append(log_prob)

            # --- B. OBSERVATION PHASE (Environment) ---
            # Use self.B instead of solver.B
            # B is [Action, State, Obs]. Select B[sigma_t, curr_state_idx, :]
            probs_obs = self.B[sigma_t, curr_state_idx, :]

            # Sample Observation o_t
            # print(sigma_t)
            o_t = torch.multinomial(probs_obs, 1).squeeze()

            # Store Observation
            obs_history.append(o_t)

            # Update input for the NEXT loop iteration
            obs_input = o_t

            # --- C. TRANSITION PHASE (Physics) ---
            # Use self.T instead of solver.T
            # T is [To, From] -> P(s' | s)
            trans_probs = self.T[:, curr_state_idx].t()
            curr_state_idx = torch.multinomial(trans_probs, 1).squeeze()

        # Return stacked history [Horizon, Batch]
        return (torch.stack(obs_history),
                torch.stack(act_history),
                torch.stack(log_prob_history))

    def compute_loss_paper_version(self, obs_data, act_data, log_probs, alpha=0):
        """
        Strict implementation of the Math Draft.

        Args:
            obs_data: [Horizon, Batch] - Sequence of observations o_t
            act_data: [Horizon, Batch] - Sequence of actions sigma_t
            log_probs: [Horizon, Batch] - log pi(sigma_t | ...)
            alpha: cost weighting factor

        Returns:
            loss: Scalar tensor for backpropagation (Eq 11)
            mean_entropy: Float for plotting
            mean_cost: float for plotting
        """
        horizon, batch_size = obs_data.shape

        # --- 1. Initialize (mu_0) ---
        # This represents the term (A ... mu_0) at t=0 (which is just mu_0)
        alpha_vec = self.mu_0.unsqueeze(0).expand(batch_size, -1).clone()

        # We accumulate the loss terms and entropy values
        loss_terms = []
        all_entropies = []

        # Calculate Cumulative Log Probability (log P_theta(y))
        # cum_log_probs[t] corresponds to sum_{i=0}^t log pi(sigma_i)
        cum_log_probs = torch.cumsum(log_probs, dim=0)

        # # Pre-compute diagonal mask for I_N
        # safe_mask = torch.diagonal(self.I_N)  # [N]

        # Expand V_safe for the batch [1, N] -> [Batch, N]
        # We do this once to avoid repeating it inside the loop
        V_safe_batch = self.V_safe.unsqueeze(0).expand(batch_size, -1)

        for t in range(horizon):
            o_t = obs_data[t]
            sigma_t = act_data[t]

            # --- STEP 1: Compute the term diag(B^sigma_t_o_t) (A ... mu_0) ---
            # self.B is [Action, State, Obs]. Permute to [Action, Obs, State]
            B_perm = self.B.permute(0, 2, 1)

            # Get the vector B^sigma_t_o_t
            B_vec = B_perm[sigma_t, o_t, :]  # Shape: [Batch, State]

            # This 'joint_belief' is the term: diag(B^sigma_t_o_t) (A ... mu_0)
            joint_belief = alpha_vec * B_vec  # Element-wise multiplication

            # --- STEP 2: Compute Denominator (Proposition 1 / Prop 5 Denom) ---
            # Denom = 1^T joint_belief
            denom = joint_belief.sum(dim=1)
            # (Add epsilon only for numerical stability of division, not changing logic)
            denom_stable = denom + 1e-10

            # --- STEP 3: Compute Numerator (Proposition 5 Num) ---
            # Num = 1^T (I_N T)^k I_N joint_belief

            # # A. Apply first I_N
            # curr_vec = joint_belief * safe_mask
            #
            # # B. Apply (I_N T)^k loop
            # for _ in range(self.k_step):
            #     # Apply T (Propagate dynamics)
            #     # Mathematical Note: Vector v * Matrix M in math is v^T M.
            #     # In PyTorch (batch, N) @ (N, N)^T = (batch, N).
            #     # So we use T.t() to match right-multiplication convention.
            #     curr_vec = torch.matmul(curr_vec, self.T.t())
            #
            #     # Apply I_N (Filter unsafe states)
            #     curr_vec = curr_vec * safe_mask
            #
            # # C. Sum (1^T ...)
            # num = curr_vec.sum(dim=1)

            # Using the precomputed safety mask.
            num = (joint_belief * V_safe_batch).sum(dim=1)

            # --- STEP 4: Compute Safety Probability & Entropy ---
            # Eq 12: P(W=0|y) = Num / Denom
            p_safe = num / denom_stable

            # Clamp for log stability
            p_safe = torch.clamp(p_safe, 1e-6, 1.0 - 1e-6)
            p_threat = 1.0 - p_safe

            # print(f"P_safe at time {t}: {p_safe}")
            # print(f"P_threat at time {t}: {p_threat}")

            # Entropy H(W_t^k | y)
            entropy_t = -(p_safe * torch.log2(p_safe) + p_threat * torch.log2(p_threat))
            all_entropies.append(entropy_t)

            # --- STEP 5: Compute Gradient Term (Equation 11) ---
            # We want to minimize H. The gradient approximation is:
            # nabla J = (1/V) sum ( H(y) * nabla log P(y) )
            # In PyTorch, we define Loss = H.detach() * log P(y)

            # cum_log_probs[t] is log P_theta(y) up to time t
            step_loss = entropy_t.detach() * cum_log_probs[t]
            loss_terms.append(step_loss)

            # --- STEP 6: Update Alpha for Next Step (The Observable Operator A) ---
            # We need alpha_{t+1} = A^theta_{o_t|sigma_t} alpha_t
            # A = T diag(B)
            # We already computed 'joint_belief' which is diag(B) alpha_t
            # So we just need to apply T.

            alpha_next_unnorm = torch.matmul(joint_belief, self.T.t())

            # Normalize to keep numerical stability (Prop 3 implies normalization by P(o|...))
            # We divide by the denominator P(o_t | ...) calculated earlier.
            alpha_vec = alpha_next_unnorm / denom_stable.unsqueeze(1)

        # --- FINAL AGGREGATION ---

        # Compute the cost term
        # Shift act_data down by 1 so that prev_act_data[t] = act_data[t-1]
        # act_data is [Horizon, Batch]
        prev_act_data = torch.roll(act_data, shifts=1, dims=0)

        start_action_index = self.prod_hmm.hmm.sensing_act_indx_dict[self.prod_hmm.hmm.no_sensing_act]
        prev_act_data[0, :] = start_action_index

        # Vectorized Lookup
        # We use the [prev, curr] indices to pull values from self.cost_matrix
        # self.cost_matrix must be a Tensor [Num_Actions, Num_Actions] on the same device.

        # This creates cost_batch of shape [Horizon, Batch]
        # It looks up C[ prev_act_data[t,b], act_data[t,b] ] for every element.
        cost_batch = self.cost_matrix[prev_act_data, act_data]

        trajectory_costs = cost_batch.sum(dim=0)  # Sum over time to get total cost per trajectory
        # Get Total Log Prob (Optimization: Re-use cum_log_probs)
        # The last element of cumsum is the total sum.
        trajectory_log_probs = cum_log_probs[-1]  # [Batch]

        cost_loss = (trajectory_log_probs * trajectory_costs.detach()).mean()

        # 1. Total Loss
        # Equation 11 implies summation over samples (mean over batch)
        # and Eq 10 implies summation over time T.
        loss_tensor = torch.stack(loss_terms)  # [Horizon, Batch]
        # total_loss = loss_tensor.sum(dim=0).mean()  # Sum over T, Mean over Batch

        entropy_loss = loss_tensor.sum(dim=0).mean()  # Sum over T, Mean over Batch

        # 2. Average Entropy (For Plotting)
        entropy_tensor = torch.stack(all_entropies)
        mean_entropy = entropy_tensor.mean().item()

        total_loss = entropy_loss + (alpha * cost_loss)

        mean_cost = trajectory_costs.mean().item()

        # # trying with some additinal exploration.
        # policy_entropy_proxy = -torch.mean(log_probs)
        # beta = 0.2
        # augmented_loss = total_loss - (beta * policy_entropy_proxy)

        return total_loss, mean_entropy, mean_cost
        # return augmented_loss, mean_entropy, mean_cost

    # def compute_loss(self, obs_data, act_data, log_probs):
    #     """
    #     Computes the IROS Objective (Entropy + Cost) and the REINFORCE Gradient.
    #
    #     Args:
    #         obs_data: [Horizon, Batch] Tensor of observation indices.
    #         act_data: [Horizon, Batch] Tensor of action indices.
    #         log_probs: [Horizon, Batch] Tensor of action log probabilities.
    #     """
    #     horizon, batch_size = obs_data.shape
    #
    #     # 1. Initialize Forward Vector (alpha_0)
    #     # Corresponds to P(Z_0, y_0) before any observations
    #     alpha_vector = self.mu_0.unsqueeze(0).expand(batch_size, -1).clone()
    #
    #     # 2. Prepare Previous Actions (for Cost calculation)
    #     # Shift actions right: [a0, a1...] -> [0, a0, a1...]
    #     prev_actions = torch.cat([torch.zeros(1, batch_size, dtype=torch.long, device=device), act_data[:-1]])
    #
    #     entropies = []
    #     costs = []
    #
    #     for t in range(horizon):
    #         o_t = obs_data[t]
    #         sigma_t = act_data[t]  # Action chosen at t
    #         sigma_prev = prev_actions[t]  # Action active during transition to t
    #
    #         # --- A. ENTROPY CALCULATION (Eq 4 & 8) ---
    #
    #         # 1. Update Alpha (Bayes Numerator Step 1)
    #         # We need the probability vector: P(o_t | s, sigma_prev)
    #         # self.B is [Action, State, Obs].
    #         # We permute to [Action, Obs, State] to slice the 'State' vector easily.
    #         B_perm = self.B.permute(0, 2, 1)
    #         B_vec = B_perm[sigma_prev, o_t, :]  # Result: [Batch, State]
    #
    #         # Element-wise multiply: diag(B) * alpha
    #         alpha_cond = alpha_vector * B_vec
    #
    #         # 2. Propagate Dynamics (Denominator P(y))
    #         # Multiply by T^T (equivalent to alpha * T)
    #         alpha_next_unnorm = torch.matmul(alpha_cond, self.T.t())
    #
    #         # Sum for denominator (P(y_{0:t}))
    #         denominator = alpha_next_unnorm.sum(dim=1) + 1e-10
    #
    #         # 3. Numerator (Safety Constrained Propagation)
    #         # We want: 1^T (I_N T)^k I_N alpha_cond
    #         safe_mask = torch.diagonal(self.I_N)
    #
    #         # Apply Safety Filter at current step
    #         pred_vec = alpha_cond * safe_mask
    #
    #         # Look ahead k steps
    #         for _ in range(self.k_step):
    #             pred_vec = torch.matmul(pred_vec, self.T.t())
    #             pred_vec = pred_vec * safe_mask
    #
    #         numerator = pred_vec.sum(dim=1)
    #
    #         # 4. Compute Entropy
    #         p_safe = numerator / denominator
    #         p_safe = torch.clamp(p_safe, 1e-6, 1.0 - 1e-6)
    #         p_threat = 1.0 - p_safe
    #
    #         # --- DEBUG BLOCK INSIDE COMPUTE_LOSS ---
    #         # expected shape of prob_safe: [Batch_Size, 1] or [Batch_Size]
    #
    #         print("\n--- DEBUG: Safety Probabilities ---")
    #         print(f"Min Safety: {p_safe.min().item():.4f}")
    #         print(f"Max Safety: {p_safe.max().item():.4f}")
    #         print(f"Mean Safety: {p_safe.mean().item():.4f}")
    #
    #         # Check if we have ANY intermediate values (0.2) or only extremes (0.0 / 1.0)
    #         intermediate_mask = (p_safe > 0.01) & (p_safe < 0.99)
    #         print(f"Number of 'Uncertain' steps: {intermediate_mask.sum().item()}")
    #         print("-----------------------------------")
    #
    #         entropy = -(p_safe * torch.log2(p_safe) + p_threat * torch.log2(p_threat))
    #         entropies.append(entropy)
    #
    #         # Update alpha_vector for the next loop iteration (normalize)
    #         alpha_vector = alpha_next_unnorm / denominator.unsqueeze(1)
    #
    #         # --- B. COST CALCULATION ---
    #         # Cost of switching from sigma_prev to sigma_t
    #         cost = self.cost_matrix[sigma_prev, sigma_t]
    #         costs.append(cost)
    #
    #     # Stack lists into tensors [Horizon, Batch]
    #     entropies = torch.stack(entropies)
    #     costs = torch.stack(costs)
    #
    #     # --- C. REINFORCE GRADIENT CALCULATION ---
    #
    #     # 1. Define Reward Function (Negative Loss)
    #     # We want to minimize Entropy and Cost -> Maximize negative.
    #     total_loss_per_step = entropies + self.alpha_cost * costs
    #
    #     # 2. Compute Returns (Cumulative Sum from t to T)
    #     returns = torch.zeros_like(total_loss_per_step)
    #     R = torch.zeros(batch_size, device=device)
    #
    #     for t in reversed(range(horizon)):
    #         R = total_loss_per_step[t] + R
    #         returns[t] = R
    #
    #     # 3. Normalize Returns (Baseline for stability)
    #     returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    #
    #     # 4. Policy Gradient Loss
    #     # Loss = E [ log_pi * Return ]
    #     # Since we want to Minimize loss, we use positive sign here if 'Returns' are Costs.
    #     # If Returns represent "Badness", we want to minimize log_prob * Badness.
    #     pg_loss = (log_probs * returns).mean()
    #
    #     return pg_loss, entropies.mean().item(), costs.mean().item()

    def train_step(self, V=100, batch_size=100, horizon=20, alpha=0):
        """
                Orchestrates one complete optimization step:
                1. Generate Data (Simulation)
                2. Compute Loss (Math)
                3. Backpropagate (Optimization)
                """
        # 1. Zero the gradients before starting
        self.optimizer.zero_grad()

        # Calculate how many chunks we need.
        num_chunks = int(np.ceil(V / batch_size))

        total_entropy = 0
        total_cost = 0

        for _ in range(num_chunks):
            # A. Generate Data (Simulate trajectories)
            # This uses self.policy_net inside to make decisions
            obs_data, act_data, log_probs = self.sample_data_batch(batch_size, horizon)

            # B. Compute Gradients (Calculate Entropy, Cost, and REINFORCE Loss)
            # This uses the data we just generated
            # loss, avg_ent, avg_cost = self.compute_loss(obs_data, act_data, log_probs)

            loss, avg_ent, avg_cost = self.compute_loss_paper_version(obs_data, act_data, log_probs, alpha)

            # Since we sum losses over chunks, we need to scale them
            loss = loss / num_chunks

            # D. Backpropagation
            loss.backward()

            # Accumulate metrics
            total_entropy += avg_ent / num_chunks
            total_cost += avg_cost / num_chunks

        # 2. Gradient Clipping (Crucial for LSTM stability)
        # Prevents "exploding gradients" which can destroy training
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        # 3. Update Weights
        self.optimizer.step()

        # Return metrics for logging
        return total_entropy, total_cost

    def train(self, iterations=1000, V=100, batch_size=100, horizon=20, alpha=0, sensor_cost_normalization=1.0):
        """
        Runs the training loop for a specified number of iterations.
        """
        print(f"Starting Training for {iterations} iterations...")
        print(f"V: {V}, Batch Size: {batch_size}, Horizon: {horizon}, k-step: {self.k_step}")

        for i in range(iterations):

            start_time = time.time()

            # Run one optimization step
            avg_ent, avg_cost = self.train_step(V, batch_size, horizon, alpha)

            # Record history
            self.history['entropy'].append(avg_ent)
            self.history['cost'].append(avg_cost)

            # Log progress every 50 iterations
            if i % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Iter {i:04d} | H(W|y): {avg_ent:.4f} | Cost: {avg_cost:.4f} | Time: {elapsed:.4f}s")

        print("Training Complete.")
        self._plot_results(sensor_cost_normalization)

    def _plot_results(self, sensor_cost_normalization=1.0):
        """
        Helper to visualize training progress.
        """
        plt.figure(figsize=(12, 5))

        # Plot 1: Safety Uncertainty (Entropy)
        plt.subplot(1, 2, 1)
        plt.plot(self.history['entropy'], label='H(W|y)', color='blue')
        plt.title('Predictive Safety Uncertainty')
        plt.xlabel('Iteration')
        plt.ylabel('Entropy (Bits)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot 2: Sensing Cost
        plt.subplot(1, 2, 2)
        # Get the LIST from the dictionary
        cost_list = self.history['cost']

        # Convert the LIST to a NUMPY ARRAY so math works
        # cost_array = np.array(cost_list)
        #
        # # Now multiply (Element-wise multiplication)
        # real_costs = cost_array * sensor_cost_normalization
        #
        # # Plot the calculated array
        # plt.plot(real_costs, label='Cost', color='orange')
        plt.plot(self.history['cost'], label='Cost', color='orange')
        plt.title('Perception Cost')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_results_for_campus_example_9_cameras_6.png")
        print("Results saved to 'training_results_for_campus_example_9_cameras_6.png'")

    def evaluate_policy(self, horizon=20, forced_trace=None):
        print("\n" + "=" * 100)
        print(" POLICY EVALUATION (Full Distribution)")
        print("=" * 100)

        # Header: We make the last column very wide to fit all actions
        print(f"{'Step':<5} | {'True State':<20} | {'Prev Obs':<12} | {'Action Distribution (Confidence)'}")
        print("-" * 100)

        # Init
        if forced_trace is not None:
            curr_s_idx = forced_trace[0]
        else:
            curr_s_idx = torch.multinomial(self.mu_0, 1).item()

        hidden = self.policy_net.init_hidden(1)
        obs_input = torch.zeros(1, dtype=torch.long, device=device)  # Start token

        for t in range(horizon):
            # A. Policy Forward Pass
            logits, hidden = self.policy_net(obs_input, hidden)
            probs = F.softmax(logits, dim=1)  # Shape: [1, num_actions]

            # --- BUILD DEBUG STRING ---
            # Loop over all possible actions to show their probabilities
            dist_str_parts = []
            for i, act_name in enumerate(self.prod_hmm.hmm.sensing_acts):
                p = probs[0, i].item()
                # Format: "Name: 0.95"
                # We mark the CHOSEN action (max prob) with a star '*' for clarity
                is_max = (p == probs.max().item())
                marker = "*" if is_max else ""
                dist_str_parts.append(f"{str(act_name)}{marker}: {p:.6f}")

            dist_str = " | ".join(dist_str_parts)

            # B. Environment Step (Sample actual transition)
            # We still pick the greedy action to drive the simulation for the test
            conf, a_idx_tensor = torch.max(probs, dim=1)
            a_idx = a_idx_tensor.item()

            o_prev_idx = obs_input.item()
            probs_obs = self.B[a_idx, curr_s_idx, :]
            o_t = torch.multinomial(probs_obs, 1)

            # --- PRINTING ---
            s_val = self.prod_hmm.prod_states[curr_s_idx]
            o_val = "START" if t == 0 else self.prod_hmm.hmm.observations_indx_to_obs_dict[
                o_prev_idx]

            print(f"{t:<5} | {str(s_val):<20} | {str(o_val):<12} | {dist_str}")

            # C. Update for next step
            obs_input = o_t
            if forced_trace is not None:
                if t < len(forced_trace) - 1:
                    curr_s_idx = forced_trace[t + 1]
            else:
                # obs_input = o_t
                trans_probs = self.T[:, curr_s_idx].t()
                curr_s_idx = torch.multinomial(trans_probs, 1).item()

        print("-" * 100 + "\n")

    def evaluate_policy_accuracy_monte_carlo(self, episodes=100, horizon=20, k_step=2, threshold=0.75,
                                             forced_trace=None, uniform_random=False):
        print("\n" + "=" * 100)
        print(f" SAFETY EVALUATION (Horizon={horizon}, k={k_step}, Thresh={threshold})")
        print("=" * 100)
        print(
            f"{'Step':<5} | {'State':<10} | {'Action':<10} | {'Obs':<10} | {'P(Safe)':<10} | {'Flag':<5} | {'Result'}")
        print("-" * 100)

        # --- 1. PRE-COMPUTATION (V_safe) ---
        # We pre-compute the "Value of Safety" vector V_safe.
        # V_safe[s] = Probability of staying safe for k steps starting from s.
        # Recursion: V_k(s) = I_N(s) * Sum(T(s, s') * V_{k-1}(s'))
        # Matrix Form: V = I_N * T * V

        # I_N = self.I_N  # [N, N]
        # T = self.T  # [N, N]
        #
        # # Start with 1.0 (All states technically safe at step 0 of lookahead)
        # # Using column vector [N, 1] for matrix multiplication
        # V_safe = torch.ones((self.T.shape[0], 1), device=device)
        #
        # # Apply recurrence k times: V = I_N @ (T @ V)
        # for _ in range(k_step):
        #     V_safe = torch.matmul(T, V_safe)  # Propagate expected value backwards
        #     V_safe = torch.matmul(I_N, V_safe)  # Mask unsafe states
        #
        # # Reshape to [1, N] for easy broadcasting with belief later
        # V_safe_row = self.V_safe.t()
        # V_safe_row = self.V_safe.unsqueeze(0)

        # --- 1. LOCAL PRE-COMPUTATION OF V_SAFE ---
        # We compute this LOCALLY to ensure it matches the 'k_step' passed to this function.
        # Do not rely on self.V_safe which might be stale or based on a different k.

        I_N = self.I_N  # [N, N]
        T = self.T  # [N, N] [Next, Curr]

        # Start with 1.0 (Safe) - Shape [N]
        V_safe_vec = torch.ones(T.shape[0], device=device)

        # Safety Mask (1=Safe, 0=Trap) - Shape [N]
        safe_mask = torch.diagonal(I_N)

        # Apply recurrence k times: V(s) = Sum( P(s'|s) * V(s') ) * IsSafe(s)
        # We use T.t() because T is [Next, Prev]. T.t() sums over outgoing edges.

        # 0th step: You must be safe NOW to be safe for 0 steps
        V_safe_vec = V_safe_vec * safe_mask

        for _ in range(k_step):
            # Propagate value backwards from future
            V_safe_vec = torch.matmul(T.t(), V_safe_vec)
            # Mask unsafe states again
            V_safe_vec = V_safe_vec * safe_mask

        # Reshape to [1, N] for broadcasting with joint_belief [Batch, N]
        V_safe_row = V_safe_vec.unsqueeze(0)

        # Trap Mask for Ground Truth Checking (1=Safe, 0=Trap)
        safety_mask = torch.diagonal(I_N).cpu().numpy()

        # Global counters for accuracy
        total_decisive = 0
        total_correct = 0
        total_wrong = 0
        total = 0

        # Episode loop

        for ep in range(episodes):

            # --- 2. GENERATE REALITY ---
            # We generate the full trace upfront so we can slice [t, t+k] easily
            true_states = []
            if forced_trace is not None:
                true_states = forced_trace
                # Ensure forced trace covers the full lookahead window
                if len(true_states) < horizon + k_step:
                    horizon = len(true_states) - k_step - 1
            else:
                curr = torch.multinomial(self.mu_0, 1).item()
                true_states = [curr]
                for _ in range(horizon):
                    probs = self.T[:, true_states[-1]]
                    next_s = torch.multinomial(probs, 1).item()
                    true_states.append(next_s)

            # --- 3. INITIALIZE AGENT ---
            hidden = self.policy_net.init_hidden(1)
            obs_input = torch.zeros(1, dtype=torch.long, device=device)  # Start Token
            # Alpha (Belief): Start with mu_0. Shape [1, N]
            alpha = self.mu_0.clone().unsqueeze(0)

            # Print header only for first episode
            if ep == 0:
                print(
                    f"{'Step':<5} | {'State':<10} | {'Action':<10} | {'Obs':<10} | {'P(Safe)':<10} | {'Flag':<5} | {'Result'}")
                print("-" * 100)

            # --- 4. EVALUATION LOOP ---
            for t in range(horizon):
                curr_s_idx = true_states[t]

                # A. SELECT ACTION (Policy)
                if uniform_random == False:

                    logits, hidden = self.policy_net(obs_input, hidden)
                    probs_action = F.softmax(logits, dim=1)
                    a_idx = torch.multinomial(probs_action, 1).item()
                    act_name = self.prod_hmm.hmm.sensing_acts[a_idx]

                else:
                    a_idx = torch.randint(low=0, high=self.num_sensor_queries, size=(1,)).item()
                    act_name = self.prod_hmm.hmm.sensing_acts[a_idx]

                # B. ENVIRONMENT (Observation)
                probs_obs = self.B[a_idx, curr_s_idx, :]
                o_idx = torch.multinomial(probs_obs, 1).item()

                # # C. COMPUTE P_SAFE (Your Logic)
                # # 1. Propagate Belief: alpha_pred = alpha * T
                # # (Note: Assuming alpha is row [1,N] and T is [N,N], we do alpha @ T)
                # alpha_pred = torch.matmul(alpha, self.T.t())
                #
                # # 2. Get Observation Vector B_vec
                # # self.B is [Action, State, Obs]. Permute to [Action, Obs, State]
                # # (Or just access directly: B[a, :, o])
                # B_perm = self.B.permute(0, 2, 1)
                #
                # # Get the vector B^sigma_t_o_t
                # B_vec = B_perm[a_idx, o_idx, :]  # Shape: [Batch, State]
                #
                # # B_vec = self.B[a_idx, :, o_idx].unsqueeze(0)  # Shape [1, N]
                #
                # # 3. Compute Joint Belief (Unnormalized Updated Belief)
                # # "joint_belief = alpha_vec * B_vec"
                # joint_belief = alpha_pred * B_vec
                #
                # # 4. Compute Denominator (P(y|...))
                # denom = joint_belief.sum(dim=1)
                # denom_stable = denom + 1e-10
                #
                # # 5. Compute Numerator (using Pre-computed V_safe)
                # # "num = (joint_belief * V_safe_batch).sum(dim=1)"
                # num = (joint_belief * V_safe_row).sum(dim=1)
                #
                # # 6. P_safe
                # p_safe = num / denom_stable
                # p_safe_val = torch.clamp(p_safe, 1e-6, 1.0 - 1e-6).item()
                #
                # # Update Alpha for next step (Normalized)
                # alpha = joint_belief / denom_stable

                # C. COMPUTE P_SAFE (EXACT TRAINING LOGIC)

                # 1. Get B vector
                # B[a, :, o] gives [State]. Unsqueeze to [1, N]
                B_vec = self.B[a_idx, :, o_idx].unsqueeze(0)

                # 2. Joint Belief (Unnormalized)
                # joint = Current_Belief * Likelihood (NO TRANSITION YET)
                joint_belief = alpha * B_vec

                # 3. Denominator
                denom = joint_belief.sum(dim=1)
                denom_stable = denom + 1e-10

                # 4. Numerator (Using V_safe)
                num = (joint_belief * V_safe_row).sum(dim=1)

                # 5. P_safe
                p_safe = num / denom_stable
                p_safe_val = torch.clamp(p_safe, 1e-6, 1.0 - 1e-6).item()

                # --- UPDATE ALPHA (Transition Happens Here) ---
                # "alpha_next_unnorm = torch.matmul(joint_belief, self.T.t())"
                alpha_next_unnorm = torch.matmul(joint_belief, self.T.t())

                # Normalize
                alpha = alpha_next_unnorm / denom_stable.unsqueeze(1)  # Broadcast div

                # D. GROUND TRUTH CHECK
                # Window: [t, t+k]
                # We assume t is current state, so we check t, t+1 ... t+k
                actual_safe = True
                for i in range(k_step + 1):
                    check_idx = t + i
                    if check_idx < len(true_states):
                        if safety_mask[true_states[check_idx]] == 0.0:
                            actual_safe = False
                            break

                # E. DECISIVENESS & RESULT FLAG
                # 0 = Fail Pred, 1 = Safe Pred
                rand_val = torch.rand(1, device=device).item()
                pred_flag = 1 if rand_val < p_safe_val else 0

                # Check correctness
                is_correct = False
                if pred_flag == 1 and actual_safe:
                    is_correct = True
                elif pred_flag == 0 and not actual_safe:
                    is_correct = True
                #
                # if p_safe_val >= threshold:
                #     pred_flag = 1  # SAFE
                #     if actual_safe: is_correct = True
                #     total_decisive += 1
                # elif (1 - p_safe_val) >= threshold:
                #     pred_flag = 0  # FAIL
                #     if not actual_safe: is_correct = True
                #     total_decisive += 1

                # if pred_flag != 2:
                #     total += 1
                #     if is_correct:
                #         total_correct += 1
                #     else:
                #         total_wrong += 1
                # else:
                #     total += 1

                if is_correct:
                    total += 1
                    total_correct += 1
                else:
                    total += 1
                    total_wrong += 1

                result_marker = "CORRECT" if (pred_flag != 2 and is_correct) else ("WRONG" if pred_flag != 2 else "-")

                # F. PRINT & NEXT STEP
                # s_val = self.prod_hmm.prod_states[curr_s_idx]
                # o_val = "START" if t == 0 else self.prod_hmm.hmm.observations_indx_to_obs_dict[
                #     o_prev_idx]

                # print(f"{t:<5} | {str(s_val):<20} | {str(o_val):<12} | {dist_str}")

                # obs_name = str(o_idx)
                # if hasattr(self.prod_hmm.hmm, 'observations_indx_to_obs_dict'):
                #     obs_name = self.prod_hmm.hmm.observations_indx_to_obs_dict.get(o_idx, str(o_idx))
                #
                # print(
                #     f"{t:<5} | {curr_s_idx:<10} | {act_name:<10} | {obs_name:<10} | {p_safe_val:.4f}     | {pred_flag:<5} | {result_marker}")

                # Safe conversion to string to handle Sets/Tuples
                s_str = str(self.prod_hmm.prod_states[curr_s_idx])
                act_str = str(act_name)

                # Handle obs_name specifically
                obs_name_raw = o_idx
                if hasattr(self.prod_hmm.hmm, 'observations_indx_to_obs_dict'):
                    obs_name_raw = self.prod_hmm.hmm.observations_indx_to_obs_dict.get(o_idx, str(o_idx))
                obs_str = str(obs_name_raw)

                # CORRECTED PRINT LINE
                print(
                    f"{t:<5} | {s_str:<10} | {act_str:<10} | {obs_str:<10} | {p_safe_val:.4f}     | {pred_flag:<5} | {result_marker}")

                obs_input = torch.tensor([o_idx], dtype=torch.long, device=device)

        # Final Stats
        print("-" * 100)
        acc = (total_correct / total) * 100.0
        print(f"Horizon: {horizon} | Correct: {total_correct} | Wrong: {total_wrong}")
        print(f"Strict Accuracy: {acc:.2f}%")
        print("=" * 100 + "\n")

        return acc

    # def evaluate_brier_score_agent(self, episodes=100, horizon=20, k_step=2, uniform_random=False):
    #     print("\n" + "=" * 100)
    #     policy_name = "UNIFORM RANDOM" if uniform_random else "LSTM"
    #     print(f" AGENT EVALUATION ({policy_name}) - Horizon={horizon}, k={k_step}")
    #     print("=" * 100)
    #
    #     # --- PRE-COMPUTATION ---
    #     I_N = self.I_N
    #     T = self.T
    #     V_safe_vec = torch.ones(T.shape[0], device=device)
    #     safe_mask = torch.diagonal(I_N)
    #
    #     V_safe_vec = V_safe_vec * safe_mask
    #     for _ in range(k_step):
    #         V_safe_vec = torch.matmul(T.t(), V_safe_vec)
    #         V_safe_vec = V_safe_vec * safe_mask
    #
    #     V_safe_row = V_safe_vec.unsqueeze(0)
    #     safety_mask_np = torch.diagonal(I_N).cpu().numpy()
    #
    #     # --- METRICS ---
    #     brier_sum = 0.0
    #     correct_samples = 0
    #     total_steps = 0
    #
    #     for ep in range(episodes):
    #         # GENERATE REALITY
    #         curr = torch.multinomial(self.mu_0, 1).item()
    #         true_states = [curr]
    #         for _ in range(horizon + k_step):
    #             probs = self.T[:, true_states[-1]]
    #             next_s = torch.multinomial(probs, 1).item()
    #             true_states.append(next_s)
    #
    #         # INITIALIZE AGENT
    #         hidden = self.policy_net.init_hidden(1)
    #         obs_input = torch.zeros(1, dtype=torch.long, device=device)
    #         alpha = self.mu_0.clone().unsqueeze(0)
    #
    #         for t in range(horizon):
    #             curr_s_idx = true_states[t]
    #
    #             # SELECT ACTION
    #             if not uniform_random:
    #                 logits, hidden = self.policy_net(obs_input, hidden)
    #                 probs_action = F.softmax(logits, dim=1)
    #                 a_idx = torch.multinomial(probs_action, 1).item()
    #             else:
    #                 a_idx = torch.randint(low=0, high=self.num_sensor_queries, size=(1,)).item()
    #
    #             # ENVIRONMENT
    #             probs_obs = self.B[a_idx, curr_s_idx, :]
    #             o_idx = torch.multinomial(probs_obs, 1).item()
    #
    #             # COMPUTE P_SAFE
    #             B_vec = self.B[a_idx, :, o_idx].unsqueeze(0)
    #             joint_belief = alpha * B_vec
    #             denom = joint_belief.sum(dim=1)
    #             denom_stable = denom + 1e-10
    #             num = (joint_belief * V_safe_row).sum(dim=1)
    #
    #             p_safe = num / denom_stable
    #             p_safe_val = torch.clamp(p_safe, 1e-6, 1.0 - 1e-6).item()
    #
    #             # UPDATE BELIEF
    #             alpha_next_unnorm = torch.matmul(joint_belief, self.T.t())
    #             alpha = alpha_next_unnorm / denom_stable.unsqueeze(1)
    #
    #             # GROUND TRUTH CHECK
    #             actual_safe = True
    #             for i in range(k_step + 1):
    #                 if safety_mask_np[true_states[t + i]] == 0.0:
    #                     actual_safe = False
    #                     break
    #
    #             outcome = 1.0 if actual_safe else 0.0
    #
    #             # RECORD METRICS
    #             brier_sum += (p_safe_val - outcome) ** 2
    #
    #             agent_pred = 1 if torch.rand(1).item() < p_safe_val else 0
    #             if (agent_pred == 1 and actual_safe) or (agent_pred == 0 and not actual_safe):
    #                 correct_samples += 1
    #
    #             total_steps += 1
    #             obs_input = torch.tensor([o_idx], dtype=torch.long, device=device)
    #
    #     avg_brier = brier_sum / total_steps if total_steps > 0 else 0
    #     acc = (correct_samples / total_steps) * 100.0 if total_steps > 0 else 0
    #
    #     print(f"Final Agent Brier Score: {avg_brier:.5f}")
    #     print(f"Final Agent Sampling Acc: {acc:.4f}%")
    #     print("=" * 100 + "\n")
    #
    #     return avg_brier, acc

    # def evaluate_brier_score_agent(self, episodes=100, horizon=20, k_step=2, uniform_random=False):
    #     print("\n" + "=" * 100)
    #     policy_name = "UNIFORM RANDOM" if uniform_random else "LSTM"
    #     print(f" AGENT EVALUATION ({policy_name}) - Horizon={horizon}, k={k_step}")
    #     print("=" * 100)
    #
    #     # --- PRE-COMPUTATION ---
    #     I_N = self.I_N
    #     T = self.T
    #     V_safe_vec = torch.ones(T.shape[0], device=device)  # Ensure device is correct
    #     safe_mask = torch.diagonal(I_N)
    #
    #     V_safe_vec = V_safe_vec * safe_mask
    #     for _ in range(k_step):
    #         V_safe_vec = torch.matmul(T.t(), V_safe_vec)
    #         V_safe_vec = V_safe_vec * safe_mask
    #
    #     V_safe_row = V_safe_vec.unsqueeze(0)
    #     safety_mask_np = torch.diagonal(I_N).cpu().numpy()
    #
    #     # --- NEW: METRIC TENSORS FOR CI COMPUTATION ---
    #     # Store the average score/accuracy of each individual episode
    #     ep_brier_scores = torch.zeros(episodes, device=device)
    #     ep_accuracies = torch.zeros(episodes, device=device)
    #
    #     for ep in range(episodes):
    #         # GENERATE REALITY
    #         curr = torch.multinomial(self.mu_0, 1).item()
    #         true_states = [curr]
    #         for _ in range(horizon + k_step):
    #             probs = self.T[:, true_states[-1]]
    #             next_s = torch.multinomial(probs, 1).item()
    #             true_states.append(next_s)
    #
    #         # INITIALIZE AGENT
    #         hidden = self.policy_net.init_hidden(1)
    #         obs_input = torch.zeros(1, dtype=torch.long, device=device)
    #         alpha = self.mu_0.clone().unsqueeze(0)
    #
    #         # Track metrics for THIS specific episode
    #         ep_brier_sum = 0.0
    #         ep_correct = 0
    #
    #         for t in range(horizon):
    #             curr_s_idx = true_states[t]
    #
    #             # SELECT ACTION
    #             if not uniform_random:
    #                 logits, hidden = self.policy_net(obs_input, hidden)
    #                 probs_action = F.softmax(logits, dim=1)
    #                 a_idx = torch.multinomial(probs_action, 1).item()
    #             else:
    #                 a_idx = torch.randint(low=0, high=self.num_sensor_queries, size=(1,)).item()
    #
    #             # ENVIRONMENT
    #             probs_obs = self.B[a_idx, curr_s_idx, :]
    #             o_idx = torch.multinomial(probs_obs, 1).item()
    #
    #             # COMPUTE P_SAFE
    #             B_vec = self.B[a_idx, :, o_idx].unsqueeze(0)
    #             joint_belief = alpha * B_vec
    #             denom = joint_belief.sum(dim=1)
    #             denom_stable = denom + 1e-10
    #             num = (joint_belief * V_safe_row).sum(dim=1)
    #
    #             p_safe = num / denom_stable
    #             p_safe_val = torch.clamp(p_safe, 1e-6, 1.0 - 1e-6).item()
    #
    #             # UPDATE BELIEF
    #             alpha_next_unnorm = torch.matmul(joint_belief, self.T.t())
    #             alpha = alpha_next_unnorm / denom_stable.unsqueeze(1)
    #
    #             # GROUND TRUTH CHECK
    #             actual_safe = True
    #             for i in range(k_step + 1):
    #                 if safety_mask_np[true_states[t + i]] == 0.0:
    #                     actual_safe = False
    #                     break
    #
    #             outcome = 1.0 if actual_safe else 0.0
    #
    #             # RECORD METRICS FOR THIS EPISODE
    #             ep_brier_sum += (p_safe_val - outcome) ** 2
    #
    #             agent_pred = 1 if torch.rand(1).item() < p_safe_val else 0
    #             if (agent_pred == 1 and actual_safe) or (agent_pred == 0 and not actual_safe):
    #                 ep_correct += 1
    #
    #             obs_input = torch.tensor([o_idx], dtype=torch.long, device=device)
    #
    #         # SAVE EPISODE AVERAGES
    #         ep_brier_scores[ep] = ep_brier_sum / horizon
    #         ep_accuracies[ep] = (ep_correct / horizon) * 100.0
    #
    #     # --- COMPUTE MEANS AND 95% CONFIDENCE INTERVALS (PURE PYTORCH) ---
    #     mean_brier = ep_brier_scores.mean().item()
    #     std_brier = ep_brier_scores.std(unbiased=True).item()
    #     # ci_brier = 1.96 * (std_brier / math.sqrt(episodes))
    #     ci_brier = 1.96 * (std_brier / (episodes ** 0.5))
    #
    #     mean_acc = ep_accuracies.mean().item()
    #     std_acc = ep_accuracies.std(unbiased=True).item()
    #     # ci_acc = 1.96 * (std_acc / math.sqrt(episodes))
    #     ci_acc = 1.96 * (std_acc / (episodes ** 0.5))
    #
    #     print(f"Final Agent Brier Score: {mean_brier:.6f} ± {ci_brier:.6f}")
    #     print(f"Final Agent Sampling Acc: {mean_acc:.6f}% ± {ci_acc:.6f}%")
    #     print("=" * 100 + "\n")
    #
    #     return mean_brier, ci_brier, mean_acc, ci_acc

    # def evaluate_brier_score_agent_no_accumulation_after_death(self, episodes=100, horizon=20, k_step=2, uniform_random=False):
    #     print("\n" + "=" * 100)
    #     policy_name = "UNIFORM RANDOM" if uniform_random else "LSTM"
    #     print(f" AGENT EVALUATION ({policy_name}) - Horizon={horizon}, k={k_step}")
    #     print("=" * 100)
    #
    #     # --- PRE-COMPUTATION ---
    #     I_N = self.I_N
    #     T = self.T
    #     V_safe_vec = torch.ones(T.shape[0], device=device)
    #     safe_mask = torch.diagonal(I_N)
    #
    #     V_safe_vec = V_safe_vec * safe_mask
    #     for _ in range(k_step):
    #         V_safe_vec = torch.matmul(T.t(), V_safe_vec)
    #         V_safe_vec = V_safe_vec * safe_mask
    #
    #     V_safe_row = V_safe_vec.unsqueeze(0)
    #     safety_mask_np = torch.diagonal(I_N).cpu().numpy()
    #
    #     # Store the average score/accuracy of each individual episode
    #     ep_brier_scores = torch.zeros(episodes, device=device)
    #     ep_accuracies = torch.zeros(episodes, device=device)
    #
    #     for ep in range(episodes):
    #         # GENERATE REALITY
    #         curr = torch.multinomial(self.mu_0, 1).item()
    #         true_states = [curr]
    #         for _ in range(horizon + k_step):
    #             probs = self.T[:, true_states[-1]]
    #             next_s = torch.multinomial(probs, 1).item()
    #             true_states.append(next_s)
    #
    #         # INITIALIZE AGENT
    #         hidden = self.policy_net.init_hidden(1)
    #         obs_input = torch.zeros(1, dtype=torch.long, device=device)
    #         alpha = self.mu_0.clone().unsqueeze(0)
    #
    #         ep_brier_sum = 0.0
    #         ep_correct = 0
    #         steps_counted = 0
    #
    #         # ONLY PRINT DETAILS FOR THE FIRST 20 EPISODES TO PREVENT TERMINAL SPAM
    #         # print_details = (ep < 20)
    #         print_details = False
    #         if print_details:
    #             print(f"\n--- Episode {ep + 1} Trace ---")
    #
    #         for t in range(horizon):
    #             curr_s_idx = true_states[t]
    #
    #             # --- HYPOTHESIS TEST: Check if state is physically absorbing
    #             is_absorbing = (self.T[curr_s_idx, curr_s_idx].item() == 1.0)
    #
    #             # SELECT ACTION
    #             if not uniform_random:
    #                 logits, hidden = self.policy_net(obs_input, hidden)
    #                 probs_action = F.softmax(logits, dim=1)
    #                 a_idx = torch.multinomial(probs_action, 1).item()
    #             else:
    #                 a_idx = torch.randint(low=0, high=self.num_sensor_queries, size=(1,)).item()
    #
    #             # ENVIRONMENT
    #             probs_obs = self.B[a_idx, curr_s_idx, :]
    #             o_idx = torch.multinomial(probs_obs, 1).item()
    #
    #             # COMPUTE P_SAFE
    #             B_vec = self.B[a_idx, :, o_idx].unsqueeze(0)
    #             joint_belief = alpha * B_vec
    #             denom = joint_belief.sum(dim=1)
    #             denom_stable = denom + 1e-10
    #             num = (joint_belief * V_safe_row).sum(dim=1)
    #
    #             p_safe = num / denom_stable
    #             p_safe_val = torch.clamp(p_safe, 1e-6, 1.0 - 1e-6).item()
    #
    #             # UPDATE BELIEF
    #             alpha_next_unnorm = torch.matmul(joint_belief, self.T.t())
    #             alpha = alpha_next_unnorm / denom_stable.unsqueeze(1)
    #
    #             # GROUND TRUTH CHECK
    #             actual_safe = True
    #             for i in range(k_step + 1):
    #                 if safety_mask_np[true_states[t + i]] == 0.0:
    #                     actual_safe = False
    #                     break
    #
    #             outcome = 1.0 if actual_safe else 0.0
    #
    #             # RECORD METRICS FOR THIS EPISODE
    #             ep_brier_sum += (p_safe_val - outcome) ** 2
    #
    #             agent_pred = 1 if torch.rand(1).item() < p_safe_val else 0
    #             is_correct = (agent_pred == 1 and actual_safe) or (agent_pred == 0 and not actual_safe)
    #             if is_correct:
    #                 ep_correct += 1
    #
    #             steps_counted += 1
    #
    #             if print_details:
    #                 act_curr_s = self.prod_hmm.prod_indx_states_dict[curr_s_idx]
    #                 final_curr_s = self.prod_hmm.hmm.indx_to_states_dict[act_curr_s[0]]
    #
    #                 print(f"t={t:02d} | TrueState={final_curr_s:03d} | Sensor={a_idx} | "
    #                       f"P(Safe)={p_safe_val:.4f} | PosteriorPred={agent_pred} | "
    #                       f"ActualSafe={actual_safe} | Correct={is_correct}")
    #
    #             obs_input = torch.tensor([o_idx], dtype=torch.long, device=device)
    #
    #             # --- NEW: STOP ON TERMINATION LOGIC ---
    #             # Break ONLY if the robot is physically dead AND the agent has realized it.
    #             if is_absorbing and p_safe_val < 0.01:
    #                 if print_details:
    #                     print(
    #                         f">>> Step {t}: Absorbing state confirmed AND agent realized failure. Terminating episode.")
    #                 break
    #
    #         # SAVE EPISODE AVERAGES
    #         if steps_counted > 0:
    #             ep_brier_scores[ep] = ep_brier_sum / steps_counted
    #             ep_accuracies[ep] = (ep_correct / steps_counted) * 100.0
    #         else:
    #             ep_brier_scores[ep] = 0.0
    #             ep_accuracies[ep] = 100.0
    #
    #     # --- COMPUTE MEANS AND 95% CONFIDENCE INTERVALS ---
    #     mean_brier = ep_brier_scores.mean().item()
    #     std_brier = ep_brier_scores.std(unbiased=True).item()
    #     ci_brier = 1.96 * (std_brier / (episodes ** 0.5))
    #
    #     mean_acc = ep_accuracies.mean().item()
    #     std_acc = ep_accuracies.std(unbiased=True).item()
    #     ci_acc = 1.96 * (std_acc / (episodes ** 0.5))
    #
    #     print(f"Final Agent Brier Score: {mean_brier:.6f} ± {ci_brier:.6f}")
    #     print(f"Final Agent Sampling Acc: {mean_acc:.6f}% ± {ci_acc:.6f}%")
    #     print("=" * 100 + "\n")
    #
    #     return mean_brier, ci_brier, mean_acc, ci_acc

    def evaluate_brier_score_agent(self, episodes=100, horizon=20, k_step=2, uniform_random=False):
        print("\n" + "=" * 100)
        policy_name = "UNIFORM RANDOM" if uniform_random else "LSTM"
        print(f" AGENT EVALUATION ({policy_name}) - Horizon={horizon}, k={k_step}")
        print("=" * 100)

        # --- PRE-COMPUTATION ---
        I_N = self.I_N
        T = self.T
        V_safe_vec = torch.ones(T.shape[0], device=device)
        safe_mask = torch.diagonal(I_N)

        V_safe_vec = V_safe_vec * safe_mask
        for _ in range(k_step):
            V_safe_vec = torch.matmul(T.t(), V_safe_vec)
            V_safe_vec = V_safe_vec * safe_mask

        V_safe_row = V_safe_vec.unsqueeze(0)
        safety_mask_np = torch.diagonal(I_N).cpu().numpy()

        ep_brier_scores = torch.zeros(episodes, device=device)
        ep_accuracies = torch.zeros(episodes, device=device)
        ep_costs = torch.zeros(episodes, device=device)

        for ep in range(episodes):
            # GENERATE REALITY
            curr = torch.multinomial(self.mu_0, 1).item()
            true_states = [curr]
            for _ in range(horizon + k_step):
                probs = self.T[:, true_states[-1]]
                next_s = torch.multinomial(probs, 1).item()
                true_states.append(next_s)

            # INITIALIZE AGENT
            hidden = self.policy_net.init_hidden(1)
            obs_input = torch.zeros(1, dtype=torch.long, device=device)
            alpha = self.mu_0.clone().unsqueeze(0)

            ep_brier_sum = 0.0
            ep_correct = 0
            ep_cost_sum = 0.0

            # Initialize previous action to the "no query" state (assumed to be the last index)
            prev_a_idx = self.num_sensor_queries - 1

            for t in range(horizon):
                curr_s_idx = true_states[t]

                # SELECT ACTION
                if not uniform_random:
                    logits, hidden = self.policy_net(obs_input, hidden)
                    probs_action = F.softmax(logits, dim=1)
                    a_idx = torch.multinomial(probs_action, 1).item()
                else:
                    a_idx = torch.randint(low=0, high=self.num_sensor_queries, size=(1,)).item()

                # NEW: Calculate 2D cost and update previous action
                ep_cost_sum += self.cost_matrix[prev_a_idx, a_idx].item()
                prev_a_idx = a_idx

                # ENVIRONMENT
                probs_obs = self.B[a_idx, curr_s_idx, :]
                o_idx = torch.multinomial(probs_obs, 1).item()

                # COMPUTE P_SAFE
                B_vec = self.B[a_idx, :, o_idx].unsqueeze(0)
                joint_belief = alpha * B_vec
                denom = joint_belief.sum(dim=1)
                denom_stable = denom + 1e-10
                num = (joint_belief * V_safe_row).sum(dim=1)

                p_safe = num / denom_stable
                p_safe_val = torch.clamp(p_safe, 1e-6, 1.0 - 1e-6).item()

                # UPDATE BELIEF
                alpha_next_unnorm = torch.matmul(joint_belief, self.T.t())
                alpha = alpha_next_unnorm / denom_stable.unsqueeze(1)

                # GROUND TRUTH CHECK
                actual_safe = True
                for i in range(k_step + 1):
                    if safety_mask_np[true_states[t + i]] == 0.0:
                        actual_safe = False
                        break

                outcome = 1.0 if actual_safe else 0.0

                # RECORD METRICS
                ep_brier_sum += (p_safe_val - outcome) ** 2

                agent_pred = 1 if torch.rand(1).item() < p_safe_val else 0
                if (agent_pred == 1 and actual_safe) or (agent_pred == 0 and not actual_safe):
                    ep_correct += 1

                obs_input = torch.tensor([o_idx], dtype=torch.long, device=device)

            # SAVE EPISODE AVERAGES
            ep_brier_scores[ep] = ep_brier_sum / horizon
            ep_accuracies[ep] = (ep_correct / horizon) * 100.0
            ep_costs[ep] = ep_cost_sum

            # --- COMPUTE MEANS AND 95% CONFIDENCE INTERVALS ---
        mean_brier = ep_brier_scores.mean().item()
        std_brier = ep_brier_scores.std(unbiased=True).item()
        ci_brier = 1.96 * (std_brier / (episodes ** 0.5))

        mean_acc = ep_accuracies.mean().item()
        std_acc = ep_accuracies.std(unbiased=True).item()
        ci_acc = 1.96 * (std_acc / (episodes ** 0.5))

        mean_cost = ep_costs.mean().item()
        std_cost = ep_costs.std(unbiased=True).item()
        ci_cost = 1.96 * (std_cost / (episodes ** 0.5))

        print(f"Final Agent Brier Score: {mean_brier:.6f} ± {ci_brier:.6f}")
        print(f"Final Agent Sampling Acc: {mean_acc:.6f}% ± {ci_acc:.6f}%")
        print(f"Final Agent Average Cost/Step: {mean_cost:.6f} ± {ci_cost:.6f}")
        print("=" * 100 + "\n")

        return mean_brier, ci_brier, mean_acc, ci_acc, mean_cost, ci_cost

    def evaluate_brier_score_agent_no_accumulation_after_death(self, episodes=100, horizon=20, k_step=2,
                                                               uniform_random=False):
        print("\n" + "=" * 100)
        policy_name = "UNIFORM RANDOM" if uniform_random else "LSTM"
        print(f" AGENT EVALUATION ({policy_name}) - Horizon={horizon}, k={k_step}")
        print("=" * 100)

        # --- PRE-COMPUTATION ---
        I_N = self.I_N
        T = self.T
        V_safe_vec = torch.ones(T.shape[0], device=device)
        safe_mask = torch.diagonal(I_N)

        V_safe_vec = V_safe_vec * safe_mask
        for _ in range(k_step):
            V_safe_vec = torch.matmul(T.t(), V_safe_vec)
            V_safe_vec = V_safe_vec * safe_mask

        V_safe_row = V_safe_vec.unsqueeze(0)
        safety_mask_np = torch.diagonal(I_N).cpu().numpy()

        ep_brier_scores = torch.zeros(episodes, device=device)
        ep_accuracies = torch.zeros(episodes, device=device)
        ep_costs = torch.zeros(episodes, device=device)

        for ep in range(episodes):
            # GENERATE REALITY
            curr = torch.multinomial(self.mu_0, 1).item()
            true_states = [curr]
            for _ in range(horizon + k_step):
                probs = self.T[:, true_states[-1]]
                next_s = torch.multinomial(probs, 1).item()
                true_states.append(next_s)

            # INITIALIZE AGENT
            hidden = self.policy_net.init_hidden(1)
            obs_input = torch.zeros(1, dtype=torch.long, device=device)
            alpha = self.mu_0.clone().unsqueeze(0)

            ep_brier_sum = 0.0
            ep_correct = 0
            ep_cost_sum = 0.0
            steps_counted = 0

            # Initialize previous action
            prev_a_idx = self.num_sensor_queries - 1

            for t in range(horizon):
                curr_s_idx = true_states[t]

                is_absorbing = (self.T[curr_s_idx, curr_s_idx].item() == 1.0)

                # SELECT ACTION
                if not uniform_random:
                    logits, hidden = self.policy_net(obs_input, hidden)
                    probs_action = F.softmax(logits, dim=1)
                    a_idx = torch.multinomial(probs_action, 1).item()
                else:
                    a_idx = torch.randint(low=0, high=self.num_sensor_queries, size=(1,)).item()

                # NEW: Calculate 2D cost and update previous action
                ep_cost_sum += self.cost_matrix[prev_a_idx, a_idx].item()
                prev_a_idx = a_idx

                # ENVIRONMENT
                probs_obs = self.B[a_idx, curr_s_idx, :]
                o_idx = torch.multinomial(probs_obs, 1).item()

                # COMPUTE P_SAFE
                B_vec = self.B[a_idx, :, o_idx].unsqueeze(0)
                joint_belief = alpha * B_vec
                denom = joint_belief.sum(dim=1)
                denom_stable = denom + 1e-10
                num = (joint_belief * V_safe_row).sum(dim=1)

                p_safe = num / denom_stable
                p_safe_val = torch.clamp(p_safe, 1e-6, 1.0 - 1e-6).item()

                # UPDATE BELIEF
                alpha_next_unnorm = torch.matmul(joint_belief, self.T.t())
                alpha = alpha_next_unnorm / denom_stable.unsqueeze(1)

                # GROUND TRUTH CHECK
                actual_safe = True
                for i in range(k_step + 1):
                    if safety_mask_np[true_states[t + i]] == 0.0:
                        actual_safe = False
                        break

                outcome = 1.0 if actual_safe else 0.0

                # RECORD METRICS
                ep_brier_sum += (p_safe_val - outcome) ** 2

                agent_pred = 1 if torch.rand(1).item() < p_safe_val else 0
                is_correct = (agent_pred == 1 and actual_safe) or (agent_pred == 0 and not actual_safe)
                if is_correct:
                    ep_correct += 1

                steps_counted += 1
                obs_input = torch.tensor([o_idx], dtype=torch.long, device=device)

                # STOP ON TERMINATION LOGIC
                if is_absorbing and p_safe_val < 0.01:
                    break

            # SAVE EPISODE AVERAGES
            if steps_counted > 0:
                ep_brier_scores[ep] = ep_brier_sum / steps_counted
                ep_accuracies[ep] = (ep_correct / steps_counted) * 100.0
                ep_costs[ep] = ep_cost_sum
            else:
                ep_brier_scores[ep] = 0.0
                ep_accuracies[ep] = 100.0
                ep_costs[ep] = 0.0

        # --- COMPUTE MEANS AND 95% CONFIDENCE INTERVALS ---
        mean_brier = ep_brier_scores.mean().item()
        std_brier = ep_brier_scores.std(unbiased=True).item()
        ci_brier = 1.96 * (std_brier / (episodes ** 0.5))

        mean_acc = ep_accuracies.mean().item()
        std_acc = ep_accuracies.std(unbiased=True).item()
        ci_acc = 1.96 * (std_acc / (episodes ** 0.5))

        mean_cost = ep_costs.mean().item()
        std_cost = ep_costs.std(unbiased=True).item()
        ci_cost = 1.96 * (std_cost / (episodes ** 0.5))

        print(f"Final Agent Brier Score: {mean_brier:.6f} ± {ci_brier:.6f}")
        print(f"Final Agent Sampling Acc: {mean_acc:.6f}% ± {ci_acc:.6f}%")
        print(f"Final Agent Average Cost/Step: {mean_cost:.6f} ± {ci_cost:.6f}")
        print("=" * 100 + "\n")

        return mean_brier, ci_brier, mean_acc, ci_acc, mean_cost, ci_cost

    # def evaluate_oracle_baseline(self, episodes=100, horizon=20, k_step=2):
    #     print("\n" + "=" * 100)
    #     print(f" ORACLE BASELINE EVALUATION - Horizon={horizon}, k={k_step}")
    #     print("=" * 100)
    #
    #     # --- PRE-COMPUTATION ---
    #     I_N = self.I_N
    #     T = self.T
    #     V_safe_vec = torch.ones(T.shape[0], device=device)
    #     safe_mask = torch.diagonal(I_N)
    #
    #     V_safe_vec = V_safe_vec * safe_mask
    #     for _ in range(k_step):
    #         V_safe_vec = torch.matmul(T.t(), V_safe_vec)
    #         V_safe_vec = V_safe_vec * safe_mask
    #
    #     safety_mask_np = torch.diagonal(I_N).cpu().numpy()
    #
    #     # --- METRICS ---
    #     brier_sum = 0.0
    #     correct_samples = 0
    #     total_steps = 0
    #
    #     for ep in range(episodes):
    #         # GENERATE REALITY
    #         curr = torch.multinomial(self.mu_0, 1).item()
    #         true_states = [curr]
    #         for _ in range(horizon + k_step):
    #             probs = self.T[:, true_states[-1]]
    #             next_s = torch.multinomial(probs, 1).item()
    #             true_states.append(next_s)
    #
    #         for t in range(horizon):
    #             curr_s_idx = true_states[t]
    #
    #             # ORACLE P_SAFE (Direct Lookup)
    #             p_safe_oracle = V_safe_vec[curr_s_idx].item()
    #             # p_safe_oracle = max(min(p_safe_oracle, 1.0 - 1e-6), 1e-6)
    #             # print(f"Current state: {self.prod_hmm.prod_states[curr_s_idx]}; Oracle P(safe) = {p_safe_oracle:.5f}")
    #
    #             # GROUND TRUTH CHECK
    #             actual_safe = True
    #             for i in range(k_step + 1):
    #                 if safety_mask_np[true_states[t + i]] == 0.0:
    #                     actual_safe = False
    #                     break
    #
    #             outcome = 1.0 if actual_safe else 0.0
    #
    #             # RECORD METRICS
    #             brier_sum += (p_safe_oracle - outcome) ** 2
    #
    #             oracle_pred = 1 if torch.rand(1).item() < p_safe_oracle else 0
    #             if (oracle_pred == 1 and actual_safe) or (oracle_pred == 0 and not actual_safe):
    #                 correct_samples += 1
    #
    #             total_steps += 1
    #
    #     avg_brier = brier_sum / total_steps if total_steps > 0 else 0
    #     acc = (correct_samples / total_steps) * 100.0 if total_steps > 0 else 0
    #
    #     print(f"Final Oracle Brier Score: {avg_brier:.5f}")
    #     print(f"Final Oracle Sampling Acc: {acc:.4f}%")
    #     print("=" * 100 + "\n")
    #
    #     return avg_brier, acc

    def evaluate_oracle_baseline(self, episodes=100, horizon=20, k_step=2):
        print("\n" + "=" * 100)
        print(f" ORACLE BASELINE EVALUATION - Horizon={horizon}, k={k_step}")
        print("=" * 100)

        # --- PRE-COMPUTATION ---
        I_N = self.I_N
        T = self.T

        # Ensuring we use self.device for consistency across the class
        V_safe_vec = torch.ones(T.shape[0], device=device)
        safe_mask = torch.diagonal(I_N)

        V_safe_vec = V_safe_vec * safe_mask
        for _ in range(k_step):
            V_safe_vec = torch.matmul(T.t(), V_safe_vec)
            V_safe_vec = V_safe_vec * safe_mask

        safety_mask_np = torch.diagonal(I_N).cpu().numpy()

        # --- NEW: METRIC TENSORS FOR CI COMPUTATION ---
        # Store the average score/accuracy of each individual episode
        ep_brier_scores = torch.zeros(episodes, device=device)
        ep_accuracies = torch.zeros(episodes, device=device)

        for ep in range(episodes):
            # GENERATE REALITY
            curr = torch.multinomial(self.mu_0, 1).item()
            true_states = [curr]
            for _ in range(horizon + k_step):
                probs = self.T[:, true_states[-1]]
                next_s = torch.multinomial(probs, 1).item()
                true_states.append(next_s)

            # Track metrics for THIS specific episode
            ep_brier_sum = 0.0
            ep_correct = 0

            for t in range(horizon):
                curr_s_idx = true_states[t]

                # ORACLE P_SAFE (Direct Lookup)
                p_safe_oracle = V_safe_vec[curr_s_idx].item()

                # GROUND TRUTH CHECK
                actual_safe = True
                for i in range(k_step + 1):
                    if safety_mask_np[true_states[t + i]] == 0.0:
                        actual_safe = False
                        break

                outcome = 1.0 if actual_safe else 0.0

                # RECORD METRICS FOR THIS EPISODE
                ep_brier_sum += (p_safe_oracle - outcome) ** 2

                oracle_pred = 1 if torch.rand(1).item() < p_safe_oracle else 0
                if (oracle_pred == 1 and actual_safe) or (oracle_pred == 0 and not actual_safe):
                    ep_correct += 1

            # SAVE EPISODE AVERAGES
            ep_brier_scores[ep] = ep_brier_sum / horizon
            ep_accuracies[ep] = (ep_correct / horizon) * 100.0

        # --- COMPUTE MEANS AND 95% CONFIDENCE INTERVALS (PURE PYTORCH) ---
        mean_brier = ep_brier_scores.mean().item()
        std_brier = ep_brier_scores.std(unbiased=True).item()
        ci_brier = 1.96 * (std_brier / (episodes ** 0.5))

        mean_acc = ep_accuracies.mean().item()
        std_acc = ep_accuracies.std(unbiased=True).item()
        ci_acc = 1.96 * (std_acc / (episodes ** 0.5))

        print(f"Final Oracle Brier Score: {mean_brier:.6f} ± {ci_brier:.6f}")
        print(f"Final Oracle Sampling Acc: {mean_acc:.6f}% ± {ci_acc:.6f}%")
        print("=" * 100 + "\n")

        return mean_brier, ci_brier, mean_acc, ci_acc
