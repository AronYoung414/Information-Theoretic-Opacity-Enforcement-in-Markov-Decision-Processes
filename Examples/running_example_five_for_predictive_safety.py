import math

from setup_and_solvers.LP_for_nominal_policy import *
from setup_and_solvers.test_gradient_entropy_calculations_for_predictive_safety import *
from setup_and_solvers.markov_decision_process import *


# # --- 1. DEFINE SEED FUNCTION (Global Helper) ---
# def set_seed(seed=0):
#     """
#     Locks all random number generators for reproducibility.
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#     print(f"Global Seed set to: {seed}")
#
#
# set_seed(42)


def running_example_predictive_safety(iter_num=1000, batch_size=100, V=100, T=3, lr=0.005, k_steps=1, alpha=0,
                                      prior_compute_flag=0, num_test_episodes=1000):
    logger.add("logs_for_examples/log_file_for_running_example_five_predictive_safety.log")

    logger.info("This is the log file for the running example with DFA based failure states for predictive safety.")

    # Initial set-up for the MDP.

    states = [0, 1, 2, 3, 4, 5]
    actions = ['a']

    prob = {
        'a': np.array([[0, 0.8, 0, 0, 0, 0.2], [0.2, 0.1, 0.5, 0, 0, 0.2], [0, 0, 0, 0.8, 0, 0.2],
                       [0, 0, 0, 0, 0.8, 0.2], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    }

    transitions = {
        (0, 'a'): [(1, 0.8), (5, 0.2)],
        (1, 'a'): [(1, 0.1), (2, 0.7), (5, 0.2)],
        (2, 'a'): [(3, 0.8), (5, 0.2)],
        (3, 'a'): [(4, 0.8), (5, 0.2)],
        (4, 'a'): [(4, 1)],
        (5, 'a'): [(5, 1)]
    }

    target = [5]
    failure_states = [5]

    initial = [0]

    initial_dist = dict([])
    # Initial distribution is uniform over the initial states. This is a simplification for the current example
    for state in states:
        if state in initial:
            initial_dist[state] = 1 / len(initial)
        else:
            initial_dist[state] = 0

    # Labeling function for the states.
    labels = dict([])
    for state in states:
        if state in failure_states:
            labels[state] = {'Fail'}
        else:
            labels[state] = {'Safe'}

    # sensor setup
    sensors = {'R', 'G', 'P', 'B', 'NO'}

    # coverage sets
    setR = {1}
    setG = {2}
    setP = {3}
    setB = {4}
    setNO = {0, 5}

    # masking actions
    sensing_action = dict([])

    sensing_action[0] = {'R'}
    sensing_action[1] = {'G'}
    sensing_action[2] = {'P'}
    sensing_action[3] = {'B'}
    # masking_action[4] = {'E'}
    sensing_action[4] = {'NO'}  # 'F' is the no sensing action.

    no_sense_act = 4

    # sensor noise
    # sensor_noise = 0.15
    sensor_noise = 0

    # sensor costs
    sensor_cost = dict([])
    sensor_cost['R'] = 10
    sensor_cost['G'] = 10
    sensor_cost['P'] = 10
    sensor_cost['B'] = 10
    # sensor_cost['E'] = 15
    sensor_cost['NO'] = 0  # Cost for not masking.

    # if threshold == 60:
    #     # eta = 3.2
    #     eta = 3.2
    #     kappa = 0.25

    # Define a threshold for sensor masking.
    # threshold = 60
    # threshold = 20

    # sensor_cost_normalization = sum(abs(cost) for cost in sensor_cost.values())
    #
    # # updating the sensor costs with normalized costs.
    # for sens in sensor_cost:
    #     sensor_cost[sens] = sensor_cost[sens] / sensor_cost_normalization
    #
    # # normalized threshold.
    # alpha = alpha / sensor_cost_normalization

    sensor_net = Sensor()
    sensor_net.sensors = sensors
    sensor_net.set_coverage('R', setR)
    sensor_net.set_coverage('G', setG)
    sensor_net.set_coverage('P', setP)
    sensor_net.set_coverage('B', setB)
    sensor_net.set_coverage('NO', setNO)

    sensor_net.sensing_actions = sensing_action
    sensor_net.sensor_noise = sensor_noise
    sensor_net.sensor_cost_dict = sensor_cost

    agent_mdp = MDP(init=initial, actlist=actions, states=states, prob=prob, trans=transitions, labels=labels,
                    init_dist=initial_dist,
                    goal_states=target)
    agent_mdp.get_supp()
    agent_mdp.gettrans()
    agent_mdp.get_reward()

    # Using the following agent policy.

    goal_policy = dict([])
    goal_policy[(0, 'a')] = 1
    goal_policy[(1, 'a')] = 1
    goal_policy[(2, 'a')] = 1
    goal_policy[(3, 'a')] = 1
    goal_policy[(4, 'a')] = 1
    goal_policy[(5, 'a')] = 1
    goal_policy[(6, 'a')] = 1

    logger.debug("Goal policy:")
    logger.debug(goal_policy)

    prior_compute_flag = prior_compute_flag

    if prior_compute_flag == 1:
        logger.info("Computing Prior Entropy via Monte Carlo...")

        # 1. Configuration
        total_simulations = 50000  # Run a large number of sims (e.g., 100k)
        horizon = T  # Use the actual Horizon T passed to the function!

        success_count = 0

        # 2. Run Simulations
        # (Optimized: No need for double loops, just one big loop)
        for _ in range(total_simulations):

            # A. Sample Initial State Correctly
            # Note: random.choice(list(init)) implies Uniform Distribution.
            # If your initial_dist is NOT uniform, use random.choices w/ weights.
            curr_state = random.choice(list(agent_mdp.init))

            # B. Rollout Trajectory
            is_success = False

            for step in range(horizon):
                # 1. Check if we hit the target/failure ALREADY?
                # (Depends on if your target is absorbing. Assuming standard Reachability:)
                if curr_state in target:
                    is_success = True
                    break  # Stop early if we reached goal

                # 2. Pick Action (Nominal Policy)
                # Assuming goal_policy is deterministic or stochastic
                # If deterministic dictionary map:
                # action = goal_policy[curr_state]
                # If stochastic distribution map:
                w_list = [goal_policy.get((curr_state, a), 0) for a in agent_mdp.actlist]
                action = random.choices(agent_mdp.actlist, weights=w_list)[0]

                # 3. Transition
                # Get next state probabilities
                next_states = list(agent_mdp.trans[curr_state][action].keys())
                probs = list(agent_mdp.trans[curr_state][action].values())

                curr_state = random.choices(next_states, weights=probs)[0]

            # Final check at end of horizon
            if curr_state in target:
                is_success = True

            if is_success:
                success_count += 1

        # 3. Calculate Global Probability ONCE
        p_success = success_count / total_simulations
        p_fail = 1.0 - p_success

        logger.debug(f"Estimated Probability of Success: {p_success:.4f}")

        # 4. Calculate Entropy ONCE
        # Handle 0log0 case safely
        if p_success <= 0 or p_success >= 1:
            total_prior_entropy = 0.0
        else:
            total_prior_entropy = -(p_success * math.log2(p_success) +
                                    p_fail * math.log2(p_fail))

        print(f"Mean prior entropy = {total_prior_entropy:.4f}")
        logger.debug(f"Mean prior entropy = {total_prior_entropy:.4f}")

    # if prior_compute_flag == 1:
    #
    #     # Computing the prior entropy.
    #     # Monte carlo simulation to obtain the approximate probability of being in the final state in T=10.
    #
    #     # prior_list = list()
    #     # iterations_list = list()
    #     total_prior = 0
    #
    #     for iterations in range(1000):
    #         prior_entropy = 0
    #         counter = 0
    #         horizon = 5
    #         final_state_goal_state = 0
    #         final_state_not_goal_state = 0
    #
    #         while counter <= 1000:
    #             new_init_state = random.choice(list(agent_mdp.init))
    #
    #             for i in range(horizon):
    #                 weights_list = list()
    #                 for action in agent_mdp.actlist:
    #                     weights_list.append(goal_policy[(new_init_state, action)])
    #                 action_to_play = random.choices(agent_mdp.actlist, weights_list)[0]
    #
    #                 post_states = list(agent_mdp.suppDict[(new_init_state, action_to_play)])
    #                 states_weights_list = list()
    #                 for st in post_states:
    #                     states_weights_list.append(agent_mdp.trans[new_init_state][action_to_play][st])
    #
    #                 next_state = random.choices(post_states, states_weights_list)[0]
    #                 new_init_state = next_state
    #
    #             if new_init_state in target:
    #                 final_state_goal_state += 1
    #             else:
    #                 final_state_not_goal_state += 1
    #
    #             counter += 1
    #
    #         probability_of_raching_final_state = final_state_goal_state / 1001
    #         # print(f"Probability of reaching goal state within T steps: {probability_of_raching_final_state}")
    #         # prior_entropy = probability_of_raching_final_state * math.log2(probability_of_raching_final_state) + (
    #         #             (1 - probability_of_raching_final_state) * math.log2(1 - probability_of_raching_final_state))
    #
    #         if probability_of_raching_final_state == 0:
    #             prior_entropy_part_1 = 0
    #         else:
    #             prior_entropy_part_1 = probability_of_raching_final_state * math.log2(
    #                 probability_of_raching_final_state)
    #
    #         if (1 - probability_of_raching_final_state) == 0:
    #             prior_entropy_part_2 = 0
    #         else:
    #             prior_entropy_part_2 = (1 - probability_of_raching_final_state) * math.log2(
    #                 1 - probability_of_raching_final_state)
    #
    #         prior_entropy = prior_entropy_part_1 + prior_entropy_part_2
    #
    #         # print(f"Prior entropy: {-prior_entropy}")
    #
    #         # prior_list.append(-prior_entropy)
    #         total_prior += (-prior_entropy)
    #
    #     # iterations_list = range(1000)
    #     # # Create the plot
    #     # plt.plot(iterations_list, prior_list)
    #     #
    #     # plt.title('Prior Distribution')
    #     # plt.xlabel('Iterations')
    #     # plt.ylabel('Entropy')
    #     #
    #     # plt.grid(True)
    #     # plt.show()
    #
    #     print(f"Mean prior entropy = {total_prior / 1000}")
    #     # print(f"Final state not goal state = {final_state_not_goal_state}")
    #
    #     logger.debug(f"Mean prior entropy = {total_prior / 1000}.")

    # Defining the states and transitions for the
    # DFA here is to simply accept the states that get to the failure state.
    dfa_states = ['q0', 'q1']
    dfa_initial = 'q0'
    dfa_accepting_states = {'q1'}
    dfa_transitions = dict([])

    dfa_transitions = {
        ('q0', 'Fail'): 'q1',
        ('q0', 'Safe'): 'q0',
        ('q1', 'Fail'): 'q1',
        ('q1', 'Safe'): 'q1'
    }

    ################################################ Run the solver ################################################
    labeled_hmm = labeledHMM(agent_mdp=agent_mdp, sensors=sensor_net, no_sensing_act=no_sense_act,
                             goal_policy=goal_policy)

    safety_dfa = SafetyDFA(dfa_states=dfa_states, initial_state=dfa_initial, accepting_states=dfa_accepting_states,
                           transition_function=dfa_transitions)

    product_hmm = ProductHMM(labeled_hmm, safety_dfa)

    # Initialize the solver.
    lstm_solver = GradientDescent_LSTM_solver(prod_hmm=product_hmm, k_step=k_steps, lr=lr)

    # Run a test episode before training and print the trace.
    # Defining the test trace for evaluation.
    # path = [0, 2, 4, 6, 9]
    path = [0, 1, 1, 1, 2]

    print(f"Initial Policy Evaluation:")
    lstm_solver.evaluate_policy(horizon=T, forced_trace=path)

    # Train the LSTM based active perception policy.
    lstm_solver.train(iterations=iter_num, V=V, batch_size=batch_size, horizon=T, alpha=alpha)

    # SAVE MODEL ---
    model_filename = "policy_k1_alpha000_running_example_2_2.pth"
    torch.save(lstm_solver.policy_net.state_dict(), model_filename)
    print(f"Policy saved to {model_filename}")

    # Final Evaluation

    # Run a test episode and print the trace.
    print(f"Final Policy Evaluation:")
    lstm_solver.evaluate_policy(horizon=T, forced_trace=path)

    # Evaluate the trained policy on multiple test episodes and compute the accuracy.
    num_test_episodes = num_test_episodes
    accuracy = lstm_solver.evaluate_policy_accuracy_monte_carlo(episodes=num_test_episodes, horizon=T, k_step=k_steps,
                                                                threshold=0.75)

    # Evaluate the baseline with uniform random policy.
    uniform_random_accuracy = lstm_solver.evaluate_policy_accuracy_monte_carlo(episodes=num_test_episodes, horizon=T,
                                                                               k_step=k_steps, uniform_random=True)

    print(f"Policy Accuracy over {num_test_episodes} episodes: {accuracy:.4f}%")

    print(f"Uniform Random Policy Accuracy over {num_test_episodes} episodes: {uniform_random_accuracy:.4f}%")
