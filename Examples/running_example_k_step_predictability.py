import math

from setup_and_solvers.LP_for_nominal_policy import *
from setup_and_solvers.test_gradient_entropy_calculations_modified_obs_vectorized_code import *
from setup_and_solvers.markov_decision_process import *


def running_example(iter_num=1000, batch_size=100, V=100, T=3, eta=8.2, kappa=3.1, threshold=20, prior_compute_flag=0,
                    exp_number=2):
    # kappa value that worked for thereshold 30 -> 3.1 for threshold 60  -> 4.1
    logger.add("logs_for_examples/log_file_for_running_example.log")

    logger.info("This is the log file for the running example with secret goal states s4 and s6 and planning "
                "gamma=0.99.")

    # Initial set-up for the MDP.

    states = [0, 1, 2, 3, 4, 5]
    actions = ['a']

    prob = {
        'a': np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0.8, 0, 0, 0.2], [0, 0, 0, 0.8, 0, 0.2],
                       [0, 0, 0, 0, 0.8, 0.2], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    }

    transitions = {
        (0, 'a'): [(1, 1)],
        (1, 'a'): [(2, 0.8), (5, 0.2)],
        (2, 'a'): [(3, 0.8), (5, 0.2)],
        (3, 'a'): [(4, 0.8), (5, 0.2)],
        (4, 'a'): [(4, 1)],
        (5, 'a'): [(5, 1)]
    }

    target = [4]
    secret_goal_states = [4]

    initial = [0]

    initial_dist = dict([])
    # Initial distribution is uniform over the initial states. This is a simplification for the current example
    for state in states:
        if state in initial:
            initial_dist[state] = 1 / len(initial)
        else:
            initial_dist[state] = 0

    # sensor setup
    sensors = {'R', 'G', 'P', 'B', 'NO'}

    # coverage sets
    setR = {1}
    setG = {2}
    setP = {3}
    setB = {4}
    setNO = {0, 5}

    # masking actions
    masking_action = dict([])

    masking_action[0] = {'R'}
    masking_action[1] = {'G'}
    masking_action[2] = {'P'}
    masking_action[3] = {'B'}
    # masking_action[4] = {'E'}
    masking_action[4] = {'F'}  # 'F' is the no masking action.

    no_mask_act = 4

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
    sensor_cost['F'] = 0  # Cost for not masking.

    # if threshold == 60:
    #     # eta = 3.2
    #     eta = 3.2
    #     kappa = 0.25

    # Define a threshold for sensor masking.
    # threshold = 60
    # threshold = 20

    sensor_cost_normalization = sum(abs(cost) for cost in sensor_cost.values())

    # updating the sensor costs with normalized costs.
    for sens in sensor_cost:
        sensor_cost[sens] = sensor_cost[sens] / sensor_cost_normalization

    # normalized threshold.
    threshold = threshold / sensor_cost_normalization

    sensor_net = Sensor()
    sensor_net.sensors = sensors
    sensor_net.set_coverage('R', setR)
    sensor_net.set_coverage('G', setG)
    sensor_net.set_coverage('P', setP)
    sensor_net.set_coverage('B', setB)
    sensor_net.set_coverage('NO', setNO)

    sensor_net.jamming_actions = masking_action
    sensor_net.sensor_noise = sensor_noise
    sensor_net.sensor_cost_dict = sensor_cost

    agent_mdp = MDP(init=initial, actlist=actions, states=states, prob=prob, trans=transitions, init_dist=initial_dist,
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

        # Computing the prior entropy.
        # Monte carlo simulation to obtain the approximate probability of being in the final state in T=10.

        # prior_list = list()
        # iterations_list = list()
        total_prior = 0

        for iterations in range(1000):
            prior_entropy = 0
            counter = 0
            horizon = 2
            final_state_goal_state = 0
            final_state_not_goal_state = 0

            while counter <= 1000:
                new_init_state = random.choice(list(agent_mdp.init))

                for i in range(horizon):
                    weights_list = list()
                    for action in agent_mdp.actlist:
                        weights_list.append(goal_policy[(new_init_state, action)])
                    action_to_play = random.choices(agent_mdp.actlist, weights_list)[0]

                    post_states = list(agent_mdp.suppDict[(new_init_state, action_to_play)])
                    states_weights_list = list()
                    for st in post_states:
                        states_weights_list.append(agent_mdp.trans[new_init_state][action_to_play][st])

                    next_state = random.choices(post_states, states_weights_list)[0]
                    new_init_state = next_state

                if new_init_state in target:
                    final_state_goal_state += 1
                else:
                    final_state_not_goal_state += 1

                counter += 1

            probability_of_raching_final_state = final_state_goal_state / 1001
            # print(f"Probability of reaching goal state within T steps: {probability_of_raching_final_state}")
            # prior_entropy = probability_of_raching_final_state * math.log2(probability_of_raching_final_state) + (
            #             (1 - probability_of_raching_final_state) * math.log2(1 - probability_of_raching_final_state))

            if probability_of_raching_final_state == 0:
                prior_entropy_part_1 = 0
            else:
                prior_entropy_part_1 = probability_of_raching_final_state * math.log2(
                    probability_of_raching_final_state)

            if (1 - probability_of_raching_final_state) == 0:
                prior_entropy_part_2 = 0
            else:
                prior_entropy_part_2 = (1 - probability_of_raching_final_state) * math.log2(
                    1 - probability_of_raching_final_state)

            prior_entropy = prior_entropy_part_1 + prior_entropy_part_2

            # print(f"Prior entropy: {-prior_entropy}")

            # prior_list.append(-prior_entropy)
            total_prior += (-prior_entropy)

        # iterations_list = range(1000)
        # # Create the plot
        # plt.plot(iterations_list, prior_list)
        #
        # plt.title('Prior Distribution')
        # plt.xlabel('Iterations')
        # plt.ylabel('Entropy')
        #
        # plt.grid(True)
        # plt.show()

        print(f"Mean prior entropy = {total_prior / 1000}")
        # print(f"Final state not goal state = {final_state_not_goal_state}")

        logger.debug(f"Mean prior entropy = {total_prior / 1000}.")

    hmm_p2 = HiddenMarkovModelP2(agent_mdp, sensor_net, goal_policy, secret_goal_states=secret_goal_states,
                                 no_mask_act=no_mask_act)

    # masking_policy_gradient = PrimalDualPolicyGradient(hmm=hmm_p2, iter_num=1000, V=10, T=10, eta=1.5, kappa=0.1,
    # epsilon=threshold) masking_policy_gradient.solver()

    # masking_policy_gradient = PrimalDualPolicyGradientTest(hmm=hmm_p2, iter_num=1000, batch_size=100, V=100, T=3,
    # eta=8.2, kappa=0.25, epsilon=threshold, sensor_cost_normalization=sensor_cost_normalization)

    # masking_policy_gradient = PrimalDualPolicyGradientTest(hmm=hmm_p2, iter_num=1000, batch_size=100, V=100, T=3,
    #                                                        eta=8.2,
    #                                                        kappa=4.1,
    #                                                        epsilon=threshold,
    #                                                        sensor_cost_normalization=sensor_cost_normalization)

    masking_policy_gradient = PrimalDualPolicyGradientOptimized(hmm=hmm_p2, iter_num=iter_num, batch_size=batch_size,
                                                                V=V,
                                                                T=T,
                                                                eta=eta,
                                                                kappa=kappa,
                                                                epsilon=threshold,
                                                                sensor_cost_normalization=sensor_cost_normalization,
                                                                exp_number=exp_number)

    masking_policy_gradient.solver()
