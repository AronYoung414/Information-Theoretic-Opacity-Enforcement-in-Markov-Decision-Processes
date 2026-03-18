import math

from setup_and_solvers.gridworld_env_multi_init_states import *
from setup_and_solvers.LP_for_nominal_policy import *
from setup_and_solvers.test_gradient_entropy_calculations_for_predictive_safety import *


def run_gridworld_example_predictive_safety(iter_num=1000, batch_size=100, V=100, T=3, lr=0.005, k_steps=1, alpha=0,
                                      prior_compute_flag=0):
    logger.add(
        "logs_for_examples/log_file_gridworld_example_predictive-safety.log")

    logger.info("This is the log file for the 6X6 gridworld version 1.")

    # Initial set-up for a 6x6 gridworld.
    ncols = 6
    nrows = 6
    target = [35]
    # target for testing.
    # target = [23]

    failure_states = [5, 7, 14, 16, 23, 25, 28, 29]
    obstacles = []
    unsafe_u = [5, 7, 14, 16, 23, 25, 28, 29]
    non_init_states = [1, 25, 9, 14, 15, 17, 19, 23, 35]
    # initial = {30, 24, 18, 12, 6, 0}
    initial = {0}
    # initial = {30}

    initial_dist = dict([])
    # considering a single initial state.
    for state in range(36):
        if state in initial:
            initial_dist[state] = 1 / len(initial)
        else:
            initial_dist[state] = 0

    # Labeling function for the states.
    labels = dict([])
    for state in range(36):
        if state in failure_states:
            labels[state] = {'Fail'}
        else:
            labels[state] = {'Safe'}

    robot_ts = read_from_file_MDP_old('robotmdp.txt')

    # sensor setup
    # sensors = {'A', 'B', 'C', 'D', 'E', 'NO'}
    sensors = {'A', 'B', 'C', 'D', 'NO'}

    # coverage sets
    # setA = {3, 4, 9, 10}
    # setB = {21, 22, 27, 28, 33, 34}
    # setC = {23, 29, 35}
    # setD = {6, 7, 8, 12, 13, 14}
    # setE = {5, 11}
    # setNO = {0, 1, 2, 15, 16, 17, 18, 19, 20, 24, 25, 26, 30, 31, 32}

    setA = {33, 34, 27, 28, 21, 22}
    setB = {18, 19, 12, 13}
    setC = {10, 4, 9}
    setD = {0, 1, 2, 3, 4, 5}
    # setE = {20}
    setNO = {6, 7, 8, 9, 11, 14, 15, 16, 17, 20, 23, 24, 25, 26, 29, 30, 31, 32, 35}

    # masking actions
    sensing_action = dict([])

    sensing_action[0] = {'A'}
    sensing_action[1] = {'B'}
    sensing_action[2] = {'C'}
    sensing_action[3] = {'D'}
    # masking_action[4] = {'E'}
    sensing_action[4] = {'NO'}  # 'NO' is the no masking action.

    no_sense_act = 4

    # sensor noise
    sensor_noise = 0.15

    # sensor costs
    sensor_cost = dict([])
    sensor_cost['A'] = 10
    sensor_cost['B'] = 10
    sensor_cost['C'] = 10
    sensor_cost['D'] = 10
    # sensor_cost['E'] = 25
    sensor_cost['NO'] = 0  # Cost for not masking.



    sensor_net = Sensor()
    sensor_net.sensors = sensors

    sensor_net.set_coverage('A', setA)
    sensor_net.set_coverage('B', setB)
    sensor_net.set_coverage('C', setC)
    sensor_net.set_coverage('D', setD)
    # sensor_net.set_coverage('E', setE)
    sensor_net.set_coverage('NO', setNO)

    sensor_net.sensing_actions = sensing_action
    sensor_net.sensor_noise = sensor_noise
    sensor_net.sensor_cost_dict = sensor_cost

    agent_gw_1 = GridworldGui(initial, nrows, ncols, robot_ts, target, obstacles, unsafe_u, initial_dist)
    agent_gw_1.mdp.get_supp()
    agent_gw_1.mdp.gettrans()
    agent_gw_1.mdp.get_reward()
    agent_gw_1.draw_state_labels()

    # # Obtain the goal policy.
    # goal_policy = LP(mdp=agent_gw_1.mdp, gamma=0.9)
    #
    # logger.debug("Goal policy:")
    # logger.debug(goal_policy)
    #
    # with open('goal_policy.pickle', 'wb') as file:
    #     pickle.dump(goal_policy, file)

    # goal_policy_file = "goal_policy.pickle"

    # Load from goal policy.
    # with open(goal_policy_file, "rb") as f:
    #     goal_policy = pickle.load(f)

    goal_policy = {
        # --- ROW 0 (Bottom) ---
        # Safe path: (0,0)->(1,0)->(2,0)->(3,0) then North
        (0, 'E'): 0.5, (0, 'N'): 0.5, (0, 'W'): 0.0, (0, 'S'): 0.0,
        (1, 'E'): 1.0, (1, 'N'): 0.0, (1, 'W'): 0.0, (1, 'S'): 0.0,
        (2, 'E'): 0.5, (2, 'N'): 0.5, (2, 'W'): 0.0, (2, 'S'): 0.0,
        (3, 'N'): 1.0, (3, 'E'): 0.0, (3, 'W'): 0.0, (3, 'S'): 0.0,  # Turn North at 3
        (4, 'N'): 1.0, (4, 'E'): 0.0, (4, 'W'): 0.0, (4, 'S'): 0.0,  # Just in case
        (5, 'W'): 1.0, (5, 'N'): 0.0, (5, 'E'): 0.0, (5, 'S'): 0.0,  # Trap at 5! Move away

        # --- ROW 1 ---
        # Trap at 7 (1,1). Avoid.
        (6, 'E'): 1.0, (6, 'N'): 0.0, (6, 'W'): 0.0, (6, 'S'): 0.0,
        (7, 'E'): 1.0, (7, 'N'): 0.0, (7, 'W'): 0.0, (7, 'S'): 0.0,  # In Trap, try escape
        (8, 'E'): 1.0, (8, 'N'): 0.0, (8, 'W'): 0.0, (8, 'S'): 0.0,
        (9, 'N'): 1.0, (9, 'E'): 0.0, (9, 'W'): 0.0, (9, 'S'): 0.0,  # Path Up
        (10, 'N'): 1.0, (10, 'E'): 0.0, (10, 'W'): 0.0, (10, 'S'): 0.0,
        (11, 'W'): 1.0, (11, 'N'): 0.0, (11, 'E'): 0.0, (11, 'S'): 0.0,

        # --- ROW 2 ---
        # Trap at 14 (2,2) and 16 (2,4). Path is tight.
        (12, 'E'): 1.0, (12, 'N'): 0.0, (12, 'W'): 0.0, (12, 'S'): 0.0,
        (13, 'S'): 1.0, (13, 'N'): 0.0, (13, 'E'): 0.0, (13, 'W'): 0.0,  # Near trap 14, go back S
        (14, 'E'): 1.0, (14, 'N'): 0.0, (14, 'W'): 0.0, (14, 'S'): 0.0,  # In Trap
        (15, 'N'): 1.0, (15, 'E'): 0.0, (15, 'W'): 0.0, (15, 'S'): 0.0,  # Path Up
        (16, 'W'): 1.0, (16, 'N'): 0.0, (16, 'E'): 0.0, (16, 'S'): 0.0,  # In Trap
        (17, 'W'): 1.0, (17, 'N'): 0.0, (17, 'E'): 0.0, (17, 'S'): 0.0,

        # --- ROW 3 ---
        # Safe zone mostly. Aim for col 5 entry.
        (18, 'E'): 1.0, (18, 'N'): 0.0, (18, 'W'): 0.0, (18, 'S'): 0.0,
        (19, 'E'): 1.0, (19, 'N'): 0.0, (19, 'W'): 0.0, (19, 'S'): 0.0,
        (20, 'E'): 1.0, (20, 'N'): 0.0, (20, 'W'): 0.0, (20, 'S'): 0.0,
        (21, 'E'): 1.0, (21, 'N'): 0.0, (21, 'W'): 0.0, (21, 'S'): 0.0,  # Move to right edge
        (22, 'N'): 1.0, (22, 'E'): 0.0, (22, 'W'): 0.0, (22, 'S'): 0.0,
        (23, 'W'): 1.0, (23, 'N'): 0.0, (23, 'E'): 0.0, (23, 'S'): 0.0,  # Trap at 23(3,5)!

        # --- ROW 4 ---
        # Traps at 25(4,1), 28(4,4). Goal is blocked from 29(4,5).
        # Must enter goal from 34(5,4).
        (24, 'E'): 1.0, (24, 'N'): 0.0, (24, 'W'): 0.0, (24, 'S'): 0.0,
        (25, 'E'): 1.0, (25, 'N'): 0.0, (25, 'W'): 0.0, (25, 'S'): 0.0,  # In Trap
        (26, 'E'): 1.0, (26, 'N'): 0.0, (26, 'W'): 0.0, (26, 'S'): 0.0,
        (27, 'N'): 1.0, (27, 'E'): 0.0, (27, 'W'): 0.0, (27, 'S'): 0.0,  # Go up to 33
        (28, 'S'): 1.0, (28, 'N'): 0.0, (28, 'E'): 0.0, (28, 'W'): 0.0,  # In Trap 28
        (29, 'S'): 1.0, (29, 'N'): 0.0, (29, 'E'): 0.0, (29, 'W'): 0.0,  # In Trap 29

        # --- ROW 5 (Top) ---
        # Goal is 35. Access from 34.
        (30, 'E'): 1.0, (30, 'N'): 0.0, (30, 'W'): 0.0, (30, 'S'): 0.0,
        (31, 'E'): 1.0, (31, 'N'): 0.0, (31, 'W'): 0.0, (31, 'S'): 0.0,
        (32, 'E'): 1.0, (32, 'N'): 0.0, (32, 'W'): 0.0, (32, 'S'): 0.0,
        (33, 'E'): 1.0, (33, 'N'): 0.0, (33, 'W'): 0.0, (33, 'S'): 0.0,  # Move to 34
        (34, 'N'): 1.0, (34, 'E'): 0.0, (34, 'W'): 0.0, (34, 'S'): 0.0,  # ENTER GOAL
        (35, 'N'): 0.0, (35, 'E'): 0.0, (35, 'W'): 0.0, (35, 'S'): 0.0,  # Goal (Stop)
    }

    # prior_compute_flag = 0

    if prior_compute_flag == 1:

        # Computing the prior entropy.
        # Monte carlo simulation to obtain the approximate probability of being in the final state in T=10.

        # prior_list = list()
        # iterations_list = list()
        total_prior = 0

        for iterations in range(1000):
            prior_entropy = 0
            counter = 0
            horizon = 12
            final_state_goal_state = 0
            final_state_not_goal_state = 0

            while counter <= 1000:
                new_init_state = random.choice(list(agent_gw_1.mdp.init))

                for i in range(horizon):
                    weights_list = list()
                    for action in agent_gw_1.mdp.actlist:
                        weights_list.append(goal_policy[(new_init_state, action)])
                    action_to_play = random.choices(agent_gw_1.actlist, weights_list)[0]

                    post_states = list(agent_gw_1.mdp.suppDict[(new_init_state, action_to_play)])
                    states_weights_list = list()
                    for st in post_states:
                        states_weights_list.append(agent_gw_1.mdp.trans[new_init_state][action_to_play][st])

                    next_state = random.choices(post_states, states_weights_list)[0]
                    new_init_state = next_state

                if new_init_state in target:
                    final_state_goal_state += 1
                else:
                    final_state_not_goal_state += 1

                counter += 1

            probability_of_raching_final_state = final_state_goal_state / 1001
            # print(f"Probability of reaching goal state within T steps: {probability_of_raching_final_state}")
            prior_entropy = probability_of_raching_final_state * math.log2(probability_of_raching_final_state) + (
                    (1 - probability_of_raching_final_state) * math.log2(1 - probability_of_raching_final_state))

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

    hmm_p2 = HiddenMarkovModelP2(agent_gw_1.mdp, sensor_net, goal_policy, secret_goal_states=secret_goal_states,
                                 no_mask_act=no_mask_act)

    # masking_policy_gradient = PrimalDualPolicyGradient(hmm=hmm_p2, iter_num=1000, V=10, T=10, eta=1.5, kappa=0.1, epsilon=threshold)
    # masking_policy_gradient.solver()

    # masking_policy_gradient = PrimalDualPolicyGradientTest(hmm=hmm_p2, iter_num=3000, batch_size=100, V=100, T=10,
    #                                                        eta=3.2,
    #                                                        kappa=0.25,
    #                                                        epsilon=threshold)

    # masking_policy_gradient = PrimalDualPolicyGradientTest(hmm=hmm_p2, iter_num=3000, batch_size=100, V=150, T=10,
    #                                                        eta=8.2,
    #                                                        kappa=0.25,
    #                                                        epsilon=threshold)

    masking_policy_gradient = PrimalDualPolicyGradientTest(hmm=hmm_p2, iter_num=iter_num, batch_size=batch_size, V=V,
                                                           T=T,
                                                           eta=eta,
                                                           kappa=kappa,
                                                           epsilon=threshold,
                                                           sensor_cost_normalization=sensor_cost_normalization,
                                                           exp_number=exp_number)

    masking_policy_gradient.solver()
