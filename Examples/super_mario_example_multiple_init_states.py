import math

from setup_and_solvers.gridworld_env_multi_init_states import *
from setup_and_solvers.LP_for_nominal_policy import *
from setup_and_solvers.test_gradient_entropy_calculations import *

logger.add("logs_for_examples/log_file_mario_example_information_theoretic_opacity.log")

logger.info("This is the log file for the 6X6 gridworld with goal states 9, 20, 23 test case.")

# Initial set-up for a 6x6 gridworld.
ncols = 6
nrows = 6
target = [9, 20, 23]
# target for testing.
# target = [23]

secret_goal_states = [9, 20, 23]
reward_states = [11, 22]

obstacles = [17, 19]
unsafe_u = [1, 15, 25, 35, 21]
non_init_states = [1, 25, 9, 14, 15, 17, 19, 23, 35]
initial = {30, 24, 18, 12, 6, 0}
# initial = {30, 12}
initial_dist = dict([])
# considering a single initial state.
for state in range(36):
    if state in initial:
        initial_dist[state] = 1/len(initial)
    else:
        initial_dist[state] = 0

robot_ts = read_from_file_MDP_old('robotmdp.txt')

# sensor setup
sensors = {'A', 'B', 'C', 'D', 'E', 'NO'}


setA = {3, 4, 9, 10}
setB = {21, 22, 27, 28, 33, 34}
setC = {23, 29, 35}
setD = {6, 7, 8, 12, 13, 14}
setE = {20}
setNO = {0, 1, 2, 15, 16, 17, 18, 19, 24, 25, 30, 31, 32, 5, 11, 26}



# sensor noise
sensor_noise = 0.15



sensor_net = Sensor()
sensor_net.sensors = sensors

sensor_net.set_coverage('A', setA)
sensor_net.set_coverage('B', setB)
sensor_net.set_coverage('C', setC)
sensor_net.set_coverage('D', setD)
sensor_net.set_coverage('E', setE)
sensor_net.set_coverage('NO', setNO)

# sensor_net.jamming_actions = masking_action
sensor_net.sensor_noise = sensor_noise
# sensor_net.sensor_cost_dict = sensor_cost

agent_gw_1 = GridworldGui(initial, nrows, ncols, robot_ts, target, obstacles, unsafe_u, initial_dist)
agent_gw_1.mdp.get_supp()
agent_gw_1.mdp.gettrans()
agent_gw_1.mdp.get_reward()
agent_gw_1.draw_state_labels()

# reward/ value matrix for the agent.
value_dict = dict()
for state in agent_gw_1.mdp.states:
    if state in reward_states:
        value_dict[state] = 1
    else:
        value_dict[state] = 0

# goal_policy = LP(mdp=agent_gw_1.mdp, gamma=0.99)

# logger.debug("Goal policy:")
# logger.debug(goal_policy)

# prior_compute_flag = 0
#
# if prior_compute_flag == 1:
#
#     # Computing the prior entropy.
#     # Monte carlo simulation to obtain the approximate probability of being in the final state in T=10.
#
#
#     # prior_list = list()
#     # iterations_list = list()
#     total_prior = 0
#
#     for iterations in range(1000):
#         prior_entropy = 0
#         counter = 0
#         horizon = 12
#         final_state_goal_state = 0
#         final_state_not_goal_state = 0
#
#         while counter <= 1000:
#             new_init_state = random.choice(list(agent_gw_1.mdp.init))
#
#             for i in range(horizon):
#                 weights_list = list()
#                 for action in agent_gw_1.mdp.actlist:
#                     weights_list.append(goal_policy[(new_init_state, action)])
#                 action_to_play = random.choices(agent_gw_1.actlist, weights_list)[0]
#
#                 post_states = list(agent_gw_1.mdp.suppDict[(new_init_state, action_to_play)])
#                 states_weights_list = list()
#                 for st in post_states:
#                     states_weights_list.append(agent_gw_1.mdp.trans[new_init_state][action_to_play][st])
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
#         probability_of_raching_final_state = final_state_goal_state/1001
#         # print(f"Probability of reaching goal state within T steps: {probability_of_raching_final_state}")
#         prior_entropy = probability_of_raching_final_state * math.log2(probability_of_raching_final_state) + ((1-probability_of_raching_final_state) * math.log2(1-probability_of_raching_final_state))
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
#
#     print(f"Mean prior entropy = {total_prior/1000}")
#     # print(f"Final state not goal state = {final_state_not_goal_state}")
#
#     logger.debug(f"Mean prior entropy = {total_prior/1000}.")

# TODO: The augmented states still consider the gridcells with obstacles. Try by omitting the obstacle filled states
#  -> reduces computation.

hmm_p2 = HiddenMarkovModelP2(agent_gw_1.mdp, sensor_net, value_dict=value_dict, secret_goal_states=secret_goal_states)

# masking_policy_gradient = PrimalDualPolicyGradient(hmm=hmm_p2, iter_num=1000, V=10, T=10, eta=1.5, kappa=0.1, epsilon=threshold)
# masking_policy_gradient.solver()

masking_policy_gradient = PrimalDualPolicyGradientTest(hmm=hmm_p2, iter_num=3000, batch_size=100, V=1000, T=12,
                                                       eta=3.2,
                                                       kappa=0.25,
                                                       epsilon=0.3)

masking_policy_gradient.solver()
