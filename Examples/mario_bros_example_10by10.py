import math

from setup_and_solvers.gridworld_env import *
from setup_and_solvers.LP_for_nominal_policy import *
from setup_and_solvers.test_gradient_entropy_calculations import *
from setup_and_solvers.testing_with_completely_autograd import *

# Initial set-up for a 6x6 gridworld.
ncols = 10
nrows = 10
target = [19, 27, 39]
# target for testing.
# target = [23]

secret_goal_states = [19, 27, 39]
obstacles = [3, 29, 35, 65, 66, 67, 68, 95, 96, 44]
unsafe_u = [8, 16, 17, 25, 37, 69]
non_init_states = [8, 16, 17, 25, 37, 49, 3, 29, 35, 65, 66, 67, 68, 69, 95, 96, 44]
initial = 54
initial_dist = dict([])
# considering a single initial state.
for state in range(100):
    if state == initial:
        initial_dist[state] = 1
    else:
        initial_dist[state] = 0

robot_ts = read_from_file_MDP_old('robotmdp.txt')

# sensor setup
sensors = {'A', 'B', 'C', 'D', 'E', 'F', 'NO'}

# coverage sets
setA = {46, 47, 56, 57}
setB = {26, 27}
setC = {18, 28}
setD = {38, 48, 58}
setE = {39, 49}
setF = {9, 19}
setNO = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 29, 30, 31, 32, 33, 34, 35,
         36, 37, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99}

# masking actions
masking_action = dict([])

masking_action[0] = {'A'}
masking_action[1] = {'B'}
masking_action[2] = {'C'}
masking_action[3] = {'D'}
masking_action[4] = {'E'}
masking_action[5] = {'F'}
masking_action[6] = {'G'}  # 'G' is the no masking action.

# sensor noise
sensor_noise = 0.15

# sensor costs
sensor_cost = dict([])
sensor_cost['A'] = 10
sensor_cost['B'] = 20
sensor_cost['C'] = 10
sensor_cost['D'] = 10
sensor_cost['E'] = 20
sensor_cost['F'] = 20
sensor_cost['G'] = 0  # Cost for not masking.

# Define a threshold for sensor masking.
threshold = 45

sensor_cost_normalization = sum(abs(cost) for cost in sensor_cost.values())

# updating the sensor costs with normalized costs.
for sens in sensor_cost:
    sensor_cost[sens] = sensor_cost[sens] / sensor_cost_normalization

# normalized threshold.
threshold = threshold / sensor_cost_normalization

sensor_net = Sensor()
sensor_net.sensors = sensors

sensor_net.set_coverage('A', setA)
sensor_net.set_coverage('B', setB)
sensor_net.set_coverage('C', setC)
sensor_net.set_coverage('D', setD)
sensor_net.set_coverage('E', setE)
sensor_net.set_coverage('F', setF)
sensor_net.set_coverage('NO', setNO)

sensor_net.jamming_actions = masking_action
sensor_net.sensor_noise = sensor_noise
sensor_net.sensor_cost_dict = sensor_cost

agent_gw_1 = GridworldGui(initial, nrows, ncols, robot_ts, target, obstacles, unsafe_u, initial_dist)
agent_gw_1.mdp.get_supp()
agent_gw_1.mdp.gettrans()
agent_gw_1.mdp.get_reward()
agent_gw_1.draw_state_labels()
goal_policy = LP(agent_gw_1.mdp)

# Computing the prior entropy.
# Monte carlo simulation to obtain the approximate probability of being in the final state in T=10.


# prior_list = list()
# iterations_list = list()
total_prior = 0

for iterations in range(1000):
    prior_entropy = 0
    counter = 0
    horizon = 10
    final_state_goal_state = 0
    final_state_not_goal_state = 0

    while counter <= 1000:
        new_init_state = agent_gw_1.mdp.init

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
    print(f"Probability of reaching goal state within T steps: {probability_of_raching_final_state}")
    prior_entropy = probability_of_raching_final_state * math.log2(probability_of_raching_final_state) + (
            (1 - probability_of_raching_final_state) * math.log2(1 - probability_of_raching_final_state))

    print(f"Prior entropy: {-prior_entropy}")

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

# TODO: The augmented states still consider the gridcells with obstacles. Try by omitting the obstacle filled states
#  -> reduces computation.

hmm_p2 = HiddenMarkovModelP2(agent_gw_1.mdp, sensor_net, goal_policy, secret_goal_states=secret_goal_states)

# masking_policy_gradient = PrimalDualPolicyGradient(hmm=hmm_p2, iter_num=1000, V=10, T=10, eta=1.5, kappa=0.1, epsilon=threshold)
# masking_policy_gradient.solver()

masking_policy_gradient = PrimalDualPolicyGradientTest(hmm=hmm_p2, iter_num=1000, batch_size=100, V=100, T=10,
                                                       eta=2.7,
                                                       kappa=0.225,
                                                       epsilon=threshold)

masking_policy_gradient.solver()