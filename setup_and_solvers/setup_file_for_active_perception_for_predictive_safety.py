from setup_and_solvers.markov_decision_process import *
from Examples.sensors import *
import itertools
from collections import defaultdict
import random
import pickle


class labeledHMM:
    """
    Wraps the MDP and Sensor classes to create a labeled Hidden Markov Model (HMM) for the robot in the environment
    as seen by the active perception agent.
    """

    def __init__(self, agent_mdp, sensors, no_sensing_act=None, fp_noise=0, goal_policy=dict([])):
        if not isinstance(agent_mdp, MDP):
            raise TypeError("Expected agent_mdp to be an instance of MDP.")

        if not isinstance(sensors, Sensor):
            raise TypeError("Expected sensors to be an instance of Sensor.")

        self.agent_mdp = agent_mdp
        self.sensors = sensors
        self.goal_policy = goal_policy  # The policy to the goal, that is known to P2. It is of the format
        # goal_policy[(s,a)]=probability
        self.sensing_acts = self.sensors.sensing_actions  # Masking actions available to P1
        self.sensing_act_indx_dict = dict()
        indx_num = 0
        for sense_query in self.sensing_acts:
            self.sensing_act_indx_dict[sense_query] = indx_num
            indx_num += 1

        self.states_indx_dict = dict()
        self.indx_to_states_dict = dict()
        indx_num = 0
        for state in self.agent_mdp.states:
            self.states_indx_dict[state] = indx_num
            self.indx_to_states_dict[indx_num] = state
            indx_num += 1

        self.labels = self.agent_mdp.labels  # The labeling function for the states. labels[state]=labels
        self.state_indx_labels_dict = dict()
        for state in self.agent_mdp.states:
            self.state_indx_labels_dict[self.states_indx_dict[state]] = self.labels[state]

        self.no_sensing_act = no_sensing_act

        self.transition_dict = defaultdict(
            lambda: defaultdict(dict))  # The transition dictionary for the augmented state-space.
        self.get_transition_dict()

        # self.transition_mat = {a: np.zeros((len(self.augmented_states), len(self.augmented_states))) for a in
        # self.masking_acts}
        self.transition_mat = np.zeros((len(self.agent_mdp.states), len(self.agent_mdp.states)))
        self.get_transition_mat()

        self.state_obs2 = defaultdict(
            lambda: defaultdict(dict))  # state_obs2 is the state observation dict. state_obs2[state][query]={sensors
        # that cover state and are queried}
        self.get_state_obs2()

        # set of all observations.
        # As we consider the masking action to be visible to the observer, the observation space is A X Sensors.

        ## Uncomment the following lines to have the observation space as (sensor observation, masking action), ie, with masking action observable.
        ## The following is for when we have one dynamic agent in the environment.

        # observations = self.sensors.sensors
        # observations.add('0')
        # print("Observations before removing 'NO':", observations)
        # observations.remove('NO')  # As 'Null' = 'No'. If not, comment this. This is for non-multiagent implementation.
        # self.observations = set()
        # # for sense_act, st_obs in itertools.product(self.sensing_acts.keys(), observations):
        # #     self.observations.add((sense_act, st_obs))
        #
        # for sense_act in self.sensing_acts.keys():
        #     obs_sense = list(self.sensing_acts[sense_act])
        #     self.observations.add((obs_sense[0], sense_act))
        #     self.observations.add(('0', sense_act))
        #
        # ## The following is for the consideration of masking action being unobservable.
        #
        # # self.observations = set(
        # #     self.sensors.sensors)  # TODO: This needs to be changed for generalization if the observation function
        # # # TODO: changes!!!
        # # self.observations.add('0')  # '0' represents the null observation.
        # # self.observations.remove('NO')  # As 'Null' = 'No'. If not, comment this.
        #
        # self.observations_indx_dict = dict()  # Defining a dictionary with [obs]=indx
        # indx_num = 0
        # for ob in self.observations:
        #     self.observations_indx_dict[ob] = indx_num
        #     indx_num += 1
        #
        # self.observations_indx_to_obs_dict = dict()  # Defining a dictionary with [indx]=obs
        # for ob in self.observations:
        #     self.observations_indx_to_obs_dict[self.observations_indx_dict[ob]] = ob

        # --- Observations setup for two dynamic agents in the environment. ---
        self.observations = set()

        for sense_act in self.sensing_acts.keys():
            # Get the logical sensors (e.g., ['C1_E', 'C1_T']) and sort them alphabetically
            obs_sense = sorted(list(self.sensing_acts[sense_act]))

            # Filter out the blind spot 'NO' sensors from becoming active tokens
            obs_sense = [s for s in obs_sense if s not in ['NO_E', 'NO_T']]

            if len(obs_sense) > 0:
                # Add the individual tokens (e.g., 'C1_E', 'C1_T')
                for s in obs_sense:
                    self.observations.add((s, sense_act))

                # Add the JOINED string for when both are in frame! (e.g., 'C1_E_C1_T')
                if len(obs_sense) > 1:
                    joined_obs = "_".join(obs_sense)
                    self.observations.add((joined_obs, sense_act))

            # Add the null token for this action
            self.observations.add(('0', sense_act))

        self.observations_indx_dict = dict()
        indx_num = 0
        for ob in self.observations:
            self.observations_indx_dict[ob] = indx_num
            indx_num += 1

        self.observations_indx_to_obs_dict = dict()
        for ob in self.observations:
            self.observations_indx_to_obs_dict[self.observations_indx_dict[ob]] = ob

        with open('observation_indx_dict.pkl', 'wb') as f:
            pickle.dump(self.observations_indx_dict, f)
        print("Observation index dictionary saved to 'observation_indx_dict.pkl' for use in GAMMS.")

        self.obs_noise = self.sensors.sensor_noise
        # self.fp_noise = fp_noise

        self.emission_prob = defaultdict(
            lambda: defaultdict(dict))  # The emission probability for the observations. emission_prob[
        # state][query][obs]=probability
        self.get_emission_prob()

        self.cost_dict = defaultdict(
            lambda: defaultdict(dict))  # The format is [sensor_query_prev][sensor_query_curr]=value.
        self.sensor_cost_dict = self.sensors.sensor_cost_dict
        self.total_sensor_cost = dict([])  # The format is total_sensor_cost[mask]=cost
        self.get_total_sensor_cost()
        self.get_cost_dict()  # Cost for each masking action taken.

        self.initial_dist = dict(
            [])  # The initial distribution of the augmented state-space. initial_dist[augstate]=probability
        # initial distribution array.
        self.mu_0 = np.zeros(len(self.agent_mdp.states))

        self.get_initial_dist()

        # # sampled initial state
        # self.initial_state = None
        # self.get_initial_state()

        # set of initial states.
        self.initial_states = set()
        self.get_initial_states()

        # self.secret_goal_states = list([])
        # self.get_secret_goal_states(secret_goal_states)

    # def get_secret_goal_states(self, secret_goal_states):
    #     # Construct a list of augmented secret states.
    #     for state in self.augmented_states:
    #         if state[0] in secret_goal_states:
    #             self.secret_goal_states.append(state)
    #     return

    # def get_augmented_states(self):
    #     for state, mask in itertools.product(self.agent_mdp.states, self.masking_acts):
    #         self.augmented_states.append((state, mask))
    #     return

    # def get_transition_dict(self): # The transition dict is the transition function such that transition_dict[
    # state][mask][next_state]=probability. for state, mask in itertools.product(self.augmented_states,
    # self.masking_acts): # TODO: Consider only the post states to populate the trans dict.

    def get_transition_dict(self):
        # The transition_dict is the transition function such that transition_dict[state][mask][next_state]=probability.
        for state, next_state in itertools.product(self.agent_mdp.states, self.agent_mdp.states):
            self.transition_dict[state][next_state] = self.get_transition_probability(state, next_state)

        return

    def get_transition_mat(self):
        # The matrix representation of the transition function. transition_mat[i, j] = probability.
        for state, next_state in itertools.product(self.agent_mdp.states, self.agent_mdp.states):
            self.transition_mat[self.states_indx_dict[state], self.states_indx_dict[next_state]] = \
                self.transition_dict[state][next_state]
        return

    def get_transition_probability(self, state, next_state):
        # This is computed as \sum_{a\in A} P(s,a,s') \pi_G(a\mid s)
        probability = 0.0
        for a in self.agent_mdp.actlist:
            probability = probability + (self.agent_mdp.P(state, a, next_state) * self.goal_policy[(state, a)])
        return float(probability)

    def get_emission_prob(self):
        # In the emission function for each state, and observation pairs.
        for state, query in itertools.product(self.agent_mdp.states, self.sensing_acts):
            for obs in self.observations:
                self.emission_prob[state][query][obs] = self.get_emission_probability(state, query, obs)
        return

    # # The following is the emission probability for when the action is visible along with the observation.
    # def get_emission_probability(self, state, query,
    #                              obs):  # TODO: Change this function for active perception.
    #     if len(self.state_obs2[state][query]) == 0:
    #         if obs[0] == '0' and obs[1] == query:
    #             return 1
    #         else:
    #             return 0
    #     else:
    #
    #         # True observation with P(obs) and 'null' observation with P(obs_noise).
    #         # With special consideration to No Sensor. States under 'NO' always return 'NO' --TODO: Check??
    #         if state in self.sensors.coverage['NO']:
    #             # Use the following when 'NO' and 'Null' are the same.
    #             if obs[0] == '0' and obs[1] == query:
    #                 return 1
    #             else:
    #                 return 0
    #             # Use the following when 'NO' is different from 'Null'.
    #             # if obs in self.state_obs2[state]:
    #             #     return 1
    #             # else:
    #             #     return 0
    #
    #         else:
    #             if obs[0] == '0' and obs[1] == query:  # as we have '0' to be the null observation.
    #                 return self.obs_noise
    #             elif obs[0] in self.state_obs2[state][query] and obs[1] == query:
    #                 return 1 - self.obs_noise
    #             else:
    #                 return 0

    ## The following is the emission probability function for when there are two dynamic agents in the environment.
    # --- 3. EMISSION PROBABILITY UPDATE ---
    def get_emission_probability(self, state, query, obs):

        # #The following is only for when we have False Positive enabled.
        # # 1. Identify the valid false positive strings for THIS specific camera
        # base_senses = [s for s in self.sensing_acts.get(query, []) if s not in ['NO_E', 'NO_T', 'NO']]
        # valid_fps = set(base_senses)
        # if len(base_senses) > 1:
        #     # Add the joint string (e.g., 'C1_E_C1_T')
        #     valid_fps.add("_".join(sorted(base_senses)))
        #
        # fp_prob_split = self.fp_noise / max(1, len(valid_fps))

        if len(self.state_obs2[state][query]) == 0:
            if obs[0] == '0' and obs[1] == query:
                return 1
            else:
                return 0
        else:
            # Special consideration for the exact physical blind spots
            if state in self.sensors.coverage['NO_E'] and state in self.sensors.coverage['NO_T']:
                if obs[0] == '0' and obs[1] == query:
                    return 1
                else:
                    return 0
            # # Use the following when we have False Positives enabled.
            # if state in self.sensors.coverage['NO_E'] and state not in self.sensors.coverage['NO_T']:
            #     if obs[0] == '0' and obs[1] == query:
            #         return 1 - self.fp_noise
            #     elif obs[0] in valid_fps and obs[1] == query:
            #         return fp_prob_split
            #     else:
            #         return 0
            else:
                if obs[0] == '0' and obs[1] == query:
                    return self.obs_noise
                # This line now safely handles 'C1_E', 'C1_T', AND 'C1_E_C1_T'
                elif obs[0] in self.state_obs2[state][query] and obs[1] == query:
                    return 1 - self.obs_noise
                else:
                    return 0

    def get_cost_dict(self):  # TODO: Rewrite the cost function for active perception.
        # Assign cost/reward/value for each of the sensor that is queried. If the query is the same as the previous
        # query, we reduce the cost (as it is similar to having simply the maintenance cost.)
        for query_prev, query_curr in itertools.product(self.sensing_acts, self.sensing_acts):
            if query_prev == query_curr:
                self.cost_dict[self.sensing_act_indx_dict[query_prev]][self.sensing_act_indx_dict[query_curr]] = 0.5 * \
                                                                                                                 self.total_sensor_cost[
                                                                                                                     query_curr]
            else:
                self.cost_dict[self.sensing_act_indx_dict[query_prev]][self.sensing_act_indx_dict[query_curr]] = \
                    self.total_sensor_cost[query_curr]
        return

    def get_total_sensor_cost(self):  # TODO: Rewrite the total sensor cost function for active perception.
        # For each of the sensors queried, the total cost of sensing.
        for query in self.sensing_acts:
            cost = 0
            if isinstance(self.sensing_acts[query], set):
                for sensor in self.sensing_acts[query]:
                    cost = cost + self.sensor_cost_dict[sensor]

                self.total_sensor_cost[query] = cost
            else:
                self.total_sensor_cost[query] = self.sensor_cost_dict[query]

        return

    def get_initial_dist(self):

        for state in self.agent_mdp.states:
            self.initial_dist[state] = self.agent_mdp.initial_distribution[state]
            self.mu_0[self.states_indx_dict[state]] = self.initial_dist[state]
        return

    ## The following is the state observation function for active perception single dynamic agent.

    # def get_state_obs2(self):
    #     for state, query in itertools.product(self.agent_mdp.states, self.sensing_acts):
    #         obs = set([])
    #         for sense in self.sensing_acts[query]:
    #             if sense == 'NO':
    #                 continue
    #             elif state in self.sensors.coverage[sense]:
    #                 obs.add(sense)
    #         # if len(obs) == 0:
    #         #     obs.add('0')
    #
    #         self.state_obs2[state][query] = obs
    #         # self.state_obs2[self.states_indx_dict[state]][self.sensing_act_indx_dict[query]] = obs
    #
    #     return

    # --- 2. GET_STATE_OBS2 UPDATE for two dynamic agents in the environment.---
    def get_state_obs2(self):
        for state, query in itertools.product(self.agent_mdp.states, self.sensing_acts):
            obs = set([])
            for sense in self.sensing_acts[query]:
                # Skip blind spots so they naturally return an empty set
                if sense in ['NO_E', 'NO_T']:
                    continue
                elif state in self.sensors.coverage[sense]:
                    obs.add(sense)

            # If both E and T are in the camera, join them into the single string!
            if len(obs) > 1:
                joined_str = "_".join(sorted(list(obs)))
                obs = {joined_str}  # Replace the set of two with a set of one joined string

            self.state_obs2[state][query] = obs
        return

    def get_initial_state(self):
        self.initial_state = random.choices(list(self.initial_dist.keys()), weights=list(self.initial_dist.values()))[0]
        return

    def get_initial_states(self):
        # Obtain the set of initial states.
        for state in self.agent_mdp.states:
            if self.initial_dist[state] > 0:
                self.initial_states.add(state)
        return


class SafetyDFA:
    """
    Deterministic Finite Automaton (DFA) for safety specifications in active perception for predictive safety.
    """

    def __init__(self, dfa_states, initial_state, accepting_states, transition_function=dict()):
        self.dfa_states = dfa_states  # Set of DFA states.
        self.dfa_initial_state = initial_state  # Initial DFA state.
        self.dfa_accepting_states = accepting_states  # Set of accepting DFA states or technically, the failure states.
        # self.dfa_labeling_func = labeling_func  # Labeling function: state -> label.
        self.dfa_transition_function = transition_function  # Transition function: (current_state, label) -> next_state.


class ProductHMM:
    """
    Construct the Product HMM of the Agent MDP and Safety DFA for active perception for predictive safety.

    Z_prod = S_agent x Q_dfa
    Output:
    - self.states: List of (z, q) tuples
    - self.trans_dict: Dict[state][action][next_state] -> Prob
    - self.obs_dict:   Dict[state][action][obs] -> Prob
    - self.fail_set:   Set of product states that are failing states (i.e., q in accepting states of DFA)
    """

    def __init__(self, hmm: labeledHMM, dfa: SafetyDFA):
        self.hmm = hmm
        self.dfa = dfa

        self.prod_states = []
        # for s, q in itertools.product(self.hmm.agent_mdp.states, self.dfa.dfa_states):
        #     self.prod_states.append((s, q))

        self.get_prod_states()

        self.prod_states_indx_dict = dict()
        self.prod_indx_states_dict = dict()

        indx = 0
        for state in self.prod_states:
            self.prod_states_indx_dict[state] = indx
            self.prod_indx_states_dict[indx] = state
            indx += 1

        self.failure_states = set()
        self.failure_states_indx_dict = dict()
        for state in self.prod_states:
            if state[1] in self.dfa.dfa_accepting_states:
                self.failure_states.add(state)
                self.failure_states_indx_dict[state] = self.prod_states_indx_dict[state]

        self.prod_transition_dict = defaultdict(
            lambda: defaultdict(dict))  # The transition dictionary for the augmented state-space.
        self.get_prod_transition_dict()

        self.prod_transition_mat = np.zeros((len(self.prod_states), len(self.prod_states)))
        self.get_prod_transition_mat()
        self.prod_emission_prob = defaultdict(
            lambda: defaultdict(dict))  # The emission probability dictionary for the product state-space.
        self.get_prod_emission_prob()
        self.prod_initial_dist = dict(
            [])  # The initial distribution of the augmented state-space. initial_dist[augstate]=probability
        # initial distribution array.
        self.prod_mu_0 = np.zeros(len(self.prod_states))

        self.get_initial_prod_dist()

    def get_prod_states(self):
        """
        Constructs reachable product states (s, q) starting from
        ALL states in self.hmm.initial_states paired with q0.
        """
        # 1. Initialize Queue with ALL possible start configurations
        # DFA usually has one start state q0.
        start_q = self.dfa.dfa_initial_state

        # The queue starts with (s, q0) for every s in the initial distribution
        queue = []
        visited = set()

        state_label_map = {
            s: next(iter(labels)) if isinstance(labels, (set, list)) else labels
            for s, labels in self.hmm.labels.items()
        }

        for start_s in self.hmm.initial_states:
            node = (start_s, start_q)
            if node not in visited:
                queue.append(node)
                visited.add(node)
        self.prod_states = list(queue)  # Keep order for indexing

        # 2. BFS to find reachable states
        # 2. BFS Loop
        idx = 0
        while idx < len(self.prod_states):
            curr_s, curr_q = self.prod_states[idx]
            idx += 1

            # --- CORRECTED OPTIMIZATION ---
            # Check if current state has outgoing transitions defined
            if curr_s in self.hmm.transition_dict:
                # Iterate over (next_state, probability) pairs
                for next_s, prob in self.hmm.transition_dict[curr_s].items():

                    # CRITICAL FIX: Only follow transitions with non-zero probability
                    if prob > 0.0:

                        # 1. Get label of the NEXT state
                        # (Ensure this helper exists or access .label directly)
                        # label = list(self.hmm.labels[next_s])[0]  # Assuming one label per state for simplicity
                        label = state_label_map[next_s]  # Assuming one label per state for simplicity

                        # 2. Step the DFA: delta(q, label) -> q'
                        next_q = self.dfa.dfa_transition_function[curr_q, label]

                        # 3. Add to Product State if it's new
                        next_node = (next_s, next_q)

                        if next_node not in visited:
                            visited.add(next_node)
                            self.prod_states.append(next_node)

    def get_prod_transition_dict(self):
        # The transition_dict is the transition function such that transition_dict[state][mask][next_state]=probability.
        for state, next_state in itertools.product(self.prod_states, self.prod_states):
            self.prod_transition_dict[state][next_state] = self.get_prod_transition_probability(state, next_state)

        return

    def get_prod_transition_mat(self):
        # The matrix representation of the transition function. transition_mat[i, j] = probability.
        for state, next_state in itertools.product(self.prod_states, self.prod_states):
            self.prod_transition_mat[self.prod_states_indx_dict[state], self.prod_states_indx_dict[next_state]] = \
                self.prod_transition_dict[state][next_state]
        return

    def get_prod_transition_probability(self, state, next_state):
        # This is computed as P(s'|s) 1{q' = delta(q, L(s'))}
        if next_state[1] == self.dfa.dfa_transition_function[(state[1], list(self.hmm.labels[next_state[0]])[0])]:
            return self.hmm.transition_dict[state[0]][next_state[0]]
        else:
            return 0.0

    def get_prod_emission_prob(self):
        # In the emission function for each state, and observation pairs.
        for state, query in itertools.product(self.prod_states, self.hmm.sensing_acts):
            for obs in self.hmm.observations:
                self.prod_emission_prob[state][query][obs] = self.hmm.emission_prob[state[0]][query][obs]
        return

    def get_initial_prod_dist(self):

        for state in self.prod_states:
            if state[1] == self.dfa.dfa_initial_state:
                self.prod_initial_dist[state] = self.hmm.initial_dist[state[0]]
                self.prod_mu_0[self.prod_states_indx_dict[state]] = self.hmm.initial_dist[state[0]]
            else:
                self.prod_initial_dist[state] = 0.0
        return

    # # the following is the sample_observation for different 'NO' and 'Null'.
    # def sample_observation(self, state):
    #     # TODO: Check if this is correct for active perception.
    #     obs_list = list(self.state_obs2[state])
    #     if state not in self.sensors.coverage[
    #         'NO']:  # probabilistic observation.
    #         obs_list.append('0')
    #         return random.choices(obs_list, weights=[1 - self.obs_noise, self.obs_noise])[0]
    #     else:  # When not under a sensor, return 'NO' with prob. 1.
    #         return random.choices(obs_list)[0]
    #
    # # The following is the sample_observation for SAME 'NO' and 'Null'.
    # def sample_observation_same_NO_Null(self, state):
    #     # TODO: Check if this is correct for active perception.
    #
    #     obs_list = list(self.state_obs2[state])
    #     if state[0] not in self.sensors.coverage[
    #         'NO']:  # probabilistic observation.
    #         obs_list.append('0')
    #         return random.choices(obs_list, weights=[1 - self.obs_noise, self.obs_noise])[0]
    #     else:  # When not under a sensor, return '0' with prob. 1.
    #         obs_new_list = list()
    #         obs_new_list.append('0')
    #         return random.choices(obs_new_list)[0]
    #
    # # The following is the sample_observation for SAME 'NO' and 'Null' but with also action observation.
    # def sample_observation_same_NO_Null_with_sensing_action(self, state):
    #     # Given an augmented state it gives a sample observation - true observation or null observation. TODO: Check
    #     #  if this is correct-- I am considering that whenever the robot is not under a sensor, it only gets null.
    #
    #     obs_list = list(self.state_obs2[state])
    #     if len(obs_list) == 0:  # To return null observation with prob. 1 when masked.
    #         obs_list.append('0')
    #         return (random.choices(obs_list)[0], state[1])
    #     elif state[0] not in self.sensors.coverage[
    #         'NO']:  # When not masked and under a sensor, probabilistic observation.
    #         obs_list.append('0')
    #         return (random.choices(obs_list, weights=[1 - self.obs_noise, self.obs_noise])[0], state[1])
    #     else:  # When not under a sensor, return '0' with prob. 1.
    #         obs_new_list = list()
    #         obs_new_list.append('0')
    #         return (random.choices(obs_new_list)[0], state[1])
    #
    # def sample_next_state(self, state, masking_act):
    #     # Given an augmented state, a masking action, the function returns a sampled next state.
    #     next_states_supp = list(self.transition_dict[state][masking_act].keys())
    #     next_states_prob = [self.transition_dict[state][masking_act][next_state] for next_state in next_states_supp]
    #     return random.choices(next_states_supp, weights=next_states_prob)[0]
