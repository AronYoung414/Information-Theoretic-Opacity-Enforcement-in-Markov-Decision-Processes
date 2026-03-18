import math

from setup_and_solvers.LP_for_nominal_policy import *
from setup_and_solvers.test_gradient_entropy_calculations_for_predictive_safety import *
from setup_and_solvers.markov_decision_process import *


def campus_example_predictive_safety(iter_num=1000, batch_size=100, V=100, T=3, lr=0.005, k_steps=1, alpha=0,
                                     num_test_episodes=1000):
    logger.add("logs_for_examples/log_file_for_campus_example_predictive_safety_9_cameras.log")

    logger.info(
        "This is the log file for the campus example with DFA based failure states for predictive safety with 9 camers in the environemt.")

    # --- 1. Map Geography & Agent Policies ---
    # Ego nodes: 1 to 43 inclusive
    ego_nodes = list(range(1, 44))
    clearance_node = 23  # Bryant Space Sci
    goal_node = 41  # The actual goal!

    # Traffic Zones (Scooter is banned from 41)
    zone_1 = [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22]
    zone_2 = [29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 42]
    valid_traffic_nodes = zone_1 + zone_2

    # Buffer Zone (Ego has left Zone 1, but hasn't entered Zone 2 yet)
    buffer_nodes = [21, 24, 25, 26, 27, 28]

    # The Ego's transition system (plant dynamics)
    # Format: current_node: [(next_node, probability), ...]
    ego_policy = {
        1: [(2, 0.5), (3, 0.5)],
        2: [(4, 1.0)],
        3: [(6, 0.2), (8, 0.8)],
        4: [(5, 0.7), (6, 0.3)],
        5: [(6, 0.5), (7, 0.5)],
        6: [(3, 0.5), (5, 0.5)],
        7: [(8, 0.7), (9, 0.15), (10, 0.15)],
        8: [(9, 0.1), (12, 0.2), (16, 0.7)],
        9: [(8, 0.45), (12, 0.45), (11, 0.1)],
        10: [(11, 0.7), (15, 0.3)],
        11: [(12, 0.5), (9, 0.5)],
        12: [(18, 0.75), (13, 0.25)],
        13: [(12, 0.5), (14, 0.1), (20, 0.4)],
        14: [(11, 0.7), (15, 0.15), (13, 0.15)],
        15: [(15, 0.4), (14, 0.4), (10, 0.2)],
        16: [(17, 1.0)],
        17: [(19, 0.25), (18, 0.75)],
        18: [(23, 0.45), (22, 0.5), (17, 0.05)],
        19: [(17, 0.5), (19, 0.5)],
        20: [(22, 0.2), (21, 0.2), (28, 0.55), (13, 0.05)],
        21: [(24, 1.0)],
        22: [(18, 0.65), (20, 0.35)],
        23: [(18, 0.9), (23, 0.1)],
        24: [(25, 0.5), (26, 0.5)],
        25: [(27, 1.0)],
        26: [(27, 1.0)],
        27: [(30, 1.0)],
        28: [(29, 1.0)],
        29: [(30, 0.7), (31, 0.3)],
        30: [(32, 1.0)],
        31: [(43, 0.9), (29, 0.1)],
        32: [(33, 0.05), (34, 0.5), (35, 0.45)],
        33: [(34, 0.85), (32, 0.1), (33, 0.05)],
        34: [(37, 0.5), (35, 0.5)],
        35: [(36, 1.0)],
        36: [(40, 1.0)],
        37: [(36, 0.9), (38, 0.1)],
        38: [(39, 1.0)],
        39: [(40, 1.0)],
        40: [(41, 0.8), (42, 0.2)],
        41: [(41, 1.0)],  # Absorbing Goal State (Chem Lab)
        42: [(40, 1.0)],
        43: [(41, 1.0)]
    }

    # ==========================================
    # 2. AGENT POLICIES (Corrected Traffic Logic)
    # ==========================================

    # Define the physical connections for the scooter's nodes
    # (e.g., if scooter is at 3, it can only drive to 5 or 6)
    traffic_adjacency = {
        # Zone 1 edges
        3: [6, 8],
        5: [6, 7],
        6: [3, 5],
        7: [5, 8, 9, 10],
        8: [16, 3, 9, 7, 12],
        9: [7, 8, 10, 11, 12],
        10: [7, 11, 15],
        11: [9, 10, 12, 14],
        12: [8, 9, 11, 13],
        13: [12, 14, 20],
        14: [11, 13, 15],
        15: [10, 14, 15],
        16: [17, 8],
        17: [16, 18],
        18: [17, 22],
        20: [22, 13],
        22: [18, 20],

        # 3: [6, 8, 3],
        # 5: [6, 7, 5],
        # 6: [3, 5, 6],
        # 7: [5, 8, 9, 10, 7],
        # 8: [16, 3, 9, 7, 12, 8],
        # 9: [7, 8, 10, 11, 12, 9],
        # 10: [7, 11, 15, 10],
        # 11: [9, 10, 12, 14, 11],
        # 12: [8, 9, 11, 13, 12],
        # 13: [12, 14, 20, 13],
        # 14: [11, 13, 15, 14],
        # 15: [10, 14, 15],
        # 16: [17, 8, 16],
        # 17: [16, 18, 17],
        # 18: [17, 22, 18],
        # 20: [22, 13, 20],
        # 22: [18, 20, 22],

        # Zone 2 edges
        29: [31, 30],
        30: [32],
        31: [29],
        32: [30, 34, 35],
        # 33: [],
        34: [32, 35, 37],
        35: [32, 34, 36],
        36: [35, 37, 40],
        37: [34, 36, 38],
        38: [39, 37],
        39: [38, 40],
        40: [42, 36, 39],
        42: [40, 36]}

    # 29: [31, 30, 29],
    # 30: [32, 30],
    # 31: [29, 31],
    # 32: [30, 34, 35, 32],
    # # 33: [],
    # 34: [32, 35, 37, 34],
    # 35: [32, 34, 36, 35],
    # 36: [35, 37, 40, 36],
    # 37: [34, 36, 38, 37],
    # 38: [39, 37, 38],
    # 39: [38, 40, 39],
    # 40: [42, 36, 39, 40],
    # 42: [40, 36, 42]}

    def get_traffic_transitions(current_traffic, current_ego):
        """Handles the context-switch and LOCAL random walks based on Ego position."""
        ego_in_zone_1 = (current_ego not in buffer_nodes) and (current_ego not in zone_2) and (current_ego != goal_node)
        active_zone = zone_1 if ego_in_zone_1 else zone_2

        # 1. Check for Context Switch (The ONLY time teleportation is allowed)
        if current_traffic not in active_zone:
            if active_zone == zone_2:
                return [(29, 0.5), (30, 0.5)]  # Spawn at Zone 2 entry
            else:
                return [(n, 1.0 / len(zone_1)) for n in zone_1]  # Uniform spawn in Zone 1

        # 2. Normal Local Random Walk (Strictly along physical edges!)
        # Look up the physical neighbors for the scooter's current node
        neighbors = traffic_adjacency.get(current_traffic, [current_traffic])

        # Distribute probability uniformly ONLY among immediate physical neighbors
        prob_per_neighbor = 1.0 / len(neighbors)
        return [(neighbor, prob_per_neighbor) for neighbor in neighbors]

    # def get_traffic_transitions(current_traffic, current_ego):
    #     """Handles the context-switch and random walks based on Ego position."""
    #     # If Ego is in Zone 1 (before buffer), track Zone 1. Otherwise, track Zone 2.
    #     ego_in_zone_1 = (current_ego not in buffer_nodes) and (current_ego not in zone_2) and (current_ego != goal_node)
    #     active_zone = zone_1 if ego_in_zone_1 else zone_2
    #
    #     # Check for Context Switch / Teleportation
    #     if current_traffic not in active_zone:
    #         if active_zone == zone_2:
    #             # Ego crossed into Buffer/Zone 2! Spawn traffic at the entry chokepoints.
    #             return [(29, 0.5), (30, 0.5)]
    #         else:
    #             # Ego restarted or spawned in Zone 1. Uniform spawn in Zone 1.
    #             return [(n, 1.0 / len(zone_1)) for n in zone_1]
    #
    #     # Normal Uniform Random Walk within the active zone
    #     return [(n, 1.0 / len(active_zone)) for n in active_zone]

    # --- 2. State Space Construction ---
    # Tuples used directly for your pipeline: (ego_node, traffic_node)
    states = [(e, c) for e in ego_nodes for c in valid_traffic_nodes]
    num_states = len(states)
    actions = ['a']

    print(f"Total States: {num_states}")

    # Temporary lookup strictly for numpy array indexing
    idx_lookup = {state: idx for idx, state in enumerate(states)}

    prob = {'a': np.zeros((num_states, num_states))}
    transitions = {}
    labels = {}
    initial = []

    # --- 3. Matrix & Label Generation ---
    for s in states:
        e, c = s

        # A. Strict Label Priority (Exactly ONE label per state)
        if e == c:
            labels[s] = {'crash'}  # Priority 1: Crash
        elif e == clearance_node:
            labels[s] = {'clear'}  # Priority 2: Clearance check
        elif e == goal_node:
            labels[s] = {'goal'}  # Priority 3: Goal
        else:
            labels[s] = {'safe'}  # Priority 4: Background safety

        # B. Transition Probabilities - Absorbing after crash btn ego and traffic.
        joint_next = []
        row_idx = idx_lookup[s]

        if e == c:
            # CRASH: The state becomes perfectly absorbing
            s_prime = (e, c)
            joint_next.append((s_prime, 1.0))
            prob['a'][row_idx, row_idx] = 1.0

        else:
            # NORMAL OPERATION: Calculate independent movements
            ego_next = ego_policy.get(e, [(e, 1.0)])  # Default to stay put if edge not defined
            traffic_next = get_traffic_transitions(c, e)

            for e_prime, p_e in ego_next:
                for c_prime, p_c in traffic_next:
                    p_joint = p_e * p_c

                    if p_joint > 0:
                        s_prime = (e_prime, c_prime)
                        joint_next.append((s_prime, p_joint))

                        # Populate the dense numpy array
                        col_idx = idx_lookup[s_prime]
                        prob['a'][row_idx, col_idx] = p_joint

        transitions[(s, 'a')] = joint_next

        # # B. Transition Probabilities - ghosting dynamics ie, ego and traffic continue to move even after crash.
        # ego_next = ego_policy.get(e, [(e, 1.0)])  # Default to stay put if edge not defined
        # traffic_next = get_traffic_transitions(c, e)
        #
        # joint_next = []
        # for e_prime, p_e in ego_next:
        #     for c_prime, p_c in traffic_next:
        #         p_joint = p_e * p_c
        #
        #         if p_joint > 0:
        #             s_prime = (e_prime, c_prime)
        #             joint_next.append((s_prime, p_joint))
        #
        #             # Populate the dense numpy array
        #             row_idx = idx_lookup[s]
        #             col_idx = idx_lookup[s_prime]
        #             prob['a'][row_idx, col_idx] = p_joint
        #
        # transitions[(s, 'a')] = joint_next

    print(f"Transition and label generation completed.")

    # --- 4. Initial Distribution & Targets ---
    # Ego starts at 1. Traffic starts uniformly in Zone 1.
    ego_start = 1
    for c_start in zone_1:
        initial.append((ego_start, c_start))

    initial_dist = {s: 0 for s in states}
    for s in initial:
        initial_dist[s] = 1.0 / len(initial)

    target = [s for s in states if s[0] == goal_node]

    # 1. Define Physical Camera Zones (9 Tightly Focused Cameras with Permanent Blind Spots)
    # Nodes 5, 10, 15, 19, 24, 29, 34, 39 are now PERMANENT BLIND SPOTS
    cam1_nodes = [1, 2, 3, 4]  # Early Warning Start
    cam2_nodes = [7, 8, 9]  # Highway A
    cam3_nodes = [11, 12, 13, 14]  # Highway B
    cam4_nodes = [16, 17, 18]  # Approach to Clearance
    cam5_nodes = [20, 21, 22]  # TODO: Check if changing this would be better for us!
    cam6_nodes = [25, 26, 27, 28]  # TODO: See if 28 going in a different camera coverage is a better thing?
    cam7_nodes = [30, 31, 32, 33]  # TODO: May be try with removing 31 from this camera..
    cam8_nodes = [35, 36, 37, 38]  # Approach to Goal
    cam9_nodes = [40, 41, 42, 43]  # DANGER: Goal Choke Point (Node 41)

    # Calculate the physical blind spots for the 'NO' sensor
    covered_nodes = set(cam1_nodes + cam2_nodes + cam3_nodes + cam4_nodes + cam5_nodes +
                        cam6_nodes + cam7_nodes + cam8_nodes + cam9_nodes)
    uncovered_ego = [n for n in ego_nodes if n not in covered_nodes]
    uncovered_traffic = [n for n in valid_traffic_nodes if n not in covered_nodes]

    # 2. Sensor Setup (Split into Ego and Traffic logical sensors)
    sensors = {
        'C1_E', 'C1_T', 'C2_E', 'C2_T', 'C3_E', 'C3_T',
        'C4_E', 'C4_T', 'C5_E', 'C5_T', 'C6_E', 'C6_T',
        'C7_E', 'C7_T', 'C8_E', 'C8_T', 'C9_E', 'C9_T',
        'NO_E', 'NO_T'
    }

    # 3. Build Joint Coverage Sets
    def build_cov(cam_nodes):
        set_E = {(e, c) for e in ego_nodes for c in valid_traffic_nodes if e in cam_nodes}
        set_T = {(e, c) for e in ego_nodes for c in valid_traffic_nodes if c in cam_nodes}
        return set_E, set_T

    setC1_E, setC1_T = build_cov(cam1_nodes)
    setC2_E, setC2_T = build_cov(cam2_nodes)
    setC3_E, setC3_T = build_cov(cam3_nodes)
    setC4_E, setC4_T = build_cov(cam4_nodes)
    setC5_E, setC5_T = build_cov(cam5_nodes)
    setC6_E, setC6_T = build_cov(cam6_nodes)
    setC7_E, setC7_T = build_cov(cam7_nodes)
    setC8_E, setC8_T = build_cov(cam8_nodes)
    setC9_E, setC9_T = build_cov(cam9_nodes)

    # The Blind Spot Coverage Sets
    setNO_E = {(e, c) for e in ego_nodes for c in valid_traffic_nodes if e in uncovered_ego}
    setNO_T = {(e, c) for e in ego_nodes for c in valid_traffic_nodes if c in uncovered_traffic}

    print(f"9-Camera Sensor coverage sets generated with permanent blind spots.")

    # 4. Masking / Sensing Actions
    sensing_action = {
        0: {'C1_E', 'C1_T'},
        1: {'C2_E', 'C2_T'},
        2: {'C3_E', 'C3_T'},
        3: {'C4_E', 'C4_T'},
        4: {'C5_E', 'C5_T'},
        5: {'C6_E', 'C6_T'},
        6: {'C7_E', 'C7_T'},
        7: {'C8_E', 'C8_T'},
        8: {'C9_E', 'C9_T'},
        9: {'NO_E', 'NO_T'}  # Action 9 is now NO sensing
    }

    no_sense_act = 9

    # 5. Sensor Noise (Phase 1: Zero noise sanity check)
    sensor_noise = 0.15

    # 6. Sensor Costs (The Asymmetric Trap)
    sensor_cost = {
        'C1_E': 10, 'C1_T': 0,
        'C2_E': 10, 'C2_T': 0,
        'C3_E': 10, 'C3_T': 0,
        'C4_E': 10, 'C4_T': 0,
        'C5_E': 10, 'C5_T': 0,  # BRUTAL Clearance Choke Point Cost
        'C6_E': 10, 'C6_T': 0,
        'C7_E': 10, 'C7_T': 0,
        'C8_E': 10, 'C8_T': 0,
        'C9_E': 10, 'C9_T': 0,  # BRUTAL Goal Choke Point Cost
        'NO_E': 0, 'NO_T': 0
    }

    # 7. Object Instantiation
    sensor_net = Sensor()
    sensor_net.sensors = sensors

    # Dynamically set coverage
    for i, (set_E, set_T) in enumerate([
        (setC1_E, setC1_T), (setC2_E, setC2_T), (setC3_E, setC3_T),
        (setC4_E, setC4_T), (setC5_E, setC5_T), (setC6_E, setC6_T),
        (setC7_E, setC7_T), (setC8_E, setC8_T), (setC9_E, setC9_T)
    ], start=1):
        sensor_net.set_coverage(f'C{i}_E', set_E)
        sensor_net.set_coverage(f'C{i}_T', set_T)

    sensor_net.set_coverage('NO_E', setNO_E)
    sensor_net.set_coverage('NO_T', setNO_T)

    sensor_net.sensing_actions = sensing_action
    sensor_net.sensor_noise = sensor_noise
    sensor_net.sensor_cost_dict = sensor_cost

    # --- 5. Object Instantiation ---
    agent_mdp = MDP(
        init=initial,
        actlist=actions,
        states=states,
        prob=prob,
        trans=transitions,
        labels=labels,
        init_dist=initial_dist,
        goal_states=target
    )

    agent_mdp.get_supp()
    agent_mdp.gettrans()
    agent_mdp.get_reward()

    # Using the baked-in agent policy (Approach A)
    goal_policy = dict([])

    # Dynamically assign probability 1.0 for action 'a' across all joint states
    for s in states:
        goal_policy[(s, 'a')] = 1.0

    logger.debug("Goal policy:")
    logger.debug(goal_policy)

    # Defining the states and transitions for the
    # DFA here is to simply accept the states that get to the failure state.
    dfa_states = ['q0', 'q1', 'q_win', 'q_fail']
    dfa_initial = 'q0'
    dfa_accepting_states = {'q_fail'}  # The target state for predictability tracking

    # Format: (current_dfa_state, mdp_label): next_dfa_state
    dfa_transitions = {
        # Transitions from q0 (Uncleared)
        ('q0', 'crash'): 'q_fail',
        ('q0', 'goal'): 'q_fail',  # Failed because goal was reached before clearance
        ('q0', 'clear'): 'q1',  # Got clearance!
        ('q0', 'safe'): 'q0',  # Background wandering

        # Transitions from q1 (Cleared)
        ('q1', 'crash'): 'q_fail',
        ('q1', 'goal'): 'q_win',  # Success! Reached goal safely after clearance
        ('q1', 'clear'): 'q1',  # Still clear
        ('q1', 'safe'): 'q1',  # Driving toward goal safely

        # Absorbing Transitions from q_win
        ('q_win', 'crash'): 'q_win',
        ('q_win', 'goal'): 'q_win',
        ('q_win', 'clear'): 'q_win',
        ('q_win', 'safe'): 'q_win',

        # Absorbing Transitions from q_fail
        ('q_fail', 'crash'): 'q_fail',
        ('q_fail', 'goal'): 'q_fail',
        ('q_fail', 'clear'): 'q_fail',
        ('q_fail', 'safe'): 'q_fail'
    }

    # --- 1. Construct the underlying models ---
    # Note: no_sense_act is now 5 (matching our updated sensor dictionary)
    labeled_hmm = labeledHMM(
        agent_mdp=agent_mdp,
        sensors=sensor_net,
        no_sensing_act=no_sense_act,
        goal_policy=goal_policy
    )

    safety_dfa = SafetyDFA(
        dfa_states=dfa_states,
        initial_state=dfa_initial,
        accepting_states=dfa_accepting_states,
        transition_function=dfa_transitions
    )

    product_hmm = ProductHMM(labeled_hmm, safety_dfa)

    # --- 2. Initialize the solver ---
    lstm_solver = GradientDescent_LSTM_solver(prod_hmm=product_hmm, k_step=k_steps, lr=lr)

    # # --- 3. Initial Evaluation ---
    # # Define a valid joint trace for evaluation based on our new map
    # # (Ego moves 1->2->4->5->9, Traffic moves 3->5->9->11->12)
    # path = [(1, 3), (2, 5), (4, 9), (5, 11), (9, 12)]
    #
    # print(f"Initial Policy Evaluation:")
    # lstm_solver.evaluate_policy(horizon=T, forced_trace=path)

    # --- 4. Train the LSTM ---
    print("\nStarting LSTM Training...")
    lstm_solver.train(iterations=iter_num, V=V, batch_size=batch_size, horizon=T, alpha=alpha)

    # SAVE MODEL (Uncomment when ready to save weights)
    model_filename = "policy_for_gamms_campus_map_9_cameras.pth"
    torch.save(lstm_solver.policy_net.state_dict(), model_filename)
    print(f"Policy saved to {model_filename}")

    # # --- 5. Final Evaluation & Monte Carlo Metrics ---
    # print(f"\nFinal Policy Evaluation:")
    # lstm_solver.evaluate_policy(horizon=T, forced_trace=path)

    print("\nRunning Monte Carlo Evaluations (This may take a moment...)")

    # Evaluate the Brier score for the trained policy.
    avg_brier_score, ci_brier, acc, ci_acc, avg_cost, ci_cost = lstm_solver.evaluate_brier_score_agent(
        episodes=num_test_episodes, horizon=T, k_step=k_steps, uniform_random=False
    )

    # # Evaluate the Brier score for the uniform random policy.
    # avg_brier_score_uni_rdn, ci_brier_uni_rdn, acc_uni_rdn, ci_acc_uni_rdn = lstm_solver.evaluate_brier_score_agent(
    #     episodes=num_test_episodes, horizon=T, k_step=k_steps, uniform_random=True
    # )

    # Evaluate Brier score and accuracy for the Oracle baseline.
    avg_brier_score_oracle, ci_brier_oracle, acc_oracle, ci_acc_oracle = lstm_solver.evaluate_oracle_baseline(
        episodes=num_test_episodes, horizon=T, k_step=k_steps
    )

    # Evaluate Brier score and accuracy for the random policy with stop after death one..
    avg_brier_score_uni_rdn_no_accu, ci_brier_uni_rdn_no_accu, acc_uni_rdn_no_accu, ci_acc_uni_rdn_no_accu, avg_cost_no_acc, ci_cost_no_accu = lstm_solver.evaluate_brier_score_agent_no_accumulation_after_death(
        episodes=num_test_episodes, horizon=T, k_step=k_steps, uniform_random=True
    )

    # # Evaluate Brier score and accuracy for the lstm policu with stop after death one..
    # avg_brier_score_no_accu, ci_brier_no_accu, acc_no_accu, ci_acc_no_accu = lstm_solver.evaluate_brier_score_agent_no_accumulation_after_death(
    #     episodes=num_test_episodes, horizon=T, k_step=k_steps, uniform_random=False
    # )

    # --- 6. Print Results ---
    print("\n=== RESULTS ===")
    print(f"Trained Policy Brier Score: {avg_brier_score:.6f} ± {ci_brier:.6f}")
    print(f"Trained Policy Accuracy:    {acc:.6f}% ± {ci_acc:.6f}%\n")
    print(f"Trained Policy Average Cost: {avg_cost:.6f} ± {ci_cost:.6f}\n")

    # print(f"Trained Policy Brier Score: {avg_brier_score_no_accu:.6f} ± {ci_brier_no_accu:.6f}")
    # print(f"Trained Policy Accuracy:    {acc_no_accu:.6f}% ± {ci_acc_no_accu:.6f}%\n")
    #
    # print(f"Uniform Random Brier Score: {avg_brier_score_uni_rdn:.6f} ± {ci_brier_uni_rdn:.6f}")
    # print(f"Uniform Random Accuracy:    {acc_uni_rdn:.6f}% ± {ci_acc_uni_rdn:.6f}%\n")

    print(f"Uniform Random Brier Score: {avg_brier_score_uni_rdn_no_accu:.6f} ± {ci_brier_uni_rdn_no_accu:.6f}")
    print(f"Uniform Random Accuracy:    {acc_uni_rdn_no_accu:.6f}% ± {ci_acc_uni_rdn_no_accu:.6f}%\n")
    print(f"Uniform Random Average Cost: {avg_cost_no_acc:.6f} ± {ci_cost_no_accu:.6f}\n")

    print(f"Oracle Baseline Brier Score: {avg_brier_score_oracle:.6f} ± {ci_brier_oracle:.6f}")
    print(f"Oracle Baseline Accuracy:    {acc_oracle:.6f}% ± {ci_acc_oracle:.6f}%")
