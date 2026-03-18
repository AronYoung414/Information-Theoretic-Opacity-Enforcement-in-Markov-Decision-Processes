import gamms
import time
import random
import pickle
import torch
import torch.nn.functional as F
import gc

# --- 1. IMPORT YOUR CODEBASE ---
from setup_and_solvers.policy import ObservationPolicyLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Observation Vocabulary...")
with open('observation_indx_dict.pkl', 'rb') as f:
    obs_to_idx_dict = pickle.load(f)

num_of_observations = len(obs_to_idx_dict)

print("Loading LSTM Policy...")
policy_net = ObservationPolicyLSTM(num_obs=num_of_observations, num_actions=10).to(device)
policy_net.load_state_dict(torch.load("policy_for_gamms_campus_map_9_cameras.pth", weights_only=True))
policy_net.eval()

lstm_hidden = policy_net.init_hidden(1)
prev_obs_tuple = None
visual_sensor_queried = 9
is_first_step = True
global_crashed = False
sim_log_text = "Initializing Monitor..."

# --- 2. DYNAMICS DATA ---
zone_1 = [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22]
zone_2 = [29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 42]
buffer_nodes = [21, 24, 25, 26, 27, 28]

ego_policy_dict = {
    1: [(2, 0.5), (3, 0.5)], 2: [(4, 1.0)], 3: [(6, 0.2), (8, 0.8)],
    4: [(5, 0.7), (6, 0.3)], 5: [(6, 0.5), (7, 0.5)], 6: [(3, 0.5), (5, 0.5)],
    7: [(8, 0.7), (9, 0.15), (10, 0.15)], 8: [(9, 0.1), (12, 0.2), (16, 0.7)],
    9: [(8, 0.45), (12, 0.45), (11, 0.1)], 10: [(11, 0.7), (15, 0.3)],
    11: [(12, 0.5), (9, 0.5)], 12: [(18, 0.75), (13, 0.25)],
    13: [(12, 0.5), (14, 0.1), (20, 0.4)], 14: [(11, 0.7), (15, 0.15), (13, 0.15)],
    15: [(15, 0.4), (14, 0.4), (10, 0.2)], 16: [(17, 1.0)],
    17: [(19, 0.25), (18, 0.75)], 18: [(23, 0.45), (22, 0.5), (17, 0.05)],
    19: [(17, 0.5), (19, 0.5)], 20: [(22, 0.2), (21, 0.2), (28, 0.55), (13, 0.05)],
    21: [(24, 1.0)], 22: [(18, 0.65), (20, 0.35)], 23: [(18, 0.9), (23, 0.1)],
    24: [(25, 0.5), (26, 0.5)], 25: [(27, 1.0)], 26: [(27, 1.0)],
    27: [(30, 1.0)], 28: [(29, 1.0)], 29: [(30, 0.7), (31, 0.3)],
    30: [(32, 1.0)], 31: [(43, 0.9), (29, 0.1)], 32: [(33, 0.05), (34, 0.5), (35, 0.45)],
    33: [(34, 0.85), (32, 0.1), (33, 0.05)], 34: [(37, 0.5), (35, 0.5)],
    35: [(36, 1.0)], 36: [(40, 1.0)], 37: [(36, 0.9), (38, 0.1)],
    38: [(39, 1.0)], 39: [(40, 1.0)], 40: [(41, 0.8), (42, 0.2)],
    41: [(41, 1.0)], 42: [(40, 1.0)], 43: [(41, 1.0)]
}

traffic_adjacency = {
    3: [6, 8], 5: [6, 7], 6: [3, 5], 7: [5, 8, 9, 10],
    8: [16, 3, 9, 7, 12], 9: [7, 8, 10, 11, 12], 10: [7, 11, 15],
    11: [9, 10, 12, 14], 12: [8, 9, 11, 13], 13: [12, 14, 20],
    14: [11, 13, 15], 15: [10, 14, 15], 16: [17, 8], 17: [16, 18],
    18: [17, 22], 20: [22, 13], 22: [18, 20], 29: [31, 30],
    30: [32], 31: [29], 32: [30, 34, 35], 34: [32, 35, 37],
    35: [32, 34, 36], 36: [35, 37, 40], 37: [34, 36, 38],
    38: [39, 37], 39: [38, 40], 40: [42, 36, 39], 42: [40, 36]
}

# --- 3. SETUP GAMMS CONTEXT & GRAPH ---
ctx = gamms.create_context(vis_engine=gamms.visual.Engine.PYGAME)
graph = ctx.graph.graph

edge_set = set()
for current_node, transitions in ego_policy_dict.items():
    for next_node, prob in transitions:
        if current_node != next_node and prob > 0:
            edge_set.add(tuple(sorted((current_node, next_node))))

for current_node, neighbors in traffic_adjacency.items():
    for next_node in neighbors:
        if current_node != next_node:
            edge_set.add(tuple(sorted((current_node, next_node))))

edges = list(edge_set)

base_coords = {
    # 0: (50, 50), 1: (-100, 360), 2: (150, 160), 3: (150, 560),
    # 4: (250, 260), 6: (250, 360), 8: (250, 460),
    # 5: (350, 260), 9: (350, 360), 12: (350, 460), 16: (350, 560),
    # 7: (450, 260), 11: (450, 360), 13: (450, 460), 18: (450, 560), 17: (450, 660),
    # 10: (550, 260), 14: (550, 360), 20: (550, 460), 22: (550, 560), 19: (550, 660),
    # 15: (650, 260), 21: (650, 360), 28: (650, 460), 23: (650, 560),
    # 24: (750, 310), 29: (750, 460),
    # 25: (850, 260), 26: (850, 360), 30: (850, 460), 31: (850, 560),
    # 27: (950, 310), 32: (950, 460), 43: (950, 860),
    # 34: (1050, 410), 35: (1050, 510), 33: (1050, 610),
    # 37: (1150, 410), 36: (1150, 510),
    # 38: (1350, 360), 40: (1400, 510),
    # 39: (1550, 360), 42: (1250, 610),
    # 41: (1050, 910)

    0: (-250, 500), 1: (50, 500), 2: (200, 200), 3: (200, 800),
    4: (350, 300), 6: (350, 500), 8: (350, 700),
    5: (500, 300), 9: (500, 500), 12: (500, 700), 16: (500, 900),
    7: (650, 300), 11: (650, 500), 13: (650, 700), 18: (650, 900), 17: (650, 1100),
    10: (800, 300), 14: (800, 500), 20: (800, 700), 22: (800, 900), 19: (800, 1100),
    15: (950, 300), 21: (950, 500), 28: (950, 700), 23: (950, 900),
    24: (1100, 400), 29: (1100, 700),
    25: (1250, 300), 26: (1250, 500), 30: (1250, 700), 31: (1250, 900),
    27: (1400, 400), 32: (1400, 700), 43: (1400, 1200),
    34: (1550, 600), 35: (1550, 800), 33: (1550, 1000),
    37: (1700, 600), 36: (1700, 800),
    38: (1950, 500), 40: (2000, 800),
    39: (2200, 500), 42: (1950, 1000),
    41: (1550, 1300)
}

# scale_x, scale_y = 1.25, 1.2
scale_x, scale_y = 2, 2
node_coords = {nid: (int(pos[0] * scale_x), int(pos[1] * scale_y)) for nid, pos in base_coords.items()}

for i in range(43):
    if i not in node_coords:
        node_coords[i] = (100 + (i * 20) % 1200, 50)

for nid, pos in node_coords.items():
    graph.add_node({'id': nid, 'x': pos[0], 'y': pos[1]})

for i, (u, v) in enumerate(edges):
    graph.add_edge({'id': i, 'source': u, 'target': v, 'length': 1.0})

ctx.visual.set_graph_visual(width=1800, height=950)

sensor_coverage = {
    0: [1, 2, 3, 4], 1: [7, 8, 9], 2: [11, 12, 13, 14],
    3: [16, 17, 18], 4: [20, 21, 22], 5: [25, 26, 27, 28],
    6: [30, 31, 32, 33], 7: [35, 36, 37, 38], 8: [40, 41, 42, 43]
}


# --- 4. THE NOISY MULTI-NODE SENSOR ---
@ctx.sensor.custom(name="ZONE_MONITOR")
class ZoneMonitorSensor(gamms.typing.ISensor):
    def __init__(self, ctx, sensor_id, target_nodes, ego_name, red_team_names, false_negative_rate=0.15):
        self.ctx = ctx
        self._sensor_id = sensor_id
        self.target_nodes = target_nodes
        self.ego_name = ego_name
        self.red_team_names = red_team_names
        self.false_negative_rate = false_negative_rate
        self._data = {'observation': '0'}  # Note: Now the string '0'

    @property
    def type(self):
        return gamms.sensor.SensorType.CUSTOM

    @property
    def data(self):
        return self._data

    @property
    def sensor_id(self):
        return self._sensor_id

    def sense(self, node_id: int):
        cam_num = int(self._sensor_id.split('_')[1]) + 1  # Use for strings (C1, C2...)

        # 1. Roll for JOINT sensor failure
        if random.random() <= self.false_negative_rate:
            self._data['observation'] = '0'
            return

        # 2. Sensor is active - accurately detect who is present
        ego_detected = False
        traffic_detected = False

        ego_agent = self.ctx.agent.get_agent(self.ego_name)
        if ego_agent and ego_agent.current_node_id in self.target_nodes:
            ego_detected = True

        for red_name in self.red_team_names:
            red_agent = self.ctx.agent.get_agent(red_name)
            if red_agent and red_agent.current_node_id in self.target_nodes:
                traffic_detected = True
                break

                # Format exact strings corresponding to the dictionary keys
        if ego_detected and traffic_detected:
            self._data['observation'] = f"C{cam_num}_E_C{cam_num}_T"
        elif ego_detected:
            self._data['observation'] = f"C{cam_num}_E"
        elif traffic_detected:
            self._data['observation'] = f"C{cam_num}_T"
        else:
            self._data['observation'] = '0'

    def update(self, data: dict):
        pass

    def set_owner(self, owner: str):
        pass


# --- 5. AGENT STRATEGIES ---
def sample_from_transitions(transitions):
    r, cumulative = random.random(), 0.0
    for node, prob in transitions:
        cumulative += prob
        if r <= cumulative: return node
    return transitions[-1][0]


def ego_policy(state):
    if global_crashed:
        state['action'] = state['curr_pos']
        return
    curr = state['curr_pos']
    transitions = ego_policy_dict.get(curr, [(curr, 1.0)])
    state['action'] = sample_from_transitions(transitions)


def traffic_1_policy(state):
    if global_crashed:
        state['action'] = state['curr_pos']
        return
    curr = state['curr_pos']
    neighbors = traffic_adjacency.get(curr, [curr])
    valid_neighbors = [n for n in neighbors if n in zone_1]
    if not valid_neighbors: valid_neighbors = [curr]
    state['action'] = random.choice(valid_neighbors)


def traffic_2_policy(state):
    if global_crashed:
        state['action'] = state['curr_pos']
        return
    curr = state['curr_pos']
    ego_pos = ctx.agent.get_agent('ego').current_node_id

    if ego_pos in buffer_nodes:
        state['action'] = random.choice([29, 30])
        return

    neighbors = traffic_adjacency.get(curr, [curr])
    valid_neighbors = [n for n in neighbors if n in zone_2]
    if not valid_neighbors: valid_neighbors = [curr]
    state['action'] = random.choice(valid_neighbors)


def monitor_active_sensing_strategy(state):
    global lstm_hidden, prev_obs_tuple, visual_sensor_queried, is_first_step

    if is_first_step:
        obs_input = torch.zeros(1, dtype=torch.long, device=device)
        is_first_step = False
    else:
        try:
            obs_idx = obs_to_idx_dict[prev_obs_tuple]
        except KeyError:
            print(f"WARNING: Tuple {prev_obs_tuple} not found! Defaulting to 0.")
            obs_idx = 0
        obs_input = torch.tensor([obs_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, lstm_hidden = policy_net(obs_input, lstm_hidden)
        current_action_idx = torch.multinomial(F.softmax(logits, dim=1), 1).item()

    if current_action_idx == 9:
        mapped_obs = '0'
    else:
        sensor_name = f"cam_{current_action_idx}"
        mapped_obs = state['sensor'][sensor_name][1]['observation']

    # EXACT TUPLE FORMAT: (string, action_integer) where action_integer is 0-9
    current_obs_tuple = (mapped_obs, current_action_idx)

    # Terminal Print Output
    ego_pos = ctx.agent.get_agent('ego').current_node_id
    t1_pos = ctx.agent.get_agent('traffic_1').current_node_id
    t2_pos = ctx.agent.get_agent('traffic_2').current_node_id
    query_str = f"Cam {current_action_idx + 1}" if current_action_idx < 9 else "SLEEP"
    print(f"Ego: {ego_pos:<2} | T1: {t1_pos:<2} | T2: {t2_pos:<2} | Query: {query_str:<6} | Tuple: {current_obs_tuple}")

    global sim_log_text

    query_str = query_str if current_action_idx < 9 else "No Query"

    # Create the display string and print it to the terminal as a backup
    sim_log_text = f"Ego: {ego_pos:<2} | Traffic 1: {t1_pos:<2} | Traffic 2: {t2_pos:<2} | Query: {query_str:<6} | Obs: {current_obs_tuple}"
    print(sim_log_text)

    prev_obs_tuple = current_obs_tuple
    visual_sensor_queried = current_action_idx
    state['action'] = state['curr_pos']


# --- 6. INITIALIZE AGENTS & SENSORS ---
ctx.agent.create_agent(name='monitor', start_node_id=0)
ctx.visual.set_agent_visual(name='monitor', color=(0, 0, 0), size=5)
ctx.agent.get_agent('monitor').register_strategy(monitor_active_sensing_strategy)

ctx.agent.create_agent(name='ego', start_node_id=1)
ctx.visual.set_agent_visual(name='ego', color=(0, 0, 255), size=54)
ctx.agent.get_agent('ego').register_strategy(ego_policy)

ctx.agent.create_agent(name='traffic_1', start_node_id=3)
ctx.visual.set_agent_visual(name='traffic_1', color=(255, 0, 0), size=40)
ctx.agent.get_agent('traffic_1').register_strategy(traffic_1_policy)

ctx.agent.create_agent(name='traffic_2', start_node_id=29)
ctx.visual.set_agent_visual(name='traffic_2', color=(255, 0, 0), size=40)
ctx.agent.get_agent('traffic_2').register_strategy(traffic_2_policy)

for action_id, target_nodes in sensor_coverage.items():
    s_name = f"cam_{action_id}"
    sensor = ZoneMonitorSensor(ctx, s_name, target_nodes, 'ego', ['traffic_1', 'traffic_2'], false_negative_rate=0.15)
    ctx.sensor.add_sensor(sensor)
    ctx.agent.get_agent('monitor').register_sensor(s_name, sensor)


# --- 7. VISUALIZATION ARTISTS ---
# def draw_enhanced_graph(draw_ctx, data):
#     for (u, v) in edges:
#         n1 = draw_ctx.graph.graph.get_node(u)
#         n2 = draw_ctx.graph.graph.get_node(v)
#         if n1 and n2:
#             draw_ctx.visual.render_line(n1.x, n1.y, n2.x, n2.y, color=(80, 80, 80), width=4)
#
#     for nid in node_coords.keys():
#         n = draw_ctx.graph.graph.get_node(nid)
#         if n: draw_ctx.visual.render_circle(n.x, n.y, radius=16, color=(30, 30, 30))
#
#
# ctx.visual.add_artist("enhanced_graph", gamms.visual.Artist(ctx, drawer=draw_enhanced_graph, layer=10))

def draw_enhanced_graph(draw_ctx, data):
    for (u, v) in edges:
        n1 = draw_ctx.graph.graph.get_node(u)
        n2 = draw_ctx.graph.graph.get_node(v)
        if n1 and n2:
            draw_ctx.visual.render_line(n1.x, n1.y, n2.x, n2.y, color=(80, 80, 80), width=4)

    for nid in node_coords.keys():
        n = draw_ctx.graph.graph.get_node(nid)
        if n:
            # Draw the thick node circle
            draw_ctx.visual.render_circle(n.x, n.y, radius=16, color=(30, 30, 30))

            # Draw the node ID number just above and to the right of the node
            try:
                draw_ctx.visual.render_text(str(nid), n.x + 18, n.y - 20, color=(0, 0, 0))
            except Exception:
                pass


ctx.visual.add_artist("enhanced_graph", gamms.visual.Artist(ctx, drawer=draw_enhanced_graph, layer=10))


def draw_active_sensor(draw_ctx, data):
    if visual_sensor_queried == 9: return
    target_nodes = sensor_coverage.get(visual_sensor_queried, [])
    sensor = draw_ctx.sensor.get_sensor(f"cam_{visual_sensor_queried}")
    is_detected = (sensor and sensor.data['observation'] != '0')

    for node_id in target_nodes:
        node = draw_ctx.graph.graph.get_node(node_id)
        if node:
            draw_ctx.visual.render_circle(node.x, node.y, radius=26, color=(255, 255, 0), width=5)
            if is_detected:
                draw_ctx.visual.render_circle(node.x, node.y, radius=16, color=(0, 255, 0))


ctx.visual.add_artist("query_viz", gamms.visual.Artist(ctx, drawer=draw_active_sensor, layer=50))


def draw_hud(draw_ctx, data):
    # # Draw the black background bar at the ACTUAL bottom (y=0 to y=60)
    # draw_ctx.visual.render_polygon([
    #     (0, 0), (1600, 0), (1600, 60), (0, 60)
    # ], color=(20, 20, 20))

    # Render the text perfectly centered inside that bottom box
    try:
        draw_ctx.visual.render_text(sim_log_text, 20, 20, color=(0, 0, 0))
    except Exception as e:
        pass


ctx.visual.add_artist("hud_viz", gamms.visual.Artist(ctx, drawer=draw_hud, layer=100))

# --- 8. MAIN LOOP ---
print("Running Autonomous PyTorch-GAMMS Simulation...")
# time.sleep(15)
step_counter = 0

while not ctx.is_terminated():
    step_counter += 1
    state_dict = {agent.name: agent.get_state() for agent in ctx.agent.create_iter()}

    if not global_crashed:
        ego_pos = state_dict['ego']['curr_pos']
        t1_pos = state_dict['traffic_1']['curr_pos']
        t2_pos = state_dict['traffic_2']['curr_pos']
        if ego_pos == t1_pos or ego_pos == t2_pos:
            global_crashed = True
            print(f"CRASH DETECTED AT NODE {ego_pos}! Agents locked.")

    for agent in ctx.agent.create_iter():
        agent.strategy(state_dict[agent.name])

    ctx.visual.simulate()
    time.sleep(3.0)

    for agent in ctx.agent.create_iter():
        agent.set_state()

    if step_counter >= 20:
        print("Reached the horizon limit. Ending simulation.")
        ctx.terminate()

try:
    del ctx
    gc.collect()
except Exception:
    pass
