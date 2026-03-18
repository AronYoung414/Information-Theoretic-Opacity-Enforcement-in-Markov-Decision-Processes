import gamms
import time
import random
import pickle
import torch
import torch.nn.functional as F
import gc

# --- 1. IMPORT YOUR CODEBASE ---
# Import just the lightweight LSTM architecture
from setup_and_solvers.policy import ObservationPolicyLSTM

# --- 2. INITIALIZE THE BRAIN & DICTIONARY ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Observation Vocabulary...")
with open('observation_indx_dict.pkl', 'rb') as f:
    obs_to_idx_dict = pickle.load(f)

num_of_observations = len(obs_to_idx_dict)

print("Loading LSTM Policy...")
# ### FILL THIS IN: Add your actual input_size, hidden_size, output_size etc. ###
policy_net = ObservationPolicyLSTM(num_obs=num_of_observations, num_actions=5).to(device)

# ### FILL THIS IN: Point this to your saved weights file ###
# policy_net.load_state_dict(torch.load("policy_k1_alpha001_2.pth"))
# Just add weights_only=True
policy_net.load_state_dict(torch.load("policy_for_gamms_running_example_1.pth", weights_only=True))
policy_net.eval()  # Set to inference mode

# Initialize the LSTM's memory
lstm_hidden = policy_net.init_hidden(1)

# Global tracking variable for the LSTM's current decision (starts at 4 = Sleep)
# last_sensor_queried = 4
# visual_sensor_queried = 4
# is_first_step = True  # Tracks the LSTM's start token

prev_obs_tuple = None  # Stores the (obs, action) to feed the LSTM on the NEXT tick
visual_sensor_queried = 4  # Tells PyGame where to draw the yellow ring
is_first_step = True  # Tracks the start token

# --- 3. SETUP GAMMS CONTEXT & GRAPH ---
ctx = gamms.create_context(vis_engine=gamms.visual.Engine.PYGAME)
graph = ctx.graph.graph

nodes = {
    0: {'x': 100, 'y': 300},  # Start
    1: {'x': 300, 'y': 300},  # Target of Sensor 0 (R)
    2: {'x': 500, 'y': 300},  # Target of Sensor 1 (G)
    3: {'x': 700, 'y': 300},  # Target of Sensor 2 (P)
    4: {'x': 900, 'y': 300},  # Goal. Target of Sensor 3 (B)
    5: {'x': 500, 'y': 100},  # Trap State
}

for nid, pos in nodes.items():
    graph.add_node({'id': nid, 'x': pos['x'], 'y': pos['y']})

edges = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 4),
    (1, 5), (2, 5), (3, 5), (5, 5)
]
for i, (u, v) in enumerate(edges):
    graph.add_edge({'id': i, 'source': u, 'target': v, 'length': 1.0})

ctx.visual.set_graph_visual(width=1280, height=720)

# Map action index (0-3) to physical node ID (1-4)
sensor_targets = {0: 1, 1: 2, 2: 3, 3: 4}


# --- 4. THE CUSTOM SENSOR ---
@ctx.sensor.custom(name="NODE_MONITOR")
class NodeMonitorSensor(gamms.typing.ISensor):
    def __init__(self, ctx, sensor_id, target_node_id, red_team_names):
        self.ctx = ctx
        self._sensor_id = sensor_id
        self.target_node_id = target_node_id
        self.red_team_names = red_team_names
        self._data = {'observation': 'Null'}
        self._owner = None

    def set_owner(self, owner: str):
        self._owner = owner

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
        self._data['observation'] = 'Null'
        for red_name in self.red_team_names:
            red_agent = self.ctx.agent.get_agent(red_name)
            if red_agent.current_node_id == self.target_node_id:
                # Returns its own integer ID (e.g., 0, 1, 2, or 3) if detected
                self._data['observation'] = int(self._sensor_id.split('_')[-1])
                break

    def update(self, data: dict):
        pass


# --- 5. AGENT STRATEGIES ---

def red_stochastic_goal_policy(state):
    curr = state['curr_pos']
    if curr == 0:
        next_n = 1
    elif curr == 1:
        next_n = 2 if random.random() < 0.8 else 5
    elif curr == 2:
        next_n = 3 if random.random() < 0.8 else 5
    elif curr == 3:
        next_n = 4 if random.random() < 0.8 else 5
    elif curr == 4:
        next_n = 4
    elif curr == 5:
        next_n = 5
    else:
        next_n = curr

    state['action'] = next_n


# def blue_active_sensing_strategy(state):
#     global last_sensor_queried, lstm_hidden
#
#     # --- TRANSLATION MAP ---
#     sensor_id_to_color = {0: 'R', 1: 'G', 2: 'P', 3: 'B'}
#
#     # 1. Extract physical observation from GAMMS and MAP it
#     if last_sensor_queried == 4:
#         mapped_obs = '0'  # Custom 'Null' representation
#     else:
#         sensor_name = f"monitor_{last_sensor_queried}"
#         raw_obs = state['sensor'][sensor_name][1]['observation']
#
#         if raw_obs == 'Null':
#             mapped_obs = '0'
#         else:
#             mapped_obs = sensor_id_to_color[raw_obs]
#
#     red_pos = ctx.agent.get_agent('robot').current_node_id
#
#     # 2. Form the observation tuple exactly as your dictionary expects it
#     current_obs_tuple = (mapped_obs, last_sensor_queried)
#     print(f"Robot: S{red_pos:<2} | Tuple: {current_obs_tuple}")
#
#     # 3. Translate to integer index
#     try:
#         obs_idx = obs_to_idx_dict[current_obs_tuple]
#     except KeyError:
#         print(f"WARNING: Tuple {current_obs_tuple} not found in dictionary! Defaulting to index 0.")
#         obs_idx = 0
#
#     obs_input = torch.tensor([obs_idx], dtype=torch.long, device=device)
#
#     # 4. Neural Network Forward Pass
#     with torch.no_grad():
#         logits, lstm_hidden = policy_net(obs_input, lstm_hidden)
#         probs_action = F.softmax(logits, dim=1)
#         next_action_idx = torch.multinomial(probs_action, 1).item()
#
#     # 5. Commit Action
#     state['action'] = state['curr_pos']  # Blue stays stationary
#     last_sensor_queried = next_action_idx


# --- 6. INITIALIZE AGENTS & SENSORS ---

# Robot

def blue_active_sensing_strategy(state):
    global lstm_hidden, prev_obs_tuple, visual_sensor_queried, is_first_step

    red_pos = ctx.agent.get_agent('robot').current_node_id

    # --- 1. THE BRAIN DECIDES THE QUERY FIRST ---
    if is_first_step:
        # Step 0: Feed Start Token
        obs_input = torch.zeros(1, dtype=torch.long, device=device)
        is_first_step = False
    else:
        # Step 1+: Feed the tuple captured at the END of the previous tick
        try:
            obs_idx = obs_to_idx_dict[prev_obs_tuple]
        except KeyError:
            print(f"WARNING: Tuple {prev_obs_tuple} not found! Defaulting to 0.")
            obs_idx = 0
        obs_input = torch.tensor([obs_idx], dtype=torch.long, device=device)

    # LSTM Forward Pass to pick the NEW action for the CURRENT state
    with torch.no_grad():
        logits, lstm_hidden = policy_net(obs_input, lstm_hidden)
        probs_action = F.softmax(logits, dim=1)
        current_action_idx = torch.multinomial(probs_action, 1).item()

    # --- 2. IMMEDIATELY OBTAIN THE OBSERVATION ---
    # Now that the brain picked the sensor, look at that specific sensor's reading
    # for the current physical reality.
    sensor_id_to_color = {0: 'R', 1: 'G', 2: 'P', 3: 'B'}

    if current_action_idx == 4:
        mapped_obs = '0'
    else:
        sensor_name = f"monitor_{current_action_idx}"
        # GAMMS already computed this data against the current Red position
        raw_obs = state['sensor'][sensor_name][1]['observation']
        mapped_obs = '0' if raw_obs == 'Null' else sensor_id_to_color[raw_obs]

    # --- 3. CONSTRUCT TUPLE & SAVE STATE ---
    # This is the reality of the current step, saved to feed the LSTM next time
    current_obs_tuple = (mapped_obs, current_action_idx)
    print(f"Robot: S{red_pos:<2} | LSTM queried: {current_action_idx} | Tuple Constructed: {current_obs_tuple}")

    # Update globals for the next tick and for the visualizer
    prev_obs_tuple = current_obs_tuple
    visual_sensor_queried = current_action_idx

    # Keep Blue physically stationary
    state['action'] = state['curr_pos']


ctx.agent.create_agent(name='robot', start_node_id=0)
ctx.visual.set_agent_visual(name='robot', color=(255, 0, 0), size=15)
ctx.agent.get_agent('robot').register_strategy(red_stochastic_goal_policy)

# Blue Agent
ctx.agent.create_agent(name='blue_observer', start_node_id=4)
ctx.visual.set_agent_visual(name='blue_observer', color=(0, 0, 255), size=5)
ctx.agent.get_agent('blue_observer').register_strategy(blue_active_sensing_strategy)

# Register Custom Sensors
for action_id, target_node in sensor_targets.items():
    s_name = f"monitor_{action_id}"
    sensor = NodeMonitorSensor(ctx, s_name, target_node, ['robot'])
    ctx.sensor.add_sensor(sensor)
    ctx.agent.get_agent('blue_observer').register_sensor(s_name, sensor)


# --- 7. VISUALIZATION ARTISTS ---


def draw_active_sensor(draw_ctx, data):
    if visual_sensor_queried == 4:
        return

    target_node_id = sensor_targets.get(visual_sensor_queried, None)

    if target_node_id is not None:
        node = draw_ctx.graph.graph.get_node(target_node_id)

        # Yellow Ring = Sensor is Active here
        draw_ctx.visual.render_circle(node.x, node.y, radius=30, color=(255, 255, 0))

        # Flash Green if detected!
        sensor = draw_ctx.sensor.get_sensor(f"monitor_{visual_sensor_queried}")
        if sensor and sensor.data['observation'] != 'Null':
            draw_ctx.visual.render_circle(node.x, node.y, radius=25, color=(0, 255, 0))


query_artist = gamms.visual.Artist(ctx, drawer=draw_active_sensor, layer=50)
ctx.visual.add_artist("query_viz", query_artist)

# --- 8. MAIN LOOP ---
print("Running Autonomous PyTorch-GAMMS Simulation...")
step_counter = 0

while not ctx.is_terminated():
    step_counter += 1

    # A. Update States (Sensors look at current physical reality)
    state_dict = {agent.name: agent.get_state() for agent in ctx.agent.create_iter()}

    # B. Run Strategies (Brain reads current physical state, picks next action)
    for agent in ctx.agent.create_iter():
        agent.strategy(state_dict[agent.name])

    # C. Draw the Frame! (Draws perfectly synchronized state BEFORE robot moves)
    ctx.visual.simulate()
    time.sleep(5.0)

    # D. Apply Actions physically (Robot transitions to the NEXT state)
    for agent in ctx.agent.create_iter():
        agent.set_state()

    # E. Termination Condition
    if step_counter >= 5:
        print(f"Reached the horizon limit. Ending simulation.")
        ctx.terminate()

# Windows file-lock cleanup
try:
    del ctx
    gc.collect()
except Exception:
    pass
