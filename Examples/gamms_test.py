# import gamms
#
#
# print("Gamms version: ", gamms.__version__)
import gamms
import time
import random

# --- 1. SETUP CONTEXT ---
ctx = gamms.create_context(vis_engine=gamms.visual.Engine.PYGAME)
graph = ctx.graph.graph

# --- 2. DEFINE THE 5-STATE + TRAP GRAPH ---
nodes = {
    0: {'x': 100, 'y': 300},  # Start
    1: {'x': 300, 'y': 300},  # Target of Sensor 0
    2: {'x': 500, 'y': 300},  # Target of Sensor 1
    3: {'x': 700, 'y': 300},  # Target of Sensor 2
    4: {'x': 900, 'y': 300},  # Goal. Target of Sensor 3
    5: {'x': 500, 'y': 500},  # Trap State
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

# Dictionary to map action index to physical node ID
sensor_targets = {0: 1, 1: 2, 2: 3, 3: 4}


# --- 3. THE CUSTOM SENSOR ---
@ctx.sensor.custom(name="NODE_MONITOR")
class NodeMonitorSensor(gamms.typing.ISensor):
    def __init__(self, ctx, sensor_id, target_node_id, red_team_names):
        self.ctx = ctx
        self._sensor_id = sensor_id
        self.target_node_id = target_node_id
        self.red_team_names = red_team_names

        # Initialize data dictionary. Default observation is 'Null'
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
        # Default to 'Null' every tick
        self._data['observation'] = 'Null'

        for red_name in self.red_team_names:
            red_agent = self.ctx.agent.get_agent(red_name)
            if red_agent.current_node_id == self.target_node_id:
                # If the robot is on our target node, return the sensor's own ID
                self._data['observation'] = int(self._sensor_id.split('_')[-1])
                break

    def update(self, data: dict):
        pass


# --- 4. AGENT STRATEGIES ---

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


# Global variable to hold our last sensor query decision (starts at 4 = Sleep)
last_sensor_queried = 4


# --- 5. INITIALIZE SENSORS & AGENTS ---

# Robot

def blue_active_sensing_strategy(state):
    global last_sensor_queried

    # --- 1. THE BRAIN DECIDES THE QUERY FIRST ---
    # Here is where your LSTM looks at its current belief
    # and decides which sensor to turn on right NOW.

    # Placeholder: randomly pick an action including 4 (Sleep)
    action_idx = random.choice([0, 1, 2, 3, 4])

    # --- 2. IMMEDIATELY OBTAIN THE OBSERVATION ---
    # Because GAMMS pre-computed the sensors at the start of the tick,
    # we can instantly read the result for the current physical state.
    if action_idx == 4:
        observation_value = 'Null'
    else:
        sensor_name = f"monitor_{action_idx}"
        observation_value = state['sensor'][sensor_name][1]['observation']

    # Fetch actual red agent position just for our console debugging
    red_pos = ctx.agent.get_agent('robot').current_node_id
    print(
        f"Robot is at Node {red_pos:<2} | LSTM queried sensor {action_idx} | Observation received: {observation_value}")

    # --- 3. COMMIT ACTIONS ---
    # Save the decision so the PyGame visualizer knows where to draw the yellow ring
    last_sensor_queried = action_idx

    # GAMMS FIX: Keep the Blue Agent physically stationary
    state['action'] = state['curr_pos']


ctx.agent.create_agent(name='robot', start_node_id=0)
ctx.visual.set_agent_visual(name='robot', color=(255, 0, 0), size=15)
ctx.agent.get_agent('robot').register_strategy(red_stochastic_goal_policy)

# Blue Agent
ctx.agent.create_agent(name='blue_observer', start_node_id=4)
ctx.visual.set_agent_visual(name='blue_observer', color=(0, 0, 255), size=5)
ctx.agent.get_agent('blue_observer').register_strategy(blue_active_sensing_strategy)

# Create and Register Sensors to Blue Agent
for action_id, target_node in sensor_targets.items():
    s_name = f"monitor_{action_id}"
    sensor = NodeMonitorSensor(ctx, s_name, target_node, ['robot'])
    ctx.sensor.add_sensor(sensor)
    ctx.agent.get_agent('blue_observer').register_sensor(s_name, sensor)


# --- 6. VISUALIZATION ARTIST ---
def draw_active_sensor(ctx, data):
    # If Blue chose to sleep (Action 4), draw nothing!
    if last_sensor_queried == 4:
        return

    target_node_id = sensor_targets.get(last_sensor_queried, None)

    if target_node_id is not None:
        node = ctx.graph.graph.get_node(target_node_id)

        # Draw Yellow Ring for the active sensor query
        ctx.visual.render_circle(node.x, node.y, radius=30, color=(255, 255, 0))

        # Check if the active sensor got a hit
        sensor = ctx.sensor.get_sensor(f"monitor_{last_sensor_queried}")
        if sensor and sensor.data['observation'] != 'Null':
            # Flash Green inside the yellow ring
            ctx.visual.render_circle(node.x, node.y, radius=25, color=(0, 255, 0))


query_artist = gamms.visual.Artist(ctx, drawer=draw_active_sensor, layer=50)

# # --- ADD NODE LABELS ---
# def draw_node_labels(ctx, data):
#     for nid in [0, 1, 2, 3, 4, 5]:
#         node = ctx.graph.graph.get_node(nid)
#         # Offset the text slightly above the node (y - 30)
#         ctx.visual.render_text(text=f"S{nid}", x=node.x - 15, y=node.y - 30, color=(255, 255, 255))
#
#
# label_artist = gamms.visual.Artist(ctx, drawer=draw_node_labels, layer=100)
# ctx.visual.add_artist("labels_viz", label_artist)

ctx.visual.add_artist("query_viz", query_artist)

# --- 7. MAIN LOOP ---
print("Running Rigorous GAMMS Active Sensing Demo...")
step_counter = 0

while not ctx.is_terminated():
    step_counter += 1

    # A. Update States
    state_dict = {agent.name: agent.get_state() for agent in ctx.agent.create_iter()}

    # B. Run Strategies
    for agent in ctx.agent.create_iter():
        agent.strategy(state_dict[agent.name])

    # C. Draw
    ctx.visual.simulate()
    time.sleep(5.0)  # 1 second delay so you can easily read the console logs

    # D. Apply Actions physically
    for agent in ctx.agent.create_iter():
        agent.set_state()

    # E. Termination Condition
    if step_counter >= 5:
        print(f"Reached the horizon limit. Ending simulation.")
        ctx.terminate()

# ctx.terminate()
