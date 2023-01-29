from data import *
from models import *
from itertools import islice
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# Helper functions for the state
def k_shortest_paths(graph, source, target, k):
    return list(
        islice(nx.shortest_simple_paths(graph, source, target, weight='expected_time'), k)
    )

def normalize_weights(weights):
    # Invert, as we want shortest expected time to be most likely
    weights = [(1.0 / weight) for weight in weights]
    # Standard normalization, weights over sum
    sum_weights = sum(weights)
    return [weight / sum_weights for weight in weights]

def redistribute_flow(graph, source, target, flow_day, flow_link, k=5):
    if not nx.has_path(graph, source, target):
        return flow_day, True

    weights, paths = [], []
    for path in k_shortest_paths(graph, source, target, k):
        weight = 0
        for i in range(len(path)-1):
            weight += graph[path[i]][path[i+1]]['expected_time']
        weights.append(weight)
        paths.append(path)

    weights_norm = normalize_weights(weights)
    for path, weight in zip(paths, weights_norm):
        for i in range(len(path)-1):
            current_node = path[i]
            next_node = path[i+1]
            edge = graph.edges[(current_node, next_node)]['OBJECTID']
            flow_day[edge] += weight * flow_link
    
    return flow_day, False

def remove_one_link(remove_this_link, flow_day, graph, k=5):
    node1 = Static.links[Static.links['OBJECTID'] == remove_this_link]['NodeIDFrom'].values[0]
    node2 = Static.links[Static.links['OBJECTID'] == remove_this_link]['NodeIDTo'].values[0]

    # done if no flow in either direction or no path without this edge
    no_path = False
    no_flow = True

    for source, target in [(node1, node2), (node2, node1)]:
        # Calculate flow on link
        order = 'increasing_order' if source < target else 'decreasing_order'
        flow_link = flow_day[order][remove_this_link]
        if flow_link > 0: no_flow = False
        # Update flow
        if graph.has_edge(source, target) and flow_link != 0:
            graph.remove_edge(source, target)
            flow_day_new, no_path_now = redistribute_flow(graph, source, target, flow_day[order], flow_link)
            no_path = no_path or no_path_now
            flow_day[order] = flow_day_new

    return flow_day, graph, no_path or no_flow

def calculate_traffic(remaining_links, flows_day):
    traffic = []
    for link in sorted(remaining_links):
        total_flow = 0
        flow_on_link1 = flows_day['increasing_order'][link]
        flow_on_link2 = flows_day['decreasing_order'][link]
        capacity = Static.link_to_capacity[link]
        length = Static.link_to_length[link]
        # Get density of traffic per lane
        if flow_on_link1 * flow_on_link2 > 0: # assume half traffic lanes in each direction
            total_flow += flow_on_link1 / (capacity / 2 * length) + flow_on_link2 / (capacity / 2 * length)
        elif flow_on_link1 != 0:
            total_flow += flow_on_link1 / (capacity * length)
        elif flow_on_link2 != 0:
            total_flow += flow_on_link2 / (capacity * length)
        traffic += [total_flow]
    return traffic

class Static:
    # Things we only need to load once
    years = ['2013', '2014', '2015', '2016']
    links = gpd.read_file('data/links.json')        
    graph = get_directed_graph(links)
    collisions = gpd.read_file('data/collisions_2013.json')
    weather = preprocess_weather(years)
    data_constant = prepare_links(links)
    dual_graph = pickle.load(open('data/dual_graph.pkl', 'rb'))
    # links and capacity
    link_to_capacity = dict(zip(links['OBJECTID'], links['Number_Tra']))
    link_to_length = dict(zip(links['OBJECTID'], links['SHAPE_Leng']))
    for link in link_to_capacity:
        if link_to_capacity[link] == None: link_to_capacity[link] = 1
        else: link_to_capacity[link] = int(link_to_capacity[link])
    
    # LUCAS
    model = RecurrentGCN(node_features = 127)
    model.load_state_dict(torch.load('saved_models/rgnn.pt'))
    model.eval()
    for p in model.parameters(): p.requires_grad = False

class State:
    def __init__(self, day, removed_links, remaining_links, flows_month, tradeoff=.5):
        self.tradeoff = tradeoff
        self.day = day
        self.removed_links = removed_links
        self.flows_month = flows_month
        self.remaining_links = [x for x in remaining_links if x not in removed_links]
        # done if flows are 0 or if there is no path without the removed links
        self.flows_day, self.is_done = self.remove_links_from_flows()        
        self.edges = self.remove_links_from_edges()
        self.node_features = self.remove_links_from_node_features()
        self.value = self.calculate_value()

    def remove_links_from_flows(self):
        # Subset graph to nodes connected to remaining links and removed links
        subsetted = Static.links[Static.links['OBJECTID'].isin(self.remaining_links + self.removed_links)]
        nodes_to = subsetted['NodeIDTo']
        nodes_from  = subsetted['NodeIDFrom']
        nodes_unique = np.unique([nodes_to, nodes_from])
        graph = Static.graph.subgraph(nodes_unique).copy()
        flow_day = self.flows_month[str(self.day)]
        is_done = False
        for remove_this_link in self.removed_links:
            flow_day, graph, is_done_now = remove_one_link(remove_this_link, flow_day, graph)
            is_done = is_done or is_done_now
        flow_day_remaining = {}
        for order in ['increasing_order', 'decreasing_order']:
            flow_day_remaining[order] = {k: v for k, v in flow_day[order].items() if k not in self.removed_links}
        return flow_day_remaining, is_done
        
    def remove_links_from_edges(self): 
        # We could use from torch_geometric.utils.convert import from_networkx
        # to convert the graph to a torch_geometric.data.Data object
        # The problem is that it doesn't preserve the node order so we'd need to
        # add the data to the networkx graph and
        # the best way seems like using set_node_attributes which takes a dictionary
        # and turning pandas dataframe into a dictionary takes way longer than relabeling
        dual_graph = Static.dual_graph.subgraph(self.remaining_links).copy()
        assert 0 not in dual_graph.nodes # check we're not already relabeled
        dual_graph = nx.convert_node_labels_to_integers(dual_graph, ordering='sorted')
        assert 0 in dual_graph.nodes # check the relabeling worked        
        return torch.tensor(np.array(list(dual_graph.edges))).long().T

    def remove_links_from_node_features(self):
        data_constant = Static.data_constant[Static.data_constant['OBJECTID'].isin(self.remaining_links)]
        X = get_X_day(data_constant, Static.weather, self.flows_day, self.day)
        return torch.tensor(X.values).float().unsqueeze(0)
    
    def calculate_value(self):
        # get total flow
        traffic = calculate_traffic(self.remaining_links, self.flows_day)
        total_flow = sum(traffic) / 2335000 * 100000 # normalize from random day
        # get total probability of collision
        output = Static.model(self.node_features, self.edges).squeeze()
        total_probability = F.softmax(output, dim=1)[:,1].sum().item() # probability of removing link
        total_probability = total_probability / 9654 * 100000 # normalize from random day
        print('traffic', total_flow)
        print('probability', total_probability)
        return (1-self.tradeoff) * total_flow + self.tradeoff * total_probability

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.names = ['current_state', 'next_state', 'action', 'reward', 'done']
        self.buffer = {name: [] for name in self.names}

    def store(self, dictionary):
        for name in self.names:
            if len(self.buffer[name]) == self.max_size:
                self.buffer[name].pop(0)
            self.buffer[name].append(dictionary[name])

    def sample(self, batch_size):
        indices = np.random.choice(len(self), batch_size, replace=False)
        return {name : [self.buffer[name][i] for i in indices] for name in self.names}
    
    def __len__(self):
        return len(self.buffer['current_state'])

# Helper functions for trajectory

def select_action(current_state, epsilon, dqn):
    if np.random.random() < epsilon: # explore
        if np.random.random() < 0.5:
            selected_action = np.random.choice(len(current_state.remaining_links))
        else:
            selected_action = select_traffic_collision(current_state)
    else: # exploit
        q_values = dqn(current_state.node_features, current_state.edges)
        selected_action = q_values.argmax().item()
    return selected_action 

def take_action(current_state, action):
    # Get new date
    current_day = pd.DatetimeIndex([current_state.day])[0]
    next_day = current_day + pd.DateOffset(days=1)
    is_done = next_day.month != current_day.month
    next_day_str = next_day.strftime('%Y-%m-%d')
    # Convert index to action
    action = sorted(current_state.remaining_links)[action]
    next_removed_links = current_state.removed_links + [action]
    # Get new flows if necessary
    if is_done:
        year, month = next_day.year, str(next_day.month).zfill(2)
        flows_month = pickle.load(open(f'flows/flow_{year}_{month}.pickle', 'rb'))
    else:
        flows_month = current_state.flows_month 
    # Initialize new state
    next_state = State(next_day_str, next_removed_links, current_state.remaining_links, flows_month)
    # Check if done because of new month or some other way
    is_done = is_done or next_state.is_done
    # Calculate reward
    reward = current_state.value - next_state.value
    return next_state, reward, is_done

def calculate_loss(batch, dqn, dqn_target, gamma, device):
    # Get batch
    current_states = batch['current_state']
    next_states = batch['next_state']
    actions = torch.tensor(batch['action'])
    rewards = torch.tensor(batch['reward'])
    dones = torch.tensor(batch['done'])
    # Forward pass
    # Every state is a different sized graph so we need loop :(
    total_loss = 0
    for i in range(len(batch['current_state'])):
        current_q = dqn(
            current_states[i].node_features.to(device), 
            current_states[i].edges.to(device)
        ).squeeze()[actions[i]]
        with torch.no_grad():
            max_next_q = dqn_target(
                next_states[i].node_features.to(device),
                next_states[i].edges.to(device)
            ).max()
        target = rewards[i] + gamma * max_next_q * ~(dones[i])
        total_loss += F.smooth_l1_loss(current_q, target)

    return total_loss

def subset_flows(flows_month, remaining_links):
    set_remaining_links = set(remaining_links)
    flows_month_new = {}
    for day in flows_month:
        flows_month_new[day] = {}
        for order in ['increasing_order', 'decreasing_order']: 
            flows_month_new[day][order] = {k: v for k, v in flows_month[day][order].items() if k in set_remaining_links}
    return flows_month_new

def new_state(big_strong_components=None, years = ['2013', '2014', '2015'],
              months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']):
    year, month = np.random.choice(years, 1)[0], np.random.choice(months, 1)[0]
    if big_strong_components != None:
        remaining_links = big_strong_components[np.random.choice(len(big_strong_components))]
    else:
        remaining_links = list(Static.links['OBJECTID'])
    day = f'{year}-{month}-01'
    flows_month = pickle.load(open(f'flows/flow_{year}_{month}.pickle', 'rb'))
    flows_month = subset_flows(flows_month, remaining_links)
    return State(day, [], remaining_links, flows_month)

def train_qlearning(num_steps, save_model=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    update, hard_update, batch_size = 10, 100, 10
    epsilon, epsilon_min, epsilon_decay = 1, .5, 1/20000
    gamma = 0.99

    memory = ReplayBuffer(max_size = 1000) 

    # LUCAS
    dqn = ConvGraphNet(input_dim = 127).to(device)
    dqn_target = ConvGraphNet(input_dim = 127).to(device)
    dqn_target.load_state_dict(dqn.state_dict())

    optimizer = torch.optim.Adam(dqn.parameters(), lr=0.001)

    def warmup(current_step, warmup_steps=2):
        if current_step < warmup_steps:
            return float(current_step / warmup_steps)
        else:                                 
            return max(0.0, float(num_steps - current_step) / float(max(1, num_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

    # Get good subsets to run on
    big_strong_components = []
    for nodes in nx.strongly_connected_components(Static.dual_graph):
        if len(nodes) >= 1000: big_strong_components += [list(nodes)]

    scores, losses = [], []
    done, score = True, 0
    for i in range(num_steps):
        if done:
            current_state = new_state(months=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11'])
            scores += [score]
            score = 0
    
        action = select_action(current_state, epsilon, dqn) # greedy with 1-epsilon and random with epsilon
        next_state, reward, done = take_action(current_state, action)
        print('reward', reward)
        score += reward
        memory.store({'current_state': current_state, 'next_state': next_state, 'action': action, 'reward': reward, 'done': done})

        if len(memory) > batch_size and i % update == 0:
            batch = memory.sample(batch_size)
            loss = calculate_loss(batch, dqn, dqn_target, gamma, device) # calculate loss and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epsilon -= epsilon_decay
            epsilon = max(epsilon, epsilon_min)

            losses += [loss]

            if (i - batch_size) % hard_update == 0:
                dqn_target.load_state_dict(dqn.state_dict())

        current_state = next_state

    print(f'Median: {np.round(np.median(scores), 2)}, Mean: {np.round(np.mean(scores),2)}')
    if save_model:
        torch.save(dqn.state_dict(), 'saved_models/new_dqn.pt')
    return dqn


def select_traffic(current_state):
    traffic = calculate_traffic(current_state.remaining_links, current_state.flows_day)
    return max(enumerate(traffic), key=lambda x: x[1])[0]

def select_collision(current_state):
    probabilities = Static.model(current_state.node_features, current_state.edges).squeeze()[:,1]
    return torch.argmax(probabilities).item()

def select_traffic_collision(current_state):
    tradeoff = current_state.tradeoff
    traffic = calculate_traffic(current_state.remaining_links, current_state.flows_day)
    probabilities = Static.model(current_state.node_features, current_state.edges).squeeze()[:,1]
    output = torch.tensor(traffic) * (1-tradeoff) + probabilities * tradeoff
    return torch.argmax(output).item()

def select_action_heuristic(current_state, method, dqn=None):
    if method == 'traffic': return select_traffic(current_state)
    if method == 'random': return np.random.choice(len(current_state.remaining_links))
    if method == 'collision': return select_collision(current_state)
    if method == 'traffic_collision': return select_traffic_collision(current_state)
    if method == 'qlearning': return select_action(current_state, 0, dqn)

def test_RL(dqn, num_trajectories):
    scores_compare = {}
    for method in ['traffic', 'random', 'collision', 'traffic_collision', 'qlearning']:
        print(method)
        scores_compare[method] = []
        done, score = True, 0
        while len(scores_compare[method]) < num_trajectories:
            if done:
                current_state = new_state(months=['12'])
                scores_compare[method] += [score]
                print(score)
                score = 0
            
            action = select_action_heuristic(current_state, method=method, dqn=dqn)
            next_state, reward, done = take_action(current_state, action)
            score += reward

            current_state = next_state
        mean = np.round(np.mean(scores_compare[method]),2)
        median = np.round(np.median(scores_compare[method]),2)
        std = np.round(np.std(scores_compare[method]),2)
        print(f'Method: {method}, Median: {median}, Mean: {mean}, Std: {std}')
        print(scores_compare[method])
    return scores_compare

def plot_q_values(dqn):
    current_state = new_state(years=['2013', '2014', '2015'], months=['12'])
    q_values = dqn(current_state.node_features, current_state.edges).detach().squeeze().numpy()
    indices = np.argsort(q_values)[:100]
    link_ids = np.array(current_state.remaining_links)[indices]
    print('Top 100 Links in Q Value:')
    print(link_ids)
    min_q = np.min(q_values)
    if min_q <= 0:
        q_values = q_values - min_q + 1
    new_links = Static.links.copy(deep=True)[Static.links['OBJECTID'].isin(current_state.remaining_links)]
    new_links.set_index('OBJECTID', inplace=True)
    new_links.sort_index(inplace=True)
    new_links['q_values'] = q_values
    new_links.plot(column = q_values, cmap = 'viridis', figsize = (10,10), legend = True,
                norm=colors.LogNorm(vmin=q_values.min(), vmax=q_values.max()))
    plt.savefig('figures/q_values.pdf', format="pdf", bbox_inches="tight")


dqn = train_qlearning(num_steps=40000, save_model=True)
# LUCAS
#dqn = ConvGraphNet(input_dim = 127)
#dqn.load_state_dict(torch.load('saved_models/dqn.pt'))
#dqn.eval()
#for param in dqn.parameters(): param.requires_grad = False
plot_q_values(dqn)
test_RL(dqn, num_trajectories=100)
