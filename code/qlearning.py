import data
import models
from itertools import islice
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import time
import networkx as nx
import geopandas as gpd
import pickle
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
import copy
import momepy
import math
import json

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
    # done if no flow in either direction or no path without this edge
    no_path = False
    edges = Static.links_to_edges[remove_this_link] 
    flow_link = flow_day['increasing_order'][remove_this_link] + flow_day['decreasing_order'][remove_this_link]
    no_flow = flow_link == 0
    for u,v in edges:
        if graph.has_edge(u,v):
            graph.remove_edge(u,v)
            for order in ['increasing_order', 'decreasing_order']:
                flow_day_new, no_path_now = redistribute_flow(graph, u,v, flow_day[order], flow_link, k=k)
                flow_day[order][remove_this_link] = flow_day_new
                no_path = no_path or no_path_now

    if no_path: print('no path!')
    if no_flow: print('no flow!')
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
    years = ['2013', '2014', '2015']
    links = gpd.read_file('data/links.json')
    graph = momepy.gdf_to_nx(links, directed=True)
    links_to_edges = {}
    for u,v,_ in graph.edges:
        edges = graph.get_edge_data(u,v)
        for key in edges:
            edge = edges[key]
            if edge['OBJECTID'] not in links_to_edges:
                links_to_edges[edge['OBJECTID']] = []
            links_to_edges[edge['OBJECTID']] += [(u,v)]
    graph = nx.Graph(graph) # convert from multigraph to graph
    weather = data.preprocess_weather(years)
    data_constant = data.prepare_links(links)
    dual_graph = pickle.load(open('data/dual_graph.pkl', 'rb'))
    # links and capacity
    link_to_capacity = dict(zip(links['OBJECTID'], links['Number_Tra']))
    link_to_length = dict(zip(links['OBJECTID'], links['SHAPE_Leng']))
    for link in link_to_capacity:
        if link_to_capacity[link] == None: link_to_capacity[link] = 1
        elif math.isnan(float(link_to_capacity[link])): link_to_capacity[link] = 1
        else: link_to_capacity[link] = int(link_to_capacity[link])

    openstreets = gpd.read_file('data/Open_Streets_Locations.csv')

    mask = np.isin(links['SegmentID'], openstreets['segmentidt'])

    osid_indices = list(links[mask]['OBJECTID'])
    
    # LUCAS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.ScalableRecurrentGCN(node_features = 127, hidden_dim_sequence=[1024,512,768,256,128,64,64]).to(device)
    model.load_state_dict(torch.load('saved_models/best_scalable_rgnn.pt', map_location=device))
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
        self.edges = self.remove_links_from_edges().to(Static.device)
        self.node_features = self.remove_links_from_node_features().to(Static.device)
        self.value, self.total_flow, self.total_probability = self.calculate_value()

    def remove_links_from_flows(self):
        # Subset graph to nodes connected to remaining links and removed links
        graph = Static.graph.copy()
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
        X = data.get_X_day(data_constant, Static.weather, self.flows_day, self.day)
        return torch.tensor(X.values).float().unsqueeze(0)
    
    def calculate_value(self):
        # get total flow
        traffic = calculate_traffic(self.remaining_links, self.flows_day)
#        print('traffic sum:', sum(traffic))
        total_flow = sum(traffic) / 2032838 * 1000 # normalize from random day
        # get total probability of collision
        output = Static.model(self.node_features, self.edges).squeeze()
        total_probability = F.softmax(output, dim=1)[:,1].sum().item() # probability of removing link
#        print('total probability:', total_probability)
        total_probability = total_probability / 7640 * 1000 # normalize from random day
        print('traffic', total_flow)
        print('probability', total_probability)
        return (1-self.tradeoff) * total_flow + self.tradeoff * total_probability, total_flow, total_probability

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
        selected_action = np.random.choice(len(current_state.remaining_links))
    else: # exploit
        q_values = dqn(current_state.node_features, current_state.edges)
        selected_action = q_values.argmax().item()
        print('max', q_values.max())
        print('min', q_values.min())
        print('selected_action:', selected_action)
    return selected_action 

def take_action(current_state, action, return_all=False):
    # Get new date
    current_day = pd.DatetimeIndex([current_state.day])[0]
    next_day = current_day + pd.DateOffset(days=1)
    is_done = next_day.month != current_day.month
    # Make sure don't go past 2015-12-31 because strange formatting in 2016 weather
    is_done = is_done or (current_day.day == 30 and current_day.year == 2015 and current_day.month == 12)
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
    reward = current_state.value - next_state.value + 1 # make more positive
    if return_all:
        return next_state, reward, is_done, next_state.total_flow, next_state.total_probability
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
    # limit loss
    total_loss = total_loss / 1000
    print('loss:', total_loss)
    return total_loss

def subset_flows(flows_month, remaining_links):
    set_remaining_links = set(remaining_links)
    flows_month_new = {}
    for day in flows_month:
        flows_month_new[day] = {}
        for order in ['increasing_order', 'decreasing_order']: 
            flows_month_new[day][order] = {k: v for k, v in flows_month[day][order].items() if k in set_remaining_links}
    return flows_month_new

def new_state(years = ['2013', '2014', '2015'],
              months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']):
    year, month = np.random.choice(years, 1)[0], np.random.choice(months, 1)[0]
    remaining_links = list(Static.links['OBJECTID'])
    day = f'{year}-{month}-01'
    flows_month = pickle.load(open(f'flows/flow_{year}_{month}.pickle', 'rb'))
    flows_month = subset_flows(flows_month, remaining_links)
    return State(day, [], remaining_links, flows_month)

def train_qlearning(num_steps, save_model=True, time_steps=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    update, hard_update, batch_size = 1, 10, 10
    epsilon, epsilon_min, epsilon_decay = 1, .1, 1/250
    gamma = 0.5

    memory = ReplayBuffer(max_size = 1000) 

    # LUCAS
    dqn = models.ConvGraphNet(input_dim = 127, hidden_dim_sequence=[256, 64]).to(device)
    dqn_target = models.ConvGraphNet(input_dim = 127, hidden_dim_sequence=[256, 64]).to(device)
    dqn_target.load_state_dict(dqn.state_dict())
    print(f'Number of parameters: {sum([p.numel() for p in dqn.parameters()])}')

    optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=.5, eps=0)

    scores, losses = [], []
    current_state = new_state()
    done, score = False, 0
    for i in range(num_steps):
        if time_steps:
            start = time.time()
        if done:
            current_state = new_state()
            print('score:', score)
            scores += [score]
            score = 0
    
        action = select_action(current_state, epsilon, dqn) # greedy with 1-epsilon and random with epsilon
        next_state, reward, done = take_action(current_state, action)
        print('reward', reward)
        score += reward
        memory.store({'current_state': current_state, 'next_state': next_state, 'action': action, 'reward': reward, 'done': done})

        if len(memory) > batch_size and i % update == 0:
            batch = memory.sample(batch_size)
            optimizer.zero_grad()
            loss = calculate_loss(batch, dqn, dqn_target, gamma, device) # calculate loss and update weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dqn.parameters(), 1)
            optimizer.step()
            if len(losses) > 10:
                scheduler.step(np.mean(losses[-10:]))
            epsilon -= epsilon_decay
            epsilon = max(epsilon, epsilon_min)

            losses += [loss.item()]

            if (i - batch_size) % hard_update == 0:
                dqn_target.load_state_dict(dqn.state_dict())

        current_state = next_state
        if time_steps and i > 0:
            print(f'Time: {time.time() - start}')
            print()

    print(f'Median: {np.round(np.median(scores), 2)}, Mean: {np.round(np.mean(scores),2)}')
    if save_model:
        torch.save(dqn.state_dict(), 'saved_models/dqn.pt')
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
    output = torch.tensor(traffic).to(Static.device) * (1-tradeoff) + probabilities * tradeoff
    return torch.argmax(output).item()

def select_action_heuristic(current_state, method, dqn=None):
    if method == 'Traffic': return select_traffic(current_state)
    if method == 'Random': return np.random.choice(len(current_state.remaining_links))
    if method == 'Collision': return select_collision(current_state)
    if method == 'traffic_collision': return select_traffic_collision(current_state)
    if method == 'Q Values': return select_action(current_state, 0, dqn)
    if method == 'Open Streets': return open_street_link(current_state.remaining_links)

def open_street_link(remaining_links):
    candidates = []
    for i in range(len(remaining_links)):
        if remaining_links[i] in Static.osid_indices:
            candidates += [i]
    
    return np.random.choice(candidates)
    

def test_RL(dqn, num_steps):
    methods = ['Open Streets', 'Q Values', 'Random']#, 'Collision', 'Traffic']
    scores_compare = {method : [] for method in methods}
    reward_compare = {method : [] for method in methods}
    collision_compare = {method : [] for method in methods}
    traffic_compare = {method : [] for method in methods}
    seeds = list(range(100))
    for seed in seeds:
        np.random.seed(seed)
        for method in methods:
            reward_compare_method = []
            collision_compare_method = []
            traffic_compare_method = []
            print(method)
            current_state = new_state()
            done, score = False, 0
            while not done:
                try:
                    action = select_action_heuristic(current_state, method=method, dqn=dqn)
                    next_state, reward, done, total_flow, total_probability = take_action(current_state, action, return_all=True)
                    score += reward
                    current_state = next_state
                except:
                    print('Excepting...')
                    current_state = new_state() 
                reward_compare_method += [reward]
                collision_compare_method += [total_probability]
                traffic_compare_method += [total_flow]                
            mean = np.round(np.mean(reward_compare_method),2)
            median = np.round(np.median(reward_compare_method),2)
            std = np.round(np.std(reward_compare_method),2)
            scores_compare[method] += [score]
            collision_compare[method] += [np.mean(collision_compare_method)]
            traffic_compare[method] += [np.mean(traffic_compare_method)]
            reward_compare[method] += [np.mean(reward_compare_method)]
            print(f'Method: {method}, Median: {median}, Mean: {mean}, Std: {std}')
    for method in methods:
        mean = np.round(np.mean(reward_compare[method]),2)
        median = np.round(np.median(reward_compare[method]),2)
        std = np.round(np.std(reward_compare[method]),2)
        print(f'Method: {method}, Median: {median}, Mean: {mean}, Std: {std}')
        print(reward_compare[method])
    plot_rl_boxplot(methods, reward_compare, 'Improvement in Traffic Congestion and Safety')
    print('Reward Compare: ', reward_compare)
    print('Collision Compare: ', collision_compare)
    print('Traffic Compare: ', traffic_compare)
    with open('data/rl_reward.json', 'w') as fp:
        json.dump(reward_compare, fp)
    return reward_compare

def plot_rl_boxplot(methods, compare, title):
    import matplotlib
    matplotlib.rcParams.update({'font.size': 12})
    #plt.figure(figsize=(8,4))
    data = np.array([compare[method] for method in methods]).T
    plt.boxplot(data, showfliers=False, labels=methods, medianprops=dict(color='black'))
    plt.title(f'{title} by Heuristics')
    plt.ylabel(f'{title}')
    plt.savefig(f'figures/rl_comparison_boxplot_improvement.pdf')
    plt.clf()

def plot_rl_by_roads(methods, reward_compare, title):
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0,(1,10)), (0, (1,1)), (5,(10,3))]
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    for i, method in enumerate(methods):
        reward_compare_method = np.array(reward_compare[method])
        average = reward_compare_method.mean(axis=0)
        std = reward_compare_method.std(axis=0)
        upper_confidence = average + std
        lower_confidence = average - std 
        xs = list(range(len(average)))
        plt.plot(xs, average, label=method, color=colors[i], linestyle=linestyles[i])
        plt.fill_between(xs, upper_confidence, lower_confidence, color=colors[i], alpha=0.2)

    plt.title(f'{title} by Number of Roads Opened')
    plt.ylabel(f'{title}')
    plt.xlabel('Number of Streets Opened')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/rl_comparison_{title}.pdf')
    plt.clf()

def plot_q_values(dqn):
    current_state = new_state()
    q_values = dqn(current_state.node_features, current_state.edges).detach().cpu().squeeze().numpy()
    q_values = (q_values-q_values.mean())/q_values.std()
    print(q_values.max())
    print(q_values.min())
    indices = np.argsort(-q_values)[:100]
    link_ids = np.array(current_state.remaining_links)[indices]
    print('Top 100 Links in Q Value:')
    print(link_ids)
    new_links = Static.links.copy(deep=True)[Static.links['OBJECTID'].isin(current_state.remaining_links)]
    new_links.plot(column = q_values, cmap = 'viridis', figsize = (10,10), legend = False)
    #plt.title(f'Q-values in Manhattan on {current_state.day}')
    plt.axis('off')
    plt.savefig('figures/q_values.pdf', format="pdf", bbox_inches="tight")
    plt.clf()

def plot_streets(dqn):
    current_state = new_state()
    q_values = dqn(current_state.node_features, current_state.edges).detach().cpu().squeeze().numpy()
    q_values = (q_values-q_values.mean())/q_values.std()
    osid_indices = Static.osid_indices
    num_top = len(osid_indices)
    print(q_values.max())
    print(q_values.min())

    indices = np.argsort(-q_values)[:num_top]
    link_ids = np.array(current_state.remaining_links)[indices]
    print(f'Top {num_top} link_ids in Q Value:')
    print(indices)
    print(f'{num_top} Open Streets indices')
    print(osid_indices)
    new_links = Static.links.copy(deep=True)[Static.links['OBJECTID'].isin(current_state.remaining_links)]
    
    default_color = '#CFE6F3'
    open_street_color = 'red'
    q_val_color = 'blue'

    new_links['colors'] = default_color
    
    new_links.loc[new_links['OBJECTID'].isin(link_ids), 'colors'] = q_val_color

    new_links.loc[new_links['OBJECTID'].isin(osid_indices), 'colors'] = open_street_color

    print(new_links['colors'].value_counts())

    new_links.to_file('new_links.shp')
    new_links.plot(color = new_links['colors'], figsize = (10,10))
    #plt.title(f'Open Streets (Red) and Highest Q-values (Blue) in Manhattan on {current_state.day}')
    plt.axis('off')
    plt.savefig('figures/streets.pdf', format="pdf", bbox_inches="tight")
    plt.clf()
    
    # NOTE: The following code may break depending on your pandas version
    # (!) make sure you run `pip install --upgrade pandas=1.5.2` before running this!
    if pd.__version__ == '1.5.2':
        ## Adding manhattan agreement level plot at neighborhood level
        neighborhoods_file_path = 'data/ntas.shp'
        neighborhoods = gpd.read_file(neighborhoods_file_path)
        manhattan_neighborhoods = neighborhoods[neighborhoods['boro_name'] == 'Manhattan']
        new_links.loc[new_links['colors'] == default_color, 'street_type'] = 0
        new_links.loc[new_links['colors'] == open_street_color, 'street_type'] = 1
        new_links.loc[new_links['colors'] == q_val_color, 'street_type'] = 2
        manhattan_neighborhoods = manhattan_neighborhoods.to_crs(new_links.crs)
        # joined_data = gpd.sjoin(new_links, manhattan_neighborhoods, how='inner', op='intersects')
        joined_data = gpd.sjoin(new_links, manhattan_neighborhoods, how='inner', predicate='intersects')

        # group by the neighborhood and count the segments in each
        segment_counts = joined_data.groupby('ntaname').size()
        # reset the index to make this a df
        segment_counts = segment_counts.reset_index(name='count')

        tag_counts = joined_data.groupby(['ntaname', 'street_type']).size().reset_index(name='count')
        filtered_data = tag_counts[tag_counts['street_type'] != 0]
        tag_counts = filtered_data.groupby(['ntaname', 'street_type']).sum().unstack(fill_value=0)
        
        # actually calculate the agreement 
        # TODO: improve this agreement metric to be directional in (-1, 1)
        tag_counts['agreement'] = 1 - abs(tag_counts['count', 1] - tag_counts['count', 2]) / (tag_counts['count', 1] + tag_counts['count', 2])
        tag_counts['agreement'].fillna(0, inplace=True)

        # create data for plotting
        plot_data = manhattan_neighborhoods.merge(tag_counts, on='ntaname', how='left')
        plot_data.rename(columns={('agreement', ''): 'agreement'}, inplace=True)
        custom_cmap = LinearSegmentedColormap.from_list('grey_to_blue', ['white', 'green'], N=256)

        # plt.figure(figsize=(15, 10))
        ax = plot_data.plot(column='agreement', cmap=custom_cmap, linewidth=0.5, edgecolor='black')
        ax.set_title('Agreement Level in Manhattan Neighborhoods')
        plt.axis('off')
        plt.savefig('figures/agreement.pdf', format="pdf", bbox_inches="tight")

train_dqn = False
## Train DQN
if train_dqn:
    dqn = train_qlearning(num_steps=2000, save_model=True, time_steps=True)

## Test DQN
test_dqn = False
if test_dqn:
    dqn = models.ConvGraphNet(input_dim = 127, hidden_dim_sequence=[256, 64]).to(Static.device)
    dqn.load_state_dict(torch.load('saved_models/dqn.pt', map_location=Static.device))
    dqn.eval()
    for param in dqn.parameters(): param.requires_grad = False
    plot_streets(dqn)
    plot_q_values(dqn)

    test_RL(dqn, num_steps=30)

# Plot comparison
plot_comparison = True
if plot_comparison:
    with open('data/rl_reward.json', 'r') as f:
        reward_compare = json.load(f)
    plot_rl_boxplot(['Q Values', 'Random', 'Open Streets'], reward_compare, 'Improvement in Traffic Congestion and Safety')


## Output
## Method: qlearning, Median: 1.02, Mean: 1.01, Std: 0.07
## Method: traffic_collision, Median: -20.21, Mean: -11.87, Std: 18.33
## Method: collision, Median: -0.86, Mean: -1.84, Std: 5.85
## Method: traffic, Median: -20.21, Mean: -11.83, Std: 18.28
## Method: random, Median: 0.72, Mean: -0.86, Std: 4.14
