import networkx as nx
import numpy as np
from queue import PriorityQueue
import random
import math

def total_weight_of_valid_paths(graph, valid_paths, weight_key):
    total_weight = 0
    for i in range(len(valid_paths) - 1):
        edge_weight = graph[valid_paths[i]][valid_paths[i + 1]][weight_key]
        total_weight += edge_weight
    return total_weight

def custom_dijkstra(graph, source, target, weight_key, max_bandwidth):
    paths = nx.single_source_dijkstra_path(graph, source, weight=weight_key)
    possible_paths = paths.get(target, [])
    weights = [graph[possible_paths[i]][possible_paths[i + 1]][weight_key] for i in range(len(possible_paths) - 1)]
    valid_paths = [possible_paths[i] for i in range(len(possible_paths)) if i == len(possible_paths) - 1 or weights[i] >= max_bandwidth]
    return valid_paths

def custom_bellman_ford(graph, source, target, weight_key, max_bandwidth):
    distances = {node: float('inf') for node in graph.nodes}
    distances[source] = 0
    for _ in range(graph.number_of_nodes() - 1):
        for edge in graph.edges(data=True):
            u, v, data = edge
            weight = data.get(weight_key, 1)
            if distances[u] + weight >= distances[v]:
                continue
            distances[v] = distances[u] + weight
    possible_paths = nx.single_source_dijkstra_path(graph, source, weight=weight_key).get(target, [])
    weights = [graph[possible_paths[i]][possible_paths[i + 1]][weight_key] for i in range(len(possible_paths) - 1)]
    valid_paths = [possible_paths[i] for i in range(len(possible_paths)) if i == len(possible_paths) - 1 or weights[i] >= max_bandwidth]
    return valid_paths

def custom_a_star(graph, source, target, weight_key, max_bandwidth):
    priority_queue = PriorityQueue()
    priority_queue.put((0, source))
    visited = set()
    visited.add(source)
    while not priority_queue.empty():
        current_cost, current_node = priority_queue.get()
        if current_node == target:
            break
        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor].get(weight_key, 1)
            if neighbor not in visited:
                visited.add(neighbor)
                priority = current_cost + weight 
                priority_queue.put((priority, neighbor))
    shortest_path = nx.shortest_path(graph, source=source, target=target, weight=weight_key)
    valid_paths = [shortest_path[i] for i in range(len(shortest_path)) if i == len(shortest_path) - 1 or graph[shortest_path[i]][shortest_path[i + 1]][weight_key] >= max_bandwidth]
    return valid_paths

def calculate_path_weight(graph, path, weight_key):
    weights = [graph[path[i]][path[i + 1]][weight_key] for i in range(len(path) - 1)]
    return sum(weights)

def simulated_annealing(graph, source, target, weight_key, max_bandwidth, initial_solution=None, temperature=1000, cooling_rate=0.95, num_iterations=1000):
    current_solution = initial_solution if initial_solution else custom_dijkstra(graph, source, target, weight_key, max_bandwidth)
    current_energy = calculate_path_weight(graph, current_solution, weight_key)
    best_solution = current_solution
    best_energy = current_energy
    for _ in range(num_iterations):
        temperature *= cooling_rate
        neighbor_solution = custom_dijkstra(graph, source, target, weight_key, max_bandwidth)
        neighbor_energy = calculate_path_weight(graph, neighbor_solution, weight_key)
        if neighbor_energy < current_energy or random.random() < math.exp((current_energy - neighbor_energy) / temperature):
            current_solution = neighbor_solution
            current_energy = neighbor_energy
        if current_energy < best_energy:
            best_solution = current_solution
            best_energy = current_energy
    return best_solution

# main
adjustment_matrix = np.loadtxt("input.txt",delimiter=':')
bandwidth_matrix = np.loadtxt("bandwith.txt",delimiter=':')

for i in range(len(adjustment_matrix)):
    for j in range(i + 1, len(adjustment_matrix[i])):
        if bandwidth_matrix[i, j] < 5:
            adjustment_matrix[i, j] = 0
            adjustment_matrix[j, i] = 0

G = nx.Graph()
for i in range(len(adjustment_matrix)):
    for j in range(i + 1, len(adjustment_matrix[i])):
        if adjustment_matrix[i, j] > 0:
            weight = bandwidth_matrix[i, j]
            G.add_edge(i + 1, j + 1, weight=weight)

G2 = nx.Graph()
for i in range(len(adjustment_matrix)):
    for j in range(i + 1, len(adjustment_matrix[i])):
        if adjustment_matrix[i, j] > 0:
            weight = adjustment_matrix[i, j]
            G2.add_edge(i + 1, j + 1, weight=weight)

max_bandwidth = 5
valid_shortest_path_dijkstra = custom_dijkstra(G, source=1, target=24, weight_key='weight', max_bandwidth=max_bandwidth)
total_weight_dijkstra = total_weight_of_valid_paths(G2, valid_shortest_path_dijkstra, weight_key='weight')
result_dijkstra = total_weight_dijkstra * max_bandwidth

valid_shortest_path_bellman_ford = custom_bellman_ford(G, source=1, target=24, weight_key='weight', max_bandwidth=max_bandwidth)
total_weight_bellman_ford = total_weight_of_valid_paths(G2, valid_shortest_path_bellman_ford, weight_key='weight')
result_bellman_ford = total_weight_bellman_ford * max_bandwidth

valid_shortest_path_a_star = custom_a_star(G, source=1, target=24, weight_key='weight', max_bandwidth=max_bandwidth)
total_weight_a_star = total_weight_of_valid_paths(G2, valid_shortest_path_a_star, weight_key='weight')
result_a_star = total_weight_a_star * max_bandwidth

valid_shortest_path_simulated_annealing = simulated_annealing(G, source=1, target=24, weight_key='weight', max_bandwidth=max_bandwidth)
total_weight_simulated_annealing = total_weight_of_valid_paths(G2, valid_shortest_path_simulated_annealing, weight_key='weight')
result_simulated_annealing = total_weight_simulated_annealing * max_bandwidth


print("valid_shortest_path_dijkstra:", valid_shortest_path_dijkstra, "distance:", total_weight_dijkstra, "distance*bandwidth:", result_dijkstra)
print("valid_shortest_path_bellman_ford:", valid_shortest_path_bellman_ford, "distance:", total_weight_bellman_ford, "distance*bandwidth:", result_bellman_ford)
print("valid_shortest_path_a_star:", valid_shortest_path_a_star, "distance:", total_weight_a_star, "distance*bandwidth:", result_a_star)
print("valid_shortest_path_simulated_annealing:", valid_shortest_path_simulated_annealing, "distance:", total_weight_simulated_annealing, "distance*bandwidth:", result_simulated_annealing)
