import min_heap
import random
import copy
import csv
import math
import timeit
import matplotlib.pyplot as plot

class DirectedWeightedGraph:

    def __init__(self):
        self.adj = {}
        self.weights = {}
        self.coord = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_node_coord(self, node, x, y):
        self.adj[node] = []
        self.coord[node] = [x, y]

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)


def dijkstra(G, source):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    #Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(source, 0)

    #Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node
    return dist


def dijkstra_approx(G, source, k):
    pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {}  # Distance dictionary
    relax_count = {} # dict to keep track of the relaxations
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    # Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
        relax_count[node] = 0 # before relaxing
    Q.decrease_key(source, 0)

    # Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        for neighbour in G.adj[current_node]:
            relax_count[neighbour] += 1
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour] and relax_count[neighbour] <= k:
                    Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                    dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                    pred[neighbour] = current_node
    return dist

def bellman_ford(G, source):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    nodes = list(G.adj.keys())

    #Initialize distances
    for node in nodes:
        dist[node] = float("inf")
    dist[source] = 0

    #Meat of the algorithm
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                if dist[neighbour] > dist[node] + G.w(node, neighbour):
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
    return dist


def bellman_ford_approx(G, source, k):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    relax_count = {} #dict of count 
    nodes = list(G.adj.keys())

    #Initialize distances
    for node in nodes:
        dist[node] = float("inf")
        relax_count[node] = 0
    dist[source] = 0

    #Meat of the algorithm        
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                relax_count[neighbour] += 1
                if dist[neighbour] > dist[node] + G.w(node, neighbour) and relax_count[neighbour] <= k:
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
    return dist

def total_dist(dist):
    total = 0
    for key in dist.keys():
        total += dist[key]
    return total

def create_random_complete_graph(n,upper):
    G = DirectedWeightedGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i,j,random.randint(1,upper))
    return G
    
def exp1():
    numNodes = 25
    maxRelaxations = 40
    maxRelaxationsList = []
    dijkstraRatio = []
    bellmanFordRatio = []

    G = create_random_complete_graph(numNodes, 30)
    
    for k in range(1, maxRelaxations + 1):  # Starting from k=1 to avoid division by zero
        maxRelaxationsList.append(k)
        G1_copy = copy.deepcopy(G)
        G2_copy = copy.deepcopy(G)

        dijkstra_graph = dijkstra_approx(G1_copy, 0, k)
        dijkstra_actual_dist = total_dist(dijkstra(G1_copy, 0))
        dijkstra_approx_dist = total_dist(dijkstra_graph)
        dijkstra_Ratio = dijkstra_approx_dist / dijkstra_actual_dist
        dijkstraRatio.append(dijkstra_Ratio)

        bellman_ford_graph = bellman_ford_approx(G2_copy, 0, k)
        bellman_actual_dist = total_dist(bellman_ford(G2_copy, 0))
        bellman_approx_dist = total_dist(bellman_ford_graph)
        bellmanFord_Ratio = bellman_approx_dist / bellman_actual_dist
        bellmanFordRatio.append(bellmanFord_Ratio)

    plot.plot(maxRelaxationsList, dijkstraRatio, label='Dijkstra Approximation')
    plot.plot(maxRelaxationsList, bellmanFordRatio, label='Bellman-Ford Approximation')
    plot.legend()
    plot.title('Ratio of Total Distance vs Number of Relaxations')
    plot.xlabel('Number of Relaxations')
    plot.ylabel('Increased Ratio of Total Distance')
    plot.show()
    
def exp2():
    max_nodes = 100
    k = 40
    num_nodes_list = []
    dijkstraRatio = []
    bellmanFordRatio = []

    for num_nodes in range(2, max_nodes + 1):
        num_nodes_list.append(num_nodes)
        G = create_random_complete_graph(num_nodes, 20)
        G1_copy = copy.deepcopy(G)
        G2_copy = copy.deepcopy(G)

        dijkstra_graph = dijkstra_approx(G1_copy, 0, k)
        dijkstra_actual_dist = total_dist(dijkstra(G1_copy, 0))
        dijkstra_approx_dist = total_dist(dijkstra_graph)
        dijkstra_Ratio = dijkstra_approx_dist / dijkstra_actual_dist
        dijkstraRatio.append(dijkstra_Ratio)

        bellman_ford_graph = bellman_ford_approx(G2_copy, 0, k)
        bellman_actual_dist = total_dist(bellman_ford(G2_copy, 0))
        bellman_approx_dist = total_dist(bellman_ford_graph)
        bellmanFord_Ratio = bellman_approx_dist / bellman_actual_dist
        bellmanFordRatio.append(bellmanFord_Ratio)

    plot.plot(num_nodes_list, dijkstraRatio, label='Dijkstra Approximation')
    plot.plot(num_nodes_list, bellmanFordRatio, label='Bellman-Ford Approximation')
    plot.legend()
    plot.title('Ratio of Total Distance vs Number of Nodes')
    plot.xlabel('Number of Nodes')
    plot.ylabel('Ratio of Total Distance')
    plot.show()

def create_random_graph(n, e, upper):
    g = DirectedWeightedGraph()

    for i in range(n):
        g.add_node(i)

    for i in range(e):
        a = random.randint(0, n - 1)
        b = random.randint(0, n - 1)

        while a == b or g.are_connected(a, b):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)

        g.add_edge(a, b, random.randint(1,upper))

    return g

def sparse_vs_dense_experiment():
    max_nodes = 100
    num_nodes_list = []
    dijkstra_sparse_ratio = []
    bellman_ford_sparse_ratio = []
    dijkstra_dense_ratio = []
    bellman_ford_dense_ratio = []

    for num_nodes in range(10, max_nodes + 1):
        num_nodes_list.append(num_nodes)

        # sparse graph with n edges
        sparse_graph = create_random_graph(num_nodes, 6*num_nodes, 10)
        sparse_graph_copy = copy.deepcopy(sparse_graph)

        # dense graph with n*(n-1) edges
        dense_graph = create_random_graph(num_nodes, num_nodes * (num_nodes - 1), 10)
        dense_graph_copy = copy.deepcopy(dense_graph)

        # Dijkstra's and BellmanFord for sparse graph
        dijkstra_sparse_graph = dijkstra_approx(sparse_graph_copy, 0, 20)
        dijkstra_sparse_actual_dist = total_dist(dijkstra(sparse_graph_copy, 0))
        dijkstra_sparse_approx_dist = total_dist(dijkstra_sparse_graph)
        dijkstra_sparse_ratio.append(dijkstra_sparse_approx_dist / dijkstra_sparse_actual_dist)

        bellman_ford_sparse_graph = bellman_ford_approx(sparse_graph_copy, 0, 20)
        bellman_ford_sparse_actual_dist = total_dist(bellman_ford(sparse_graph_copy, 0))
        bellman_ford_sparse_approx_dist = total_dist(bellman_ford_sparse_graph)
        bellman_ford_sparse_ratio.append(bellman_ford_sparse_approx_dist / bellman_ford_sparse_actual_dist)

        # Dijkstra's and BellmanFord for dense graph
        dijkstra_dense_graph = dijkstra_approx(dense_graph_copy, 0, 20)
        dijkstra_dense_actual_dist = total_dist(dijkstra(dense_graph_copy, 0))
        dijkstra_dense_approx_dist = total_dist(dijkstra_dense_graph)
        dijkstra_dense_ratio.append(dijkstra_dense_approx_dist / dijkstra_dense_actual_dist)

        bellman_ford_dense_graph = bellman_ford_approx(dense_graph_copy, 0, 20)
        bellman_ford_dense_actual_dist = total_dist(bellman_ford(dense_graph_copy, 0))
        bellman_ford_dense_approx_dist = total_dist(bellman_ford_dense_graph)
        bellman_ford_dense_ratio.append(bellman_ford_dense_approx_dist / bellman_ford_dense_actual_dist)
    fig, (ax1, ax2) = plot.subplots(1, 2, figsize=(12, 5))
    ax1.plot(num_nodes_list, dijkstra_sparse_ratio, label='Dijkstra Approximation')
    ax1.plot(num_nodes_list, bellman_ford_sparse_ratio, label='Bellman-Ford Approximation')
    ax1.legend()
    ax1.set_title('Sparse Graph - Ratio of Total Distance vs Number of Nodes')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Ratio of Total Distance')

    ax2.plot(num_nodes_list, dijkstra_dense_ratio, label='Dijkstra Approximation')
    ax2.plot(num_nodes_list, bellman_ford_dense_ratio, label='Bellman-Ford Approximation')
    ax2.legend()
    ax2.set_title('Dense Graph - Ratio of Total Distance vs Number of Nodes')
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Ratio of Total Distance')

    plot.tight_layout()
    plot.show()
    
#Assumes G represents its nodes as integers 0,1,...,(n-1)
def mystery(G):
    n = G.number_of_nodes()
    d = init_d(G)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][j] > d[i][k] + d[k][j]: 
                    d[i][j] = d[i][k] + d[k][j]
    return d

def init_d(G):
    n = G.number_of_nodes()
    d = [[float("inf") for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if G.are_connected(i, j):
                d[i][j] = G.w(i, j)
        d[i][i] = 0
    return d


def mystery_exp(max_nodes):
    runs = 30
    avg_times = []
    node_num_list = []

    for node_num in range(1, max_nodes + 1):
        total_time = 0

        for _ in range(runs):
            G = create_random_complete_graph(node_num, 100)

            start = timeit.default_timer()
            mystery(G)
            time_elapsed = timeit.default_timer() - start

            total_time += time_elapsed

        avg_times.append(total_time / runs)
        node_num_list.append(node_num)


    plot.title("Runtime of Mystery Algorithm")
    plot.xlabel("log(Number of nodes)")
    plot.ylabel("log(Run time)")
    plot.loglog(node_num_list, avg_times)
    plot.show()
def a_star(G, s, d, h):
    pred = {}
    dist = {}
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    #Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(s, 0)

    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        if current_node == d:
            return
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour) + h[neighbour])
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node


# Creating graph and heuristic for experiment suite 2
def tube_map():
    g = DirectedWeightedGraph()
    with open('london_stations.csv', 'r') as node:
        csvreader = csv.reader(node)
        next(csvreader)
        for row in csvreader:
            g.add_node_coord(int(row[0]), float(row[1]), float(row[2]))

    with open('london_connections.csv', 'r') as edge:
        csvEdgeReader = csv.reader(edge)
        next(csvEdgeReader)
        for row in csvEdgeReader:
            g.add_edge_line(int(row[0]), int(row[1]), int(row[3]), int(row[2]))
    return g


def heuristic_function(g, t):
    h = {t: 0}
    x1 = g.coord[t][0]
    y1 = g.coord[t][1]
    for node in g.adj:
        x2 = g.coord[node][0]
        y2 = g.coord[node][1]
        h[node] = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return h



def ad_exp1():
    # create graph and heuristic
    g = tube_map()
    time1 = []
    time2 = []

    for node in g.adj:
        h = heuristic_function(g, node)
        start = timeit.default_timer()
        a_star(g, 1, node, h)
        end = timeit.default_timer()
        time1.append(end - start)

        start = timeit.default_timer()
        dijkstra(g, 1)
        end = timeit.default_timer()
        time2.append(end - start)


    plot.plot(time1, label="A*")
    plot.plot(time2, label="Dijkstra")

    plot.legend()
    plot.title("Target node vs. Time")
    plot.xlabel("Target Node")
    plot.ylabel("Time (seconds)")

    plot.show()


def quicksort(adj, h):
    L = []
    for node in adj:
        L.append(node)
    copy = quicksort_copy(L, h)
    for i in range(len(L)):
        L[i] = copy[i]
    return L


def quicksort_copy(L, h):
    if len(L) < 2:
        return L
    pivot = L[0]
    left, right = [], []
    for node in L[1:]:
        if h[node] < h[pivot]:
            left.append(node)
        else:
            right.append(node)
    return quicksort_copy(left, h) + [pivot] + quicksort_copy(right, h)

# time based on distance from target
def ad_exp2():
    # create graph and heuristic
    g = tube_map()
    time1 = []
    dist = []
    time2 = []
    target = 11
    h = heuristic_function(g, target)
    # what happens if we sort nodes based on heuristic... can we see a better relationship?
    L = quicksort(g.adj, h)

    for node in L:
        dist.append(h[node])
        start = timeit.default_timer()
        a_star(g, node, target, h)
        end = timeit.default_timer()
        time1.append(end - start)

        start = timeit.default_timer()
        dijkstra(g, node)
        end = timeit.default_timer()
        time2.append(end - start)

    plot.plot(dist, time1, label="A*")
    plot.plot(dist, time2, label="Dijkstra")

    plot.legend()
    plot.title("Euclidean Distance vs. Time")
    plot.xlabel("Euclidean Distance to Target Node Station " + str(target))
    plot.ylabel("Time (seconds)")

    plot.show()

def ad_exp3():
    # create graph and heuristic
    g = tube_map()
    time1 = []
    time2 = []
    time3 = []
    # pick pairs with lots of transfers

    line1 = [11, 49, 82, 84, 87, 113, 114, 137, 140, 143, 148, 159, 163, 185, 192, 193, 197, 206, 212, 237, 246, 278,
             279, 281, 298]
    line10 = [1, 5, 9, 10, 17, 30, 31, 39, 57, 60, 73, 74, 75, 95, 99, 107, 110, 116, 117, 118, 125, 126, 126, 128,
              130, 131, 132, 133, 134, 145, 146, 151, 160, 176, 182, 187, 190, 194, 197, 210, 220, 222, 223, 232, 234,
              235, 236, 251, 252, 265, 266, 271, 303]
    line13 = [4,13,19,20,27,32,42,43,61,63,64,65,69,70,79,86,97,106,120,135,152,155,
              171,201,203,204,217,219,225,238,247,262,284,292]
    target = 4
    dist = []
    linedist = []
    h = heuristic_function(g, target)
    L = quicksort(g.adj, h)
    for node in L:
        dist.append(h[node])
        start = timeit.default_timer()
        a_star(g, node, target, h)
        end = timeit.default_timer()
        diff = end-start
        time1.append(diff)
        if node in line13:
            time3.append(diff)
            linedist.append(h[node])

        start = timeit.default_timer()
        dijkstra(g, node)
        end = timeit.default_timer()
        time2.append(end - start)

    plot.plot(dist, time1, label="A*")
    plot.plot(dist, time2, label="Dijkstra")
    plot.plot(linedist, time3, label="A* Line 13 Stations")

    plot.legend()
    plot.title("Source Node vs. Time to Station 4")
    plot.xlabel("Euclidean Distance of Source Node to Target")
    plot.ylabel("Time (seconds)")

    plot.show()

def ad_exp4():
    # create graph and heuristic
    g = tube_map()
    time1 = []
    time2 = []
    line1 = [11, 49, 82, 84, 87, 113, 114, 137, 140, 143, 148, 159, 163, 185, 192, 193, 197, 206, 212, 237, 246, 278,
             279, 281, 298]

    target = 11
    h = heuristic_function(g, target)
    L = quicksort(line1, h)
    for node in L:
        start = timeit.default_timer()
        a_star(g, node, target, h)
        end = timeit.default_timer()
        time1.append(end - start)

        start = timeit.default_timer()
        dijkstra(g, node)
        end = timeit.default_timer()
        time2.append(end - start)

    plot.plot(time1, label="A*")
    plot.plot(time2, label="Dijkstra")

    plot.legend()
    plot.title("Source Node vs. Time on Line 1")
    plot.xlabel("Euclidean Distance of Source Node to Target on Line 1")
    plot.ylabel("Time (seconds)")

    plot.show()

def find_min_dist(m):
    if len(m) < 2:
        return m[0]
    mindiff = float("inf")
    for i in range(len(m)-1):
        comp = m[i+1]-m[i]
        if comp < mindiff:
            mindiff = comp
    return mindiff


def bsp_value(m,n):
    last = []
    value = [[0 for _ in range(n+1)] for _ in range((len(m)+1))]
    # row
    for i in range(len(m)+1):
        # col
        for j in range(n+1):
            if j==0 and i>=2:
                value[i][j] = find_min_dist(m[:i])
            elif j >= i-1:
                value[i][j] = math.inf
                # last 2 have bigger gap than current max min gap
            elif j+2 == i:
                l = m[:i]
                print(l)
                value[i][j] = l[-1]-l[0]
            else:
                #optimal list from previous iteration
                last.append(m[i-1])
                l = last
                print(l)
                h = []
                for a in range(i):
                    if a!=(i-2):
                        h.append(m[a])
                print(h)
                print(find_min_dist(h))
                if len(l)>=3:
                    print(min(value[i-1][j], l[-1]-l[-2]))
                    value[i][j] = max(min(value[i-1][j], l[-1]-l[-2]), find_min_dist(h))
                    if find_min_dist(h)>min(value[i-1][j], l[-1]-l[-2]):
                        last = h
            print(value)
    return value[len(m)][n]


def bsp_solution(m, n):
    v = [[m for _ in range(n+1)] for _ in range((len(m)+1))]
    for i in range(len(m)+1):
        # col
        for j in range(n+1):
            if j == 0:
                v[i][j] = m[0:i]
            # can remove all nums
            elif j >= i:
                v[i][j] = []
            # can remove all but end points
            elif j+2 == i:
                v[i][j] = [v[i][0][0], v[i][0][-1]]
            # remove 1 endpoint
            elif j+1 == i:
                v[i][j] = [v[i][0][0]]
            else:
                opt1 = copy.deepcopy(v[i-1][j])
                # first option -- keep what was removed last time -- just add on extra num
                opt1.append(m[:i][-1])
                # or remove second last from previous
                opt2 = copy.deepcopy(v[i-2][j-1])
                opt2.append(m[:i][-1])
                print(str(opt1) + "     " + str(opt2))
                if find_min_dist(opt1) >= find_min_dist(opt2):
                    v[i][j] = opt1
                else:
                    v[i][j] = opt2

    for i in range(len(m)+1):
        line = " "
        for j in range(n+1):
            line += str(v[i][j]) + "     "
        print(line)

    return v[len(m)][n]

ad_exp1()


def bsp_value(L, m):
    solution = bsp_solution(L, m)
    n = len(solution)
    
    distances = [solution[i+1] - solution[i] for i in range(n-1)]
    
    min_distance_index = distances.index(min(distances))
    
    return max(solution[min_distance_index], solution[min_distance_index+1]) - min(solution[min_distance_index], solution[min_distance_index+1])

L = [2 ,7, 10, 14]
m = 0 

max_value = bsp_value(L, m)
print(max_value)

