import min_heap
import random
import csv
import math
import timeit
import matplotlib.pyplot as plot

class DirectedWeightedGraph:

    def __init__(self):
        self.adj = {}
        self.weights = {}
        self.coord = {}
        self.line = {}

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
        self.line[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def add_edge_line(self, node1, node2, weight, line):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        if line not in self.line[node1]:
            self.line[node1].append(line)
        if line not in self.line[node2]:
            self.line[node2].append(line)
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
        a_star(g, 100, node, h)
        end = timeit.default_timer()
        time1.append(end - start)

        start = timeit.default_timer()
        dijkstra(g, 100)
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


def ad_exp2():
    # create graph and heuristic
    g = tube_map()
    time1 = []
    dist = []
    time2 = []
    target = 4
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
    L = []
    L2 = []
    g = tube_map()
    time1 = []
    time2 = []
    h = heuristic_function(g, 4)

    # station 4 part of 1 line
    for node in g.adj:
        if g.line[4][0] in g.line[node]:
            L.append(node)
        else:
            L2.append(node)

    for node in L:
        start = timeit.default_timer()
        a_star(g, node, 4, h)
        end = timeit.default_timer()
        time1.append(end - start)

        start = timeit.default_timer()
        dijkstra(g, node)
        end = timeit.default_timer()
        time2.append(end - start)

    plot.plot(time1, label="A*")
    plot.plot(time2, label="Dijkstra")

    plot.legend()
    plot.title("Source Node vs. Time")
    plot.xlabel("Source Node")
    plot.ylabel("Time (seconds)")

    plot.show()


ad_exp2()



