import copy
from collections import deque
import random
import matplotlib.pyplot as plot

#Undirected graph using an adjacency list
class Graph:

    def __init__(self, n):
        self.adj = {}
        for i in range(n):
            self.adj[i] = []

    def are_connected(self, node1, node2):
        return node2 in self.adj[node1]

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self):
        self.adj[len(self.adj)] = []

    def add_edge(self, node1, node2):
        if node1 not in self.adj[node2]:
            self.adj[node1].append(node2)
            self.adj[node2].append(node1)

    def number_of_nodes(self):
        return len(self.adj)


# Breadth First Search
def BFS(G, node1, node2):
    Q = deque([node1])
    marked = {node1 : True}
    for node in G.adj:
        if node != node1:
            marked[node] = False
    while len(Q) != 0:
        current_node = Q.popleft()
        for node in G.adj[current_node]:
            if node == node2:
                return True
            if not marked[node]:
                Q.append(node)
                marked[node] = True
    return False


def BFS2(G, node1, node2):
    nodelist = []
    Q = deque([node1])
    marked = {node1 : True}
    for node in G.adj:
        if node != node1:
            marked[node] = False
    while len(Q) != 0:
        current_node = Q.popleft()
        nodelist.append(current_node)
        for i in range(len(nodelist)-1):
            if not G.are_connected(nodelist[i],nodelist[i+1]):
                pos = i+1
                while len(nodelist)>pos:
                    nodelist.remove(nodelist[pos])
                break

        for node in G.adj[current_node]:
            if node == node2:
                nodelist.append(node)
                return nodelist
            if not marked[node]:
                Q.append(node)
                marked[node] = True
    return []


def BFS3(G, node1):
    pre_dict = {}
    Q = deque([node1])
    marked = {node1 : True}
    for node in G.adj:
        if node != node1:
            marked[node] = False
    while len(Q) != 0:
        current_node = Q.popleft()
        for node in G.adj[current_node]:
            if not marked[node]:
                pre_dict[node]=current_node
                Q.append(node)
                marked[node] = True
    return pre_dict


# Depth First Search
def DFS(G, node1, node2):
    S = [node1]
    marked = {}
    for node in G.adj:
        marked[node] = False
    while len(S) != 0:
        current_node = S.pop()
        if not marked[current_node]:
            marked[current_node] = True
            for node in G.adj[current_node]:
                if node == node2:
                    return True
                S.append(node)
    return False


# Depth First Search
def DFS2(G, node1, node2):
    nodelist=[]
    S = [node1]
    marked = {}
    for node in G.adj:
        marked[node] = False
    while len(S) != 0:
        current_node = S.pop()
        if not marked[current_node]:
            marked[current_node] = True
            nodelist.append(current_node)
            for node in G.adj[current_node]:
                if node == node2:
                    nodelist.append(node)
                    return nodelist
                S.append(node)
    return []


# Depth First Search
def DFS3(G, node1):
    pre_dict={}
    S = [node1]
    marked = {}
    for node in G.adj:
        marked[node] = False
    while len(S) != 0:
        current_node = S.pop()
        if not marked[current_node]:
            marked[current_node] = True
            for node in G.adj[current_node]:
                S.append(node)
                if not marked[node]:
                    pre_dict[node]=current_node
    return pre_dict


# Use the methods below to determine minimum Vertex Covers
def add_to_each(sets, element):
    copy = sets.copy()
    for set in copy:
        set.append(element)
    return copy


def power_set(set):
    if set == []:
        return [[]]
    return power_set(set[1:]) + add_to_each(power_set(set[1:]), set[0])


def is_vertex_cover(G, C):
    for start in G.adj:
        for end in G.adj[start]:
            if not(start in C or end in C):
                return False
    return True


def MVC(G):
    nodes = [i for i in range(len(G.adj))]
    subsets = power_set(nodes)
    min_cover = nodes
    for subset in subsets:
        if is_vertex_cover(G, subset):
            if len(subset) < len(min_cover):
                min_cover = subset
    return min_cover


def is_IS(G, Set):
    for u in Set:
        for v in Set:
            if u != v and G.are_connected(u, v):
                return False
    return True


def MIS(G):
    nodes = [i for i in range(G.number_of_nodes())]
    subsets = power_set(nodes)
    max_independent_set = []

    for subset in subsets:
        if is_IS(G, subset):
            if len(subset) > len(max_independent_set):
                max_independent_set = subset

    return max_independent_set


def create_random_graph(n,e):
    g = Graph(n)
    for i in range(e):
        a = random.randint(0,n-1)
        b = random.randint(0,n-1)

        while a==b or g.are_connected(a,b):
            a = random.randint(0,n-1)
            b = random.randint(0,n-1)

        g.add_edge(a,b)
    return g


def print_graph(g,n):
    for i in range(n):
        print(i, "Adjacent: ", g.adj[i])


def has_cycle(g):
    visited = set()

    def dfs(node,parent):
        visited.add(node)
        for neighbour in g.adj[node]:
            if neighbour not in visited:
                if dfs(neighbour,node):
                    return True
            elif parent != neighbour:
                return True
        return False
    for node in g.adj:
        if node not in visited:
            if dfs(node, None):
                return True
    return False


def is_connected(g):
    if not g.adj:
        return False

    start_node = next(iter(g.adj))
    visited = set()
    queue = deque([start_node])

    while queue:
        node = queue.popleft()
        visited.add(node)

        for neighbour in g.adj[node]:
            if neighbour not in visited:
                queue.append(neighbour)
    return len(visited) == len(g.adj)


def experiment_1():
    num_nodes = 100
    num_edges_range = 100
    num_simulations = 1000
    cycle_probabilities = []

    for i in range(int(num_edges_range)):
        has_cycle_count = 0
        for a in range(num_simulations):
            g = create_random_graph(num_nodes,i)
            if has_cycle(g):
                has_cycle_count += 1
        cycle_probabilities.append(has_cycle_count/num_simulations)

    plot.plot(cycle_probabilities)
    plot.title("Number of Edges v. Cycle Probability")
    plot.xlabel("Edges")
    plot.ylabel("Probability")
    plot.show()


def experiment_2():
    num_nodes = 30
    num_edges_range = num_nodes * (num_nodes-1) // 2
    num_simulations = 100
    connection_probabilities = []

    for i in range(0,num_edges_range,2):
        has_connection_count = 0
        for a in range(num_simulations):
            g = create_random_graph(num_nodes,i)
            if is_connected(g):
                has_connection_count += 1
        connection_probabilities.append(has_connection_count/num_simulations)

    plot.plot(connection_probabilities)
    plot.title("Number of Edges v. Connection Probability")
    plot.xlabel("Edges")
    plot.ylabel("Probability")
    plot.show()


def approx1(G):
    C = {}
    max_deg = 0
    v = 0

    # create deep copy of G
    g = copy.deepcopy(G)

    for j in range(len(G.adj)):
        # find vertex with highest degree -- most edges connected
        for i in range(len(g.adj)):
            degree = len(g.adj[i])
            if degree > max_deg:
                max_deg, v = degree, i
        C[len(C)] = v

        # remove all edges incident to v
        for i in range(len(g.adj)):
            if v in g.adj[i]:
                g.adj[i].remove(v)

        # check if C is a VC
        if is_vertex_cover(G, C):
            return C


def approx2(G):
    C = {}
    # randomly select vertex not in C and add it to C
    for i in range(len(G.adj)):
        v = random.randint(0, len(G.adj))
        while v in C:
            v = random.randint(0, len(G.adj))
        C[len(C)] = v

        # check if C is a VC
        if is_vertex_cover(G, C):
            return C


def approx3(G):
    C = {}  # nodes in vertex cover
    B = {}  # nodes connected to a node in the vertex cover C
    # create a deep copy of G
    g = copy.deepcopy(G)
    # at-most you need to add every node
    for i in range(len(G.adj)):
        # take a random vertex
        v = random.randint(0, len(g.adj)-1)
        # vertex needs to be connected to something to get an edge
        p = (len(g.adj[v]))

        # if not connected to another node, check if there are any edges left in g
        if p == 0:
            if sum(sum(g.adj[i][j] for j in range(len(g.adj[i]))) for i in range(len(g.adj))) == 0:
                # if no edges in g, add all nodes not accounted for in C or B
                for node in g.adj:
                    if node not in C and node not in B:
                        C[len(C)] = node
                return C

        # there is an edge, it just needs to be found
        while p == 0:
            v = random.randint(0, len(g.adj)-1)
            p = (len(g.adj[v]))
        u = g.adj[v][random.randint(0, len(g.adj[v])-1)]
        C[len(C)] = v
        C[len(C)] = u

        # remove edges incident to u or v
        for node in g.adj:
            if v in g.adj[node]:
                g.adj[node].remove(v)
                B[len(B)] = v
            if u in g.adj[node]:
                g.adj[node].remove(u)
                B[len(B)] = u

        # check if C is a VC
        if is_vertex_cover(G, C):
            return C


def approximation_experiment1():
    graphs = []
    g = 100  # number of graphs to be generated

    C, C1, C2, C3 = [], [], [], []
    Ci, C1i, C2i, C3i = [], [], [], []

    # For 1-28 edges (max edges in 8 node graph is 28)
    for j in range(1, (8*(8-1)//2)+1):
        # create graphs
        for i in range(g):
            graphs.append(create_random_graph(8, j))

        # run MVC and approximations on each graph and record the size of the VC
        for i in range(len(graphs)-1):
            Ci.append(len(MVC(graphs[i])))
            C1i.append(len(approx1(graphs[i])))
            C2i.append(len(approx2(graphs[i])))
            C3i.append(len(approx3(graphs[i])))

        # compare the average vertex cover size of the approximations to the MVC
        MVCsum = sum(Ci[i] for i in range(len(Ci)))
        C.append(1)
        C1.append((sum(C1i[a] for a in range(len(C1i))))/MVCsum)
        C2.append((sum(C2i[a] for a in range(len(C2i))))/MVCsum)
        C3.append((sum(C3i[a] for a in range(len(C3i))))/MVCsum)

    # plot data
    plot.plot(C, label = "MVC")
    plot.plot(C1, label = "approx1")
    plot.plot(C2, label = "approx2")
    plot.plot(C3, label = "approx3")
    plot.legend()
    plot.title("Number of Edges v Expected Performance")
    plot.xlabel("Edges")
    plot.ylabel("Expected Performance")
    plot.show()


def approximation_experiment2():
    graphs = []
    g = 100  # number of graphs to be generated

    C, C1, C2, C3 = [], [], [], []
    Ci, C1i, C2i, C3i = [], [], [], []

    # For 3-10 nodes
    for j in range(3, 15):
        # create graphs
        for i in range(g):
            graphs.append(create_random_graph(j, 3))

        # run MVC and approximations on each graph and record the size of the VC
        for i in range(len(graphs)-1):
            Ci.append(len(MVC(graphs[i])))
            C1i.append(len(approx1(graphs[i])))
            C2i.append(len(approx2(graphs[i])))
            C3i.append(len(approx3(graphs[i])))

        # compare the average vertex cover size of the approximations to the MVC
        MVCsum = sum(Ci[i] for i in range(len(Ci)))
        C.append(1)
        C1.append((sum(C1i[a] for a in range(len(C1i))))/MVCsum)
        C2.append((sum(C2i[a] for a in range(len(C2i))))/MVCsum)
        C3.append((sum(C3i[a] for a in range(len(C3i))))/MVCsum)

    x = [i for i in range(3,15)]
    # plot data
    plot.plot(x,C, label = "MVC")
    plot.plot(x,C1, label = "approx1")
    plot.plot(x,C2, label = "approx2")
    plot.plot(x,C3, label = "approx3")
    plot.legend()
    plot.title("Number of Nodes v Expected Performance")
    plot.xlabel("Nodes")
    plot.ylabel("Expected Performance")
    plot.show()

def approximation_experiment3():
    graphs = []
    g = 100  # number of graphs to be generated

    C, C1, C2, C3 = [], [], [], []
    Ci, C1i, C2i, C3i = [], [], [], []

    # For 3-10 nodes
    for j in range(3, 15):
        # create graphs
        for i in range(g):
            graphs.append(create_random_graph(j, j//2))

        # run MVC and approximations on each graph and record the size of the VC
        for i in range(len(graphs)-1):
            Ci.append(len(MVC(graphs[i])))
            C1i.append(len(approx1(graphs[i])))
            C2i.append(len(approx2(graphs[i])))
            C3i.append(len(approx3(graphs[i])))

        # compare the average vertex cover size of the approximations to the MVC
        MVCsum = sum(Ci[i] for i in range(len(Ci)))
        C.append(1)
        C1.append((sum(C1i[a] for a in range(len(C1i))))/MVCsum)
        C2.append((sum(C2i[a] for a in range(len(C2i))))/MVCsum)
        C3.append((sum(C3i[a] for a in range(len(C3i))))/MVCsum)

    x = [i for i in range(3,15)]
    # plot data
    plot.plot(x,C, label = "MVC")
    plot.plot(x,C1, label = "approx1")
    plot.plot(x,C2, label = "approx2")
    plot.plot(x,C3, label = "approx3")
    plot.legend()
    plot.title("Number of Nodes v Expected Performance")
    plot.xlabel("Nodes")
    plot.ylabel("Expected Performance")
    plot.show()


def vertex_cover_independent_set_experiment(num_graphs, n, e):
    mvc_sizes = []
    mis_sizes = []
    combined_sizes = []

    for i in range(num_graphs):
        random_graph = create_random_graph(n, e)

        mvc = MVC(random_graph)
        mis = MIS(random_graph)

        mvc_sizes.append(len(mvc))
        mis_sizes.append(len(mis))

        combined_sizes = [mvc_sizes[i] + mis_sizes[i] for i in range(len(mvc_sizes))]
    x = [i for i in range(1, num_graphs+1)]
    plot.figure(figsize=(10, 5))
    plot.subplot(1, 2, 1)
    plot.plot(x, mvc_sizes, label='MVC Size')
    plot.plot(x, mis_sizes, label='MIS Size')
    plot.xlabel('Graph Number')
    plot.ylabel('Size')
    plot.title('MVC and MIS Sizes')
    plot.legend()

    plot.subplot(1, 2, 2)
    plot.plot(x, combined_sizes, label='MVC Size + MIS Size')
    plot.xlabel('Graph Number')
    plot.ylabel('Size')
    plot.title('MVC + MIS Sizes')
    plot.legend()

    plot.show()


experiment_1()

