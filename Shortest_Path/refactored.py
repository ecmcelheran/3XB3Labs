from abc import ABC, abstractmethod
import min_heap

class SPAlgorithm(ABC):
    @abstractmethod
    def calc_sp(graph: Graph, source: int, dest: int) -> float:
        pass

class Dijkstra(SPAlgorithm):

    def calc_sp(graph: Graph, source: int, dest: int) -> float:
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
        return dist[dest]


class Bellman_Ford(SPAlgorithm):
    def calc_sp(graph: Graph, source: int, dest: int) -> float:
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
        return dist[dest]


class Astar:
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
class Adapter(SPAlgorithm):
    def __init__(self, G):
        self.class_Astar = Astar()
        self.G = G  

    def find_shortest_path(self, start, destination):
        h = heuristic_function(self.G, destination)
        self.class_Astar.a_star(self.G, start, destination, h)

class ShortPathFinder:
    def __init__(self, graph: Graph, algorithm: SPAlgorithm):
        self.__graph = graph
        self.__algorithm = algorithm

    def calc_short_path(self, source: int, dest: int) -> float:
        return self.__algorithm.calc_sp(self.__graph, source, dest)

    def set_graph(self, graph: Graph):
        self.__graph = graph

    def set_algorithm(self, algorithm: SPAlgorithm):
        self.__algorithm = algorithm

class Graph(ABC):

    def __init__(self):
        self.adj = {}
        self.weights = {}
        self.coord = {}

    @abstractmethod
    def get_adj_nodes(self, node: int) -> list[int]:
        pass

    @abstractmethod
    def add_node(self, node:int):
        pass

    @abstractmethod
    def add_edge(self, start: int, end:int, w:float):
        pass

    @abstractmethod
    def get_num_of_nodes(self) -> int:
        pass

    @abstractmethod
    def w(self, node1:int) -> float:
        pass

class WeightedGraph(Graph):


    def w(self, node1:int, node2:int) -> float:
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]
        

class HeuristicGraph(WeightedGraph):
    def __init__(self, heuristic: dict[int, float]):
        super().__init__()
        self.__heuristic: dict[int, float] = {}

    def get_heuristic(self) -> dict[int, float]:
        return self.__heuristic
    
