from __future__ import annotations
import collections
from typing import Any, Deque
from collections import deque, defaultdict, Container


# --------- Graph Definition -------------- #
class Graphs:
    def __init__(self, graph_dict: dict[Any, list] = {}) -> None:
        self.adjList = graph_dict
        self.adjMat = [[edge for edge in vertex] for vertex in graph_dict]
        
    def addEdge(self, startingVertex: Any, endVertex: Any, cost: int = 1) -> None:              # O(1)
        if not isinstance(cost, int):
            raise TypeError("Cost Must be an Integer Value.")
        self.adjMat[startingVertex][endVertex] = cost
        
    def removeEdge(self, startingVertex: Any, endVertex: Any) -> None:                          # O(1)
        self.adjMat[startingVertex][endVertex] = 0
        
    def existEdge(self, startingVertex: Any, endvertex: Any) -> bool:                           # O(1)
        return self.adjMat[startingVertex][endvertex] != 0
    
    def countVertex(self) -> int:                                                               # O(|V|)
        return len(self.adjList)
        
    def countEdge(self) -> int:                                                                 # O|V|^2 as len(|V|) == len(|E|) 
        count: int = 0
        for i in self.adjMat:
            for j in self.adjMat:
                if self.adjMat[i][j] != 0:
                    count += 1
        return count

    def listVertex(self) -> list[int] or list[str]:                                             # O(|V|)
        list_vertex: list[Any] = [vertex for vertex in self.adjList]
        return list_vertex
        
    def listEdge(self) -> list[int] or list[str]:                                               # O(|V|^2)
        list_edge: list[Any] = [[edge for edge in vertex] for vertex in self.adjList]
        return list_edge
        
    def visualEdge(self) -> list[str]:
        for vertex in self.adjMat:
            for edge in vertex:
                if self.adjMat[vertex][edge] != 0:
                    print(vertex, "--", edge)
                    
    def adjacentMatrix(self) -> list[list[int]]:
        return self.adjMat

#-------------------- Graph Basics -------------- #
def produceEdgelist(graph: dict[Any, list[Any]]) -> list[tuple[Any, ...]]:
    return [(vertex, edge) for vertex, adjvertex in graph.items() for edge in adjvertex]

# ---------- Graph Algorithms ------------------ #
def convertgraph(graph: list[list[int]]) -> dict[Any, list[Any]]:
    adjList: dict[Any, list[Any]] = defaultdict(list)
    for vertex in graph:
        for edge in vertex:
            if graph[vertex][edge] != 0:
                adjList[vertex].append(edge)
    return adjList


def depthFirstSearch(graph: dict[Any, list[Any]], vertex: Any) -> list[Any]:
    result: list[Any] = []
    visited: list[Any] = []
    stack: Container[Any] = [vertex]
    while stack:
        dqnode = stack.pop()
        if dqnode not in visited:
            visited.append(dqnode)
            result.append(dqnode)
            stack.extend(reversed(graph[dqnode]))
    return result


# Time Complexity : O(|V| + |E|)
def breadthFirstSearch(graph: dict[Any, list[Any]], vertex: Any) -> list[Any]:
    result: list[Any] = []
    visited: list[Any] = []
    queue: Deque[Any] = deque([vertex]) 
    while queue:
        dqnode = queue.popleft()
        if dqnode not in visited:
            visited.append(dqnode)
            result.append(dqnode)
            queue.extend(graph[dqnode])
    return result


# Detecting Cycle in Directed Graphs
# https://algocoding.wordpress.com/2015/04/02/detecting-cycles-in-a-directed-graph-with-dfs-python/
class IsCyclicDirected:
    def IsCycleExist(self, graph: dict[Any, list[Any]]) -> bool:
        color: dict[Any, str] = {vertex: "white" for vertex in graph}
        foundCycle: list[bool] = [False]
        for vertex in graph:
            if color[vertex] == "white":
                self.dfsVisit(graph, vertex, color, foundCycle)
            if foundCycle[0] is False:
                break
        return foundCycle[0]


    def dfsVisit(self, graph: dict[Any, list[Any]], vertex: Any, foundCycle: list[bool], color: dict[Any, str]) -> Any: 
        if foundCycle[0] is True:
            return 
        color[vertex] = "gray"
        for adjVertex in graph[vertex]:
            if color[adjVertex] == "gray":
                foundCycle[0] = True
                return
            if color[adjVertex] == "white":
                self.dfsVisit(graph= graph, vertex= adjVertex, color= color, foundCycle= foundCycle)
        color[vertex] = "black"
        
    

# Detecting Cycle in Undirected Graphs
class isCyclicUndirected: 
    def IsCyclic(self, graph: dict[Any, list[Any]]) -> bool: 
        marked: dict[Any, bool] = {vertex: False for vertex in graph}
        foundCycle: list[bool] = [False]
        for vertex in graph:
            if marked[vertex] is False:
                self.dfsVisit(graph, foundCycle, vertex, marked, vertex)
            if marked[vertex] is True:
                break
        return foundCycle[0]


    def dfsVisit(self, graph: dict[Any, list[Any]], foundCycle: list[bool], vertex: Any, marked: dict[Any, bool], predVertex: Any) -> Any:
        if foundCycle[0] is True:
            return
        marked[vertex] = True
        for adjVertex in graph[vertex]:
            if marked[adjVertex] and adjVertex != predVertex:
                foundCycle[0] = True
                break
            if marked[adjVertex] is False:
                self.dfsVisit(graph= graph, foundCycle= foundCycle, vertex= adjVertex, marked= marked, predVertex= vertex)


# DFS TopoSorting of the Vertices of the directed graph
class TopoSortingDFS:
    def dfsTopoSort(self, graph: dict[Any, list[Any]]) -> list[Any]: 
        result: list[Any] = []
        color: dict[Any, str] = {vertex: "white" for vertex in graph}
        foundCycle: list[bool] = [False]
        for vertex in graph:
            if color[vertex] == "white":
                self.dfsVisit(graph, vertex, foundCycle, color, result)
            if foundCycle[0] is True:
                break
        if foundCycle[0] is True:
            return []
        return result.reverse()


    def dfsVisit(self, graph: dict[Any, list[Any]], vertex: Any, foundCycle: list[bool], color: dict[Any, str], result: list[Any]) -> Any: 
        if foundCycle[0] is True:
            return
        color[vertex] = "gray"
        for adjvertex in graph[vertex]:
            if color[adjvertex] == "gray":
                foundCycle[0] = True
                return
            if color[adjvertex] == "white":
                self.dfsVisit(graph, adjvertex, foundCycle, color, result)
        color[vertex] = "black"
        result.append(vertex)


# Dijkstra's algorithm: Single Shortest Path from source to every vertexes of the graph: Cost can only be positive
def DijkstraAlgo(graph: dict[Any, dict[Any, int]], current: Any) -> dict[Any, int]:
    unvisited: dict[Any, Any] = {vertex: None for vertex in graph}                                                  # marking all vertices as None 
    visited: dict[Any, int] = {}                                                                                    # visited dictionary <Vertex : Cost>
    currentDistance: int = 0                                                                                        # current source vertex cost
    unvisited[current] = currentDistance                                                                            # setting <current_vertex : cost> in unvisited dictionary 

    while True:                                                                                                     # looping through all the adjacent vertex of the current vertex
        for adjVertex, distance in unvisited[current].items():
            if adjVertex not in visited:                                                                        
                continue
            newDistance: int = currentDistance + distance                                                           # while iterating, adding adjacent_vertex.cost to the current cost 
            if unvisited[adjVertex] is None or unvisited[adjVertex] > newDistance:                                  
                unvisited[adjVertex] = newDistance
        visited[current] = currentDistance                                                                          # adding current vertex in the visited dictionary
        del unvisited[current]                                                                                      # now removing the current vertex from the unvisited dictionary
        if not unvisited:                                                                                           # if unvisited dictionary is empty then End while
            break
        candidates: list[tuple[Any, int]] = [vertex for vertex in unvisited.items() if vertex[1]]                   # all the vertices in the unvisited dictionary if there cost != None
        current, currentDistance = sorted(candidates, key= lambda x: x[1])[0]                                       # Sorting list[tuple[vertex, cost]] based on there cost in ascending  
                                                                                                                    # and modifying the current(Vertex) and currentDistance(Vertex's cost)
    return visited                                                                                                  # returning the dictionary containing single shortest path to every vertex


# Bellmanford Algo: single shortest path and here cost can also be negative




# Driver Code
if __name__ == "__main__":
    graph: dict[int, list[int]] = { 
        0: [1, 2],
        1: [2], 
        2: [0, 3], 
        3: [3]
    }

    print(depthFirstSearch(graph, 2))
    print(breadthFirstSearch(graph, 2))
    directedGraph: dict[int, list[int]] = {
        0 : [1], 
        1 : [2], 
        2 : [3], 
        3 : [4], 
        4 : [1]
    }

    directedWeightedGraph: dict[str, dict[str, int]] = {
        'a': {'b': 1}, 
        'b': {'c': 2, 'b': 5}, 
        'c': {'d': 1},
        'd': {}
    }


    
    g: list[list[int]] = [[1,0, 0], [0, 1, 1], [1, 0, 1]]
    adjlist: dict[int, list[int]] = defaultdict(list)
    adjlist: dict[int, list[int]] = { for vertex, adjvertex in g.items() for edge in adjvertex if g[vertex][edge] != 0}










