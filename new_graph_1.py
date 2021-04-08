from __future__ import annotations
from collections import deque, defaultdict
from typing import Container, NewType, Any

_vertex = NewType("_vertex", Any)

def depth_first_search(graph: dict[_vertex, list[_vertex]], current: _vertex) -> list[_vertex]:
    result: list[_vertex] = []
    visited: dict[_vertex, bool] = {vertex: False for vertex in graph}
    stack: Container[_vertex] = [current]
    
    while stack:
        dqnode: _vertex = stack.pop()
        if not visited[dqnode]:
            visited[dqnode] = True
            result.append(dqnode)
            stack.extend(graph[dqnode][::-1])       # Reverse the List
    
    return result       # result list hold the dfs order of _vertexes



def breath_first_search(graph: dict[_vertex, list[_vertex]], current: _vertex) -> list[_vertex]:
    result: list[_vertex] = []
    visited: dict[_vertex, bool] = {vertex: False for vertex in graph}
    queue: deque[_vertex] = deque([current])

    while queue:
        dqnode: _vertex = queue.popleft()
        if not visited[dqnode]:
            visited[dqnode] = True
            result.append(dqnode)
            queue.extend(graph[dqnode])
    
    return result


# Detect cycle in undirected and directed graph
# graph coloring
def detect_cycle(graph: dict[_vertex, list[_vertex]], current: _vertex) -> bool:
    colors: dict[_vertex, str] = {vertex: 'white' for vertex in graph}
    colors[current] = 'gray'
    stack: Container[_vertex] = [(None, current)]   # (u, v)
    # for cycle count
    #cycle_count: int = 0

    while stack:
        prev, node = stack.pop()
        for neighbour in graph[node]:
            if neighbour == prev:
                pass
            elif colors[neighbour] == 'gray':
                return True
                #cycle_count += 1
            else:
                colors[neighbour] = 'gray'
                stack.append((node, neighbour))
    
    #return cycle_count-1 : for cycle count
    return False



# Graph Topological sort
def topolocial_sort(graph: dict[_vertex, list[_vertex]], current: _vertex) -> list[_vertex]:
    # topo vars
    stack: Container[_vertex] = []
    order: list[_vertex] = []

    # dfs vars                                                      # --- dfs begins
    dfs_stack: Container[_vertex] = [current]
    visited: dict[_vertex, bool] = {vertex: False for vertex in graph}

    while dfs_stack:
        dqnode: _vertex = dfs_stack.pop()
        if not visited[dqnode]:
            visited[dqnode] = True
            dfs_stack.extend(graph[dqnode][::-1])
                                                                    # --- dfs end
            while stack and dqnode not in graph[stack[-1]]:
                order.append(stack.pop())
            stack.append(dqnode)

    return stack + order[::-1]



# Dijkstra Algorithms: Shortest distance
def dijkstraAlgo(graph: dict[_vertex, dict[_vertex, int]], current: _vertex) -> dict[_vertex, int]:
    unvisited: dict[_vertex, int|None] = {vertex: None for vertex in graph}
    visited: dict[_vertex, int] = {}
    currentDistance: int = 0
    unvisited[current] = currentDistance

    while True:
        for neighbour, distance in unvisited[current].items():
            if neighbour not in visited:
                continue
            
            # --
            newDistance: int = currentDistance + distance

            # --
            if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:
                unvisited[neighbour] = newDistance
        
        visited[current] = currentDistance
        del unvisited[current]
        if not unvisited:
            break
        
        candidates: list[tuple[_vertex, int]] = [vertexTup for vertexTup in unvisited.items() is vertexTup[1]]
        current, currentDistance = sorted(candidates, key= lambda x: x[1])[0]
    
    return visited


# Prim's Algorithm : Minimal Spanning Tree
def primsAlgo(graph: dict[Any, dict[Any, int]], current: Any) -> dict[Any, int|float]:
    unvisited: dict[Any, Any] = {vertex: None for vertex in graph}
    visited: dict[Any, Any] = {}
    currentDistance: int = 0
    unvisited[current] = currentDistance

    while True:
        for adjVertex, distance in unvisited[current].items():
            if adjVertex not in visited:
                continue
            # --- 
            newDistance: int = distance
            # ---
            if unvisited[adjVertex] is None or unvisited[adjVertex] > newDistance:
                unvisited[adjVertex] = newDistance
        visited[current] = currentDistance
        del unvisited[current]
        if not unvisited:
            break
        candidates: list[tuple[Any, int]] = [vertexTup for vertexTup in unvisited.items() if vertexTup[1]]    
        current, currentDistance = sorted(candidates, key= lambda x: x[1])[0]
    
    return visited

        

# Bellmanford Algo: single shortest path and here cost can also be negative
# @ https://gist.github.com/joninvski/701720
def bellmanfordAlgo(graph: dict[Any, list[Any]], current: Any) -> dict[Any, int|float]:
    # Step:1 - Initialization
    unvisited: dict[Any, Any] = {vertex: float('inf') for vertex in graph}
    visited: dict[Any, Any] = {}
    currentDistance: int = 0
    unvisited[current] = currentDistance

    # Step:2 - Relaxation
    def relax(vertex: Any, adjVertex: Any, graph: dict[Any, list[Any]], unvisited: dict[Any, Any], visited: dict[Any, Any]) -> None:
        if unvisited[adjVertex] > unvisited[vertex] + graph[vertex][adjVertex]:
            unvisited[adjVertex] = unvisited[vertex] + graph[vertex][adjVertex]
            visited[adjVertex] = vertex

    # Step:3 - RUN LOOP for |V| - 1 times
    for _ in range(len(graph)-1):
        for vertex in graph:
            for adjVertex in graph[vertex]:
                relax(vertex, adjVertex, graph, unvisited, visited)

    # Step:4 - Checking for any Negative-weight Cycles
    for vertex in graph:
        for adjVertex in graph[vertex]:
            assert unvisited[adjVertex] <= unvisited[vertex] + graph[vertex][adjVertex]

    return visited


# Strongly Connected Components : Kojaraju and Tarjan algo 








# Driver Code
if __name__ == "__main__":
    graph: dict[_vertex, list[_vertex]] = { 
        0: [1, 2],
        1: [2], 
        2: [0, 3], 
        3: [3]
    }

    #print(depth_first_search(graph, 2))
    #print(breath_first_search(graph, 2))

    # Detect cycle in directed graph
    # it does not have cycle
    graph_1: dict[_vertex, list[_vertex]] = {
        0: [4],
        4: [0, 5],
        5: [4, 6],
        6: [5]
    }

    # it has cycle
    graph_2: dict[_vertex, list[_vertex]] = {
        1: [8, 2],
        2: [1, 3],
        3: [8, 2],
        8: [3, 1]
    }

    # this graph(directed) has 1 cycle
    graph_3: dict[_vertex, list[_vertex]] = {
        0: [1],
        1: [2],
        2: [0, 3],
        3: [4, 5],
        4: [],
        5: [4]
    } 

    #print(detect_cycle(graph_1, 0))
    #print(detect_cycle(graph_2, 1))
    #print(detect_cycle(graph_3, 0))

    # topological sort graph
    graph_4: dict[_vertex, list[_vertex]] = {
        'a': ['b', 'c'],
        'b': ['d'],
        'c': ['d'],
        'd': ['e'],
        'e': []
    }
    print(topolocial_sort(graph_4, 'a'))

    
    # dijkstra algo
    directedWeightedGraph: dict[_vertex, dict[_vertex, int]] = {
        'a': {'b': 1}, 
        'b': {'c': 2, 'b': 5}, 
        'c': {'d': 1},
        'd': {}
    }

