from collections import deque


def bfs(graph, start, goal):
    visited = set()
    queue =deque([[start]])

    if start == goal:
        return "Start and goal nodes are the same"

    while queue:
        path= queue.popleft()
        node = path[-1]

        if node not in visited:
            neighbors = graph[node]
            
            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor) 
                queue.append(new_path)

                if neighbor == goal:
                    return new_path

            visited.add(node)

    return "No path found between start and goal"


graph = {
    'A': ['B','L'],
    'L': ['O','Q'],
    'B': ['C','X'],
    'C': ['E'],
    'X': ['Y'],
    'E':['Z'],
    'Y':['J'],
    'J': ['N','M'],
    'M': ['G'], 

    'O': [],
    'Q': [],
    'Z': [],
    'N': [],
    'G': []
}

start_node = 'A'
End_node= 'G'
print("BFS PATH: ", bfs(graph, start_node, End_node))