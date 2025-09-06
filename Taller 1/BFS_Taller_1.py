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
'A': ['F','G'], 
'S': ['B','D'],
'E': ['K', 'L'],

'F': ['M'],
'G': [], 
'B': ['H','R'],
'D': ['J'],
'K': ['I'],
'L': ['CC'],

'M': ['N'],
'H': ['O','Q'],
'R': ['X','T'],
'J':['Y'],
'I': [], 
'CC':['DD'],

'N': [],
'O': ['P'],
'Q': ['U'],
'X': [],
'T': ['GG'],
'Y': ['Z'],
'CC':['DD','EE'],

'P': [],
'U': ['V','W'],
'GG': [],
'Z': ['AA','BB'],
'DD': [],
'EE': ['FF'],

'V': [],
'W': [],
'AA': [],
'BB': [],
'FF': []
}

start_node = 'S'
End_node= 'W'
print("BFS PATH: ", bfs(graph, start_node, End_node))