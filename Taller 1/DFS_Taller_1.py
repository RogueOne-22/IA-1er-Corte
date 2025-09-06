def dfs (graph, start, goal, path=None):
    if path is None:
        path=[start]

    if start == goal:
        return path
    if start not in graph:
        return None
    
    for node in graph[start]:
        if node not in path:
            new_path = path +[node]
            result = dfs(graph, node, goal, new_path)

        if result is not None:
            return result
        
    return None


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
print("DFS PATH: ", dfs(graph, start_node, End_node))
