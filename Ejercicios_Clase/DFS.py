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
print("DFS PATH: ", dfs(graph, start_node, End_node))
