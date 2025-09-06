import heapq

def ucs(grafo, inicio, meta):
    cola = [(0, inicio)]
    visitados = set()
    costo_acumulado = {inicio: 0}
    padre = {inicio: None}

    while cola:
        costo, nodo = heapq.heappop(cola)
        if nodo in visitados:
            continue
        visitados.add(nodo)
        if nodo == meta:
            break
        for vecino, costo_arista in grafo[nodo]:
            nuevo_costo = costo + costo_arista
            if vecino not in costo_acumulado or nuevo_costo < costo_acumulado[vecino]:
                costo_acumulado[vecino] = nuevo_costo
                heapq.heappush(cola, (nuevo_costo,vecino))
                padre[vecino] = nodo
            
    return reconstruir_camino(padre, meta), costo_acumulado.get(meta, float('inf'))

def reconstruir_camino(padre, meta):
    camino = []
    nodo = meta
    while nodo is not None:
        camino.append(nodo)
        nodo = padre.get(nodo)
    return list(reversed(camino))


grafo_costo = {
    'A': [('B', 1), ('L', 1)],
    'L': [('O', 2), ('Q', 2)],
    'B': [('C', 3), ('X', 3)],
    'C': [('E', 4)],
    'X': [('Y', 4)],
    'E': [('Z', 5)],
    'Y': [('J', 5)],
    'J': [('N', 1), ('M', 1)],
    'M': [('G', 2)], 
    'O': [],
    'Q': [],
    'Z': [],
    'N': [],
    'G': []
}

camino, costo = ucs(grafo_costo, 'A', 'G')
print("UCS - PATH: ", camino, "Costo: ", costo)
        
