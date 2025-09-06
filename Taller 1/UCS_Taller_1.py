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
'A': [('F', 1),('G', 1)], 
'S': [('B',1),('D',1)],
'E': [('K',1), ('L',1)],

'F': [('M',2)],
'G': [], 
'B': [('H',2),('R',2)],
'D': [('J',2)],
'K': [('I',2)],
'L': [('CC',2)],

'M': [('N',3)],
'H': [('O',3),('Q',3)],
'R': [('X',3),('T',3)],
'J':[('Y',3)],
'I': [], 
'CC':[('DD',3)],

'N': [],
'O': [('P', 4)],
'Q': [('U',4)],
'X': [],
'T': [('GG',4)],
'Y': [('Z',4)],
'CC':[('DD',4),('EE',4)],

'P': [],
'U': [('V',3),('W',3)],
'GG': [],
'Z': [('AA',3),('BB',3)],
'DD': [],
'EE': [('FF',3)],

'V': [],
'W': [],
'AA': [],
'BB': [],
'FF': []
}

camino, costo = ucs(grafo_costo, 'S', 'W')
print("UCS - PATH: ", camino, "Costo: ", costo)