import heapq

# 1. Definición del laberinto
size_x = 27
size_y = 14
estado_inicial = (0, 0)
estado_meta = (26, 13)

# Obstáculos (según la imagen)
obstaculos = set([
  (1, 5), (1, 6), (1, 10), (1, 11),
  (2, 5), (2, 6), (2, 10), (2, 11),
  (3, 5), (3, 6), (3, 10), (3, 11),
  (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11),
  (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11),
  (6, 5), (6, 6), 
  (7, 5),(7, 6),(7, 9),(7, 10),
  (8, 5), (8, 6), (8, 9), (8, 10), (8, 12), (8, 13), 
  (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 8), (9, 9), (9, 12), (9, 13),
  (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 8), (10, 9), (10, 12), (10, 13),
  (11, 12), (11, 13),
  (12, 12), (12, 13),
  (13, 0), (13, 1), (13, 7), (13, 8), (13, 12), (13, 13),
  (14, 0), (14, 1), (14, 3), (14, 4), (14, 7), (14, 8), (14, 12), (14, 13),
  (15, 3), (15, 4), (15, 7), (15, 8), (15, 12), (15, 13),
  (16, 3), (16, 4), (16, 7), (16, 8), (16, 9), (16, 10), (16, 11), (16, 12), (16, 13),
  (17, 3), (17, 4), (17, 5), (17, 6), (17, 7), (17, 8), (17, 9), (17, 10), (17, 11), (17, 12), (17, 13),
  (20, 1), (20, 2), (20, 10), (20, 11), (20, 12),
  (21, 1), (21, 2), (21, 3), (21, 4), (21, 5), (21, 6), (21, 7), (21, 10), (21, 11), (21, 12),
  (22, 4), (22, 5), (22, 6), (22, 7), (22, 10), (22,11), (22, 12),
  (23, 4), (23, 5), (23, 8), (23, 9),
  (24, 4), (24, 5), (24, 8), (24,9 ),
  (25, 4), (25, 5), 
  (26, 4), (26, 5), 
 
])

acciones = {
    "arriba": (0, 1),
    "abajo": (0, -1),
    "derecha": (1, 0),
    "izquierda": (-1, 0)
}

# 2. Funciones
def heuristica(estado):
    """Distancia Manhattan"""
    return abs(estado[0] - estado_meta[0]) + abs(estado[1] - estado_meta[1])

def vecinos(estado):
    """Genera vecinos válidos"""
    result = []
    for accion, (dx, dy) in acciones.items():
        nuevo = (estado[0] + dx, estado[1] + dy)
        if (0 <= nuevo[0] < size_x and 0 <= nuevo[1] < size_y) and (nuevo not in obstaculos):
            result.append((nuevo, accion))
    return result

def mostrar_laberinto(path=set()):
    """Visualización del laberinto"""
    for y in range(size_y - 1, -1, -1):
        fila = ""
        for x in range(size_x):
            if (x, y) == estado_inicial:
                fila += " α "
            elif (x, y) == estado_meta:
                fila += " Ω "
            elif (x, y) in path:
                fila += " Θ "
            elif (x, y) in obstaculos:
                fila += " ■ "
            else:
                fila += " . "
        print(fila)
    print()

# 3. Algoritmo
def a_star():
    frontera = []
    heapq.heappush(frontera, (heuristica(estado_inicial), 0, estado_inicial, []))
    visitados = set()

    while frontera:
        f, g, estado, camino = heapq.heappop(frontera)

        if estado in visitados:
            continue
        visitados.add(estado)

        if estado == estado_meta:
            return camino + [estado]

        for vecino, accion in vecinos(estado):
            if vecino not in visitados:
                nuevo_camino = camino + [estado]
                nuevo_g = g + 1
                nuevo_f = nuevo_g + heuristica(vecino)
                heapq.heappush(frontera, (nuevo_f, nuevo_g, vecino, nuevo_camino))

    return None

# 4. Simulación
print(" Laberinto inicial con obstáculos:")
mostrar_laberinto()

ruta = a_star()

if ruta:

  print("🎉 Ruta encontrada con A*:")

  # Dividir la ruta en bloques de 10
  for i in range(0, len(ruta), 10):
    bloque = ruta[i:i+10]
    print(" → ".join(map(str, bloque)))
    print("")

  mostrar_laberinto(path=set(ruta))
else:
    print("⚠️ No hay ruta posible hasta la meta.")
