# 🔎 Comparación de Algoritmos con Grafos  
En este punto se muestran tres implementaciones de algoritmos clásicos de búsqueda en grafos:  

- **BFS (Breadth First Search)**  
- **DFS (Depth First Search)**  
- **UCS (Uniform Cost Search)**  

Cada uno tiene un enfoque distinto para recorrer el grafo y encontrar un camino desde un **nodo inicial** hasta un **nodo meta**.

---

## 📌 1. BFS (Breadth First Search)

### 🔹 Descripción
- Explora los nodos **nivel por nivel** usando una **cola FIFO**.  
- Siempre encuentra el **camino más corto en número de pasos** (cuando no hay pesos en las aristas).  

### 👀 Visualización
- El camino resultante muestra el recorrido más **corto en cantidad de aristas** desde el inicio hasta la meta.  
- Se puede observar cómo expande primero los vecinos directos y luego los más lejanos.

---

## 📌 2. DFS (Depth First Search)

### 🔹 Descripción
- Explora un camino **lo más profundo posible** antes de retroceder.  
- No garantiza el camino más corto, pero suele encontrar una solución rápidamente si el objetivo está en una rama profunda.  

### 👀 Visualización
- El recorrido muestra una **exploración lineal** hacia un camino posible.  
- El resultado puede no ser el camino más eficiente, pero evidencia cómo el algoritmo se enfoca en **profundidad**.

---

## 📌 3. UCS (Uniform Cost Search)

### 🔹 Descripción
- Extiende BFS pero considerando **costos en las aristas**.  
- Utiliza una **cola de prioridad** para expandir siempre el nodo con **menor costo acumulado**.  
- Garantiza el **camino más barato en términos de peso**, no necesariamente el más corto en cantidad de pasos.  

### 👀 Visualización 
- El camino encontrado es el de **menor costo total**.  
- Se observa cómo puede dar un recorrido más largo en número de pasos, pero más **eficiente en costo**.  

---

## 🚀 Diferencias Clave

| Algoritmo | Estructura usada | Garantiza camino más corto | Considera costos | Exploración |
|-----------|-----------------|----------------------------|-----------------|-------------|
| **BFS**  | Cola (FIFO)      | ✅ Sí (en grafos no ponderados) | ❌ No            | Por niveles |
| **DFS**  | Pila | ❌ No                      | ❌ No            | En profundidad |
| **UCS**  | Cola de prioridad| ✅ Sí (en grafos ponderados)   | ✅ Sí            | Por costo acumulado |

---

## 🎯 Conclusión
- Usa **BFS** si quieres el camino más corto en pasos y el grafo no tiene pesos.  
- Usa **DFS** si buscas explorar rápido y profundamente, sin importar la optimalidad.  
- Usa **UCS** si el grafo tiene **costos diferentes en las aristas** y necesitas el camino más económico.  

---

✍️ Autor: *Paula*  
