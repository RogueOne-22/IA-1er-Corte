# 🚀 Resolución de Laberinto con Algoritmo de Búsqueda

Este ejercicio implementa el **algoritmo A\*** para encontrar la ruta más corta entre un **estado inicial** y un **estado meta** en un circuito lleno de obstáculos.  

La representación se hace en una grilla, donde:  
- **α** es el punto inicial.  
- **∩** es la meta.  
- **■** son los obstáculos.  
- **Θ*** es el camino encontrado.  

---

## 📌 Características principales del código

- **Definición del laberinto:**  
  Se define un tablero con dimensiones, un punto inicial `(0,0)` y una meta `(26,14)`.

- **Obstáculos personalizados:**  
  Se cargan desde una matriz basada en una imagen (`Circuito.png`), por lo que el laberinto se ajusta a obstáculos reales.
  
  <img width="1463" height="716" alt="Circuito" src="https://github.com/user-attachments/assets/197f4adb-3e90-4746-ae91-de75e3f8fc82" />


- **Algoritmo implementado:**  
  - Función de **heurística Manhattan** para estimar la distancia.  
  - Uso de una **cola de prioridad (`heapq`)** para expandir nodos óptimos.  
  - Expansión de vecinos en 4 direcciones (arriba, abajo, derecha, izquierda).  

- **Visualización del laberinto y la ruta:**  
  - Representación en consola.  
  - Impresión de la ruta paso a paso, con **10 puntos del recorrido por línea** para mayor legibilidad.
    
  
  <img width="729" height="311" alt="Captura de pantalla 2025-09-06 211450" src="https://github.com/user-attachments/assets/5a362f0f-f9bb-439f-be38-a13855e5e2df" />

## 📸 Visualización de la solucion:
  
  <img width="969" height="561" alt="Captura de pantalla 2025-09-06 211502" src="https://github.com/user-attachments/assets/9eaafea8-5056-4552-bc23-7a071ad0a0dd" />

## 🧠 Conceptos importantes
  - A* garantiza la ruta más corta si la heurística es admisible.
  - Heurística Manhattan funciona en grids donde solo se permiten movimientos en las 4 direcciones.
  - La impresión en bloques de 10 pasos por línea facilita seguir el recorrido.
---
