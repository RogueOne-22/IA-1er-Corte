# IA-1er-Corte - Laboratorio 2
Introduccion a IA

# 🧠 Laboratorio ·2:  IA y Campo de Potenciales Artificiales

## 📚 Librerías utilizadas
- **NumPy**: Librería para cálculos matemáticos y operaciones con vectores/matrices. Se usa para calcular distancias, gradientes y movimientos del agente.  
- **Matplotlib**: Librería para visualización de datos. Se emplea para mostrar el campo de potencial como un mapa de contornos y animar la trayectoria del agente.  
- **FuncAnimation (Matplotlib.animation)**: Permite crear animaciones dinámicas paso a paso, en este caso para mostrar cómo el agente se desplaza hacia el objetivo.  

---

## 🤖 Conceptos teóricos

### 🔹 Agente Inteligente
Un **agente inteligente** es una entidad que percibe su entorno mediante sensores, procesa la información y ejecuta acciones que modifican su entorno. No necesariamente es “inteligente” en el sentido humano, sino que actúa de manera **racional** en función de los datos de entrada.

### 🔹 Cambio de Potenciales en IA
El **campo de potenciales artificiales** es un modelo utilizado en robótica móvil para la navegación:
- **Fuerza atractiva**: atrae al agente hacia el objetivo.  
- **Fuerza repulsiva**: aleja al agente de los obstáculos.  
- La suma de estas fuerzas genera un **campo de potencial total**.  
- El agente se mueve en dirección descendente del potencial (mínimos energéticos).  

Este método es simple y eficiente, aunque puede tener problemas con mínimos locales que atrapan al agente.

---

## 🧩 Resumen del Código de Campo de Potenciales
1. **Definición de funciones**  
   - `calcular_potencial`: combina el atractivo y el repulsivo.  
   - `calcular_direccion`: obtiene la dirección de movimiento mediante gradiente.  
   - `visualizar_campo_potencial`: muestra el campo y la trayectoria.  

2. **Parámetros iniciales**  
   - Posición del agente: `(0,0)`  
   - Objetivo: `(10,10)`  
   - Obstáculos: puntos definidos en el espacio.  

3. **Simulación**  
   - En cada paso, el agente se mueve un poco hacia la dirección de menor potencial.  
   - Se guarda la trayectoria.  

4. **Visualización**  
   - El campo de potencial se muestra en colores.  
   - Obstáculos como ❌ negras.  
   - Objetivo como ⚪ rojo.  
   - El recorrido del agente se anima paso a paso.  

---

## 🎯 Finalidad
Este programa ejemplifica cómo un **agente inteligente** puede navegar hacia una meta evitando obstáculos mediante un enfoque de **IA basada en campos de potenciales artificiales**, útil en robótica móvil y planificación de trayectorias.  

