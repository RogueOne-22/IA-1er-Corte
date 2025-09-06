# ⚡ Campo de Potenciales Artificiales Resumen del Código: Campo de Potenciales Artificiales

Este código implementa un modelo de **campo de potenciales artificiales** para que un agente se desplace hacia un objetivo evitando obstáculos.  

## Flujo del Código

1. **Cálculo del Potencial Total**  
   - Se combina: Un potencial atractivo (atrae al agente hacia el objetivo) y Un potencial repulsivo (evita que el agente choque con obstáculos).  

2. **Gradiente y Dirección de Movimiento**  
   - Se calcula el gradiente del campo en cada posición.  
   - El agente se mueve en la dirección de menor potencial.  

3. **Simulación del Movimiento**  
   - El agente comienza en (0,0).  
   - En cada paso, se actualiza la posición sumando un pequeño desplazamiento en la dirección calculada.  
   - Se almacena la trayectoria recorrida.  

4. **Visualización con Matplotlib**  
   - Se dibuja el campo de potencial con un mapa de contornos.  
   - Obstáculos se muestran con "X" negras.  
   - El objetivo se muestra con un círculo rojo.  
   - La trayectoria del agente se anima paso a paso.

🎯 Resultado

  *El agente se mueve desde el inicio (0,0) hasta el objetivo (10,10).
  *Es desviado por obstáculos cercanos gracias al campo repulsivo.
  ```
python
K_atractivo = 0.5   # fuerza hacia el objetivo
K_repulsivo = 1.0   # fuerza de obstáculos
radio_repulsion = 5.0
```
  * La animación muestra el campo de potencial y la trayectoria del robot.

## Finalidad

El programa permite entender cómo un robot puede navegar en un entorno desconocido con obstáculos, utilizando fuerzas artificiales que lo guían hacia su meta evitando colisiones.  
