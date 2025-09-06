# 🤖 Conceptos de Agentes Inteligentes y Modelos de Toma de Decisiones

## 🧠 Agente Inteligente
Un **agente inteligente** es una entidad capaz de reconocer su entorno y tomar decisiones basadas en las percepciones del ambiente por medio de **sensores**. Estas percepciones son interpretadas de manera racional, lo que permite responder a los estímulos recibidos.  

Aunque comúnmente se denomina “inteligente”, es más apropiado hablar de un **agente racional**, ya que no entiende realmente las percepciones, sino que responde al entorno de acuerdo con su interpretación de la información de entrada y el procesamiento previo.

---

## ⚡ Cambio de Potencial
El **modelo de cambio de potencial** permite planear rutas o trayectorias en un espacio definido, incluso cuando las características puntuales son desconocidas.  

Es utilizado frecuentemente en **robots móviles**, ya que agiliza la toma de decisiones de movimiento. Se establecen fuerzas **repulsoras** (causadas por los obstáculos) y **atractoras** (hacia la meta), y la suma de dichas fuerzas constituye el campo de trabajo.  

El robot se modela como una partícula cargada dentro de este campo:  
- Con cada movimiento, se recalcula el campo.  
- El equilibrio entre fuerzas garantiza la movilidad del robot sin chocar con los obstáculos.  
- Los **gradientes** permiten calcular la trayectoria adecuada.  

🔴 **Problema común:**  
Dependiendo del tamaño de las vías de tránsito o de la cercanía a los obstáculos, el robot puede entrar en un estado de **oscilación (mínimos locales)**, perdiendo fluidez en su movimiento.  

✅ **Soluciones posibles:**  
- Agregar pequeñas perturbaciones que ayuden al robot a salir del estado de estancamiento.  

---

## 🔄 Modelo BDI (Belief–Desire–Intention)
El **modelo BDI** (Creencias–Deseos–Intenciones) busca simplificar el comportamiento humano para aplicarlo a la toma de decisiones en sistemas inteligentes.  

En robótica, la información para la toma de decisiones proviene de los sensores, que permiten la interacción del robot con el mundo real. Este modelo combina **arquitecturas reactivas, deliberativas y bio-inspiradas**.  

### 📌 Componentes del modelo:
- **Creencias (Beliefs):** Información actual del entorno captada por los sensores.  
- **Deseos (Desires):** Objetivos que el robot desea alcanzar.  
- **Intenciones (Intentions):** El “cómo” y “cuándo” el robot cumplirá los objetivos definidos.  

El estado de **intención** surge como resultado de combinar las creencias y los deseos, lo que da lugar a acciones concretas del robot.  
