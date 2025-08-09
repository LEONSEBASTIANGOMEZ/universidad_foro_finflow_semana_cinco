# Modelo de Red Neuronal para Predicción de Riesgo Financiero y Modelo de algoritmo adversarial con cuatro competidores para la captacion de clientes con tasas de interes favorable.

Este repositorio contiene una implementación práctica de una red neuronal multicapa (MLP) orientada a la predicción del **riesgo crediticio** de un cliente, utilizando datos sintéticos generados con la librería `Faker`. Este caso de estudio simula el funcionamiento de una solución como una herramienta de análisis financiero automatizado.

con algunos matices de mejora, como los indicadores que afectan negativamente el puntaje de riesgo del usuario como su educacion, monto solicitado, etc; tambien se implemento un nivel de riesgo de cliente de 1 a 10 en contraste con el proyecto anterior que solamente contaba con una expresion booleana.

seguidamente se implemento en el desarrollo la generacion de cuatro competidores frente a la estrategia principal de nuestra organizacion FinFlow la cual implementara un algoritmo `Minimax` para afrontar la competencia y su version mejorada con `Alfa-Beta`.

su sistema cuenta con un control de rentabilidad cuyo objetivo no es la captacion global de todos los clientes si no solo aquellos que representen una significativa rentabilidad para la compania.

al final del ejercicio el sistema arrojara estadisticas de los clientes obtenidos y graficos que puedan demostrar su efectividad frente a diferentes escenarios de riesgo.

---

## 📁 Estructura del proyecto

```
universidad_foro_finflow_semana_tres/
├── app.py              # Codigo del proyecto
├── requirements.txt    # Lista de dependencias necesarias para el proyecto
```

## 📦 Instalación del proyecto

Descarga el codigo fuente mediante el comando git clone https://github.com/LEONSEBASTIANGOMEZ/universidad_foro_finflow_semana_cinco

## 📦 Instalación de dependencias

Se recomienda implementar un entorno virtual con Python 3.11, una vez activado el entorno virtual ejecutar el comando.

```
pip install -r requirements.txt
```

## ▶️ Ejecución del código

```
python app.py
```

Esto realizará las siguientes acciones:

- Generara datos simulados de clientes para su evaluacion de riesgo crediticio

- Entrena una red neuronal para evaluar su riesgo crediticio avanzado.

- Genera competidores los cuales se enfrentaran al algoritmo Minmax presentado

- Evaluar el desempeño del modelo presentando resultados y graficas de las retabilidades captadas frente al general.

- Muestra el reporte de clasificación en consola.

## 📚 Créditos

- Librerías: TensorFlow, scikit-learn, matplotlib