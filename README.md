# 🚗 Vehicle Registrations Forecasting

Este proyecto utiliza un modelo LSTM (Long Short-Term Memory) para realizar pronósticos de registros vehiculares a partir de datos históricos.
Se entrenaron múltiples modelos con el objetivo de comparar su desempeño y capturar los distintos patrones temporales presentes en la serie.

Los resultados y las predicciones se integraron en un dashboard interactivo construido con Streamlit, donde se pueden visualizar:

* Las tendencias históricas

* Las predicciones futuras generadas por los modelos

* Las métricas de evaluación de desempeño

El proyecto busca mostrar cómo los modelos de redes neuronales recurrentes pueden aplicarse al análisis y predicción de datos temporales reales.

## Estructura del Proyecto

- `Data/`: Contiene los datos históricos de registros vehiculares.
- `Outputs/Models/Scalers`: Almacena los escaladores utilizados para normalizar los datos.
- `Outputs/Models/LSTM`: Almacena los modelos LSTM entrenados.
- `Notebooks/`: Incluye notebooks de Jupyter con análisis exploratorio y entrenamiento de modelos.
- `dashboard.py`: Código fuente del dashboard interactivo.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.

## Instalación
1. Clona este repositorio:
   ```bash
   git clone https://github.com/Roberto13-Vil/vehicle-registrations-forecasting.git
   cd vehicle-registrations-forecasting
   ```
2. Crea un entorno virtual e instala las dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Ejecuta el dashboard:
   ```bash
   streamlit run dashboard.py
   ```

¡Disfruta explorando las predicciones de registros vehiculares! 🚘📊


