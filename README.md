# MLOps Challenge: California Housing Price Predictor

Este es un proyecto de MLOps para predecir precios de viviendas en California.

Autor: Mauricio Orozco Nava
Curso: MLOps - MIOTI Tech & Business School

## Dataset

California Housing de scikit-learn:
- 20,640 registros
- 8 variables predictoras
- 1 variable target (precio medio de vivienda)

## Crear el entorno del proyecto

conda create -n production_mlops python=3.13 -y
conda activate production_mlops
pip install mlflow dvc scikit-learn pandas numpy fastapi uvicorn

## Estructura del proyecto

mlops_challenge/
├── data/raw/              # Dataset (versionado con DVC)
├── models/                # Modelo guardado (.pkl)
├── mlruns/                # Experimentos MLflow
├── src/
│   ├── load_data.py       # Descarga del dataset
│   ├── train.py           # Entrenamiento y logging
│   └── api.py             # API FastAPI
├── dvc.yaml               # Pipeline DVC
└── requirements.txt       # Dependencias

## Cargar datos

python src/load_data.py

## Entrenar modelos

python src/train.py

## MLflow

Para ver los experimentos:

mlflow ui --port 5001 --backend-store-uri file:./mlruns

Abrir http://localhost:5001

## API con FastAPI

Ejecutar la API:

python -m uvicorn src.api:app --reload --port 8000

Probar predicción con curl:

curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.9, "AveBedrms": 1.0, "Population": 322, "AveOccup": 2.5, "Latitude": 37.88, "Longitude": -122.23}'

Respuesta: {"prediction": 4.3114, "price_usd": "$431,145"}
