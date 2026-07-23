# Diabetes Intelligence

Aplicación educativa en Streamlit con dos módulos de aprendizaje supervisado:

- Clasificación del riesgo de diabetes con el dataset Pima Indians Diabetes.
- Regresión lineal de la prevalencia histórica por país usando datos anuales de la OMS (1980-2014), procesados por Our World in Data.

## Ejecución

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Si el histórico no se descarga automáticamente:

```powershell
python DESCARGAR_DATASET.py
```


