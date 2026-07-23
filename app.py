import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

st.set_page_config(
    page_title="Diabetes Intelligence",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# ESTILO
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.main-title {
    font-size: 2.7rem;
    font-weight: 800;
    margin-bottom: 0;
    letter-spacing: -0.04em;
}
.subtitle {
    color: #9da6b5;
    margin-top: 0.2rem;
    font-size: 1.05rem;
}
.hero-box {
    padding: 1.8rem;
    border: 1px solid rgba(120, 130, 160, 0.25);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(44, 62, 90, 0.28), rgba(15, 20, 30, 0.35));
    margin-bottom: 1.2rem;
}
.result-card {
    padding: 1.6rem;
    border-radius: 18px;
    border: 1px solid rgba(120, 130, 160, 0.25);
    background: rgba(24, 29, 40, 0.72);
}
.risk-low {
    border-left: 7px solid #2ecc71;
}
.risk-medium {
    border-left: 7px solid #f1c40f;
}
.risk-high {
    border-left: 7px solid #e74c3c;
}
.section-card {
    padding: 1.2rem 1.35rem;
    border: 1px solid rgba(120, 130, 160, 0.22);
    border-radius: 16px;
    background: rgba(25, 29, 40, 0.55);
    margin-bottom: 1rem;
}
.label-muted {
    color: #9da6b5;
    font-size: 0.9rem;
}
.big-number {
    font-size: 2rem;
    font-weight: 800;
}
[data-testid="stMetric"] {
    border: 1px solid rgba(120, 130, 160, 0.22);
    padding: 1rem;
    border-radius: 15px;
    background: rgba(25, 29, 40, 0.55);
}
.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CARGA Y LIMPIEZA
# -----------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("diabetes.csv")
    df = df.drop_duplicates().copy()

    # En este dataset, varios ceros representan valores clínicos faltantes.
    columnas_cero_invalido = [
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
    ]
    for col in columnas_cero_invalido:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    return df

df = cargar_datos()
TARGET = "Outcome"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age"
]
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

@st.cache_resource
def entrenar_modelos(X_train, y_train):
    modelos = {
        "Árbol de Decisión": DecisionTreeClassifier(
            random_state=42, max_depth=5
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced"
        ),
        "Regresión Logística": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=9))
        ])
    }

    for modelo in modelos.values():
        modelo.fit(X_train, y_train)

    return modelos

modelos = entrenar_modelos(X_train, y_train)

def evaluar_modelos():
    filas = []
    predicciones = {}

    for nombre, modelo in modelos.items():
        pred = modelo.predict(X_test)
        predicciones[nombre] = pred

        filas.append({
            "Modelo": nombre,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0)
        })

    resultados = pd.DataFrame(filas).sort_values("F1", ascending=False)
    return resultados, predicciones

resultados, predicciones = evaluar_modelos()
mejor_nombre = resultados.iloc[0]["Modelo"]
mejor_modelo = modelos[mejor_nombre]
mejor_pred = predicciones[mejor_nombre]
mejor_accuracy = resultados.iloc[0]["Accuracy"]

# -----------------------------
# DATOS HISTÓRICOS
# -----------------------------
URL_HISTORICO = "https://ourworldindata.org/grapher/diabetes-prevalence-who-gho.csv?v=1&csvType=full&useColumnShortNames=false"

@st.cache_data
def cargar_historico():
    rutas = ["diabetes_historico_who.csv", URL_HISTORICO]
    ultimo_error = None
    for ruta in rutas:
        try:
            datos = pd.read_csv(ruta)
            if datos.empty:
                continue
            columnas_base = {"Entity", "Code", "Year"}
            if not columnas_base.issubset(datos.columns):
                continue
            valor_cols = [c for c in datos.columns if c not in columnas_base]
            if not valor_cols:
                continue
            valor = valor_cols[0]
            datos = datos.rename(columns={
                "Entity": "Pais",
                "Code": "Codigo",
                "Year": "Anio",
                valor: "Prevalencia"
            })
            datos["Anio"] = pd.to_numeric(datos["Anio"], errors="coerce")
            datos["Prevalencia"] = pd.to_numeric(datos["Prevalencia"], errors="coerce")
            datos = datos.dropna(subset=["Pais", "Anio", "Prevalencia"]).copy()
            datos["Anio"] = datos["Anio"].astype(int)
            return datos
        except Exception as error:
            ultimo_error = error
    return None

df_historico = cargar_historico()

# -----------------------------
# MENÚ
# -----------------------------
st.sidebar.title("🩺 Diabetes Intelligence")
pagina = st.sidebar.radio(
    "Navegación",
    [
        "🏠 Inicio",
        "📊 Análisis exploratorio",
        "🤖 Predicción",
        "📈 Comparación de modelos",
        "📈 Tendencia histórica",
        "📋 Dataset",
        "ℹ️ Acerca del proyecto"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Este sistema es educativo y no sustituye un diagnóstico médico."
)

# -----------------------------
# INICIO
# -----------------------------
if pagina == "🏠 Inicio":
    positivos = int(df[TARGET].sum())
    negativos = int(len(df) - positivos)

    st.markdown("""
    <div class="hero-box">
        <div class="main-title">🩺 Diabetes Intelligence</div>
        <div class="subtitle">
            Sistema inteligente para análisis, clasificación y regresión aplicada a diabetes
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Pacientes", f"{len(df):,}")
    c2.metric("🧬 Variables", len(X.columns))
    c3.metric("🏆 Mejor modelo", mejor_nombre)
    c4.metric("🎯 Accuracy", f"{mejor_accuracy:.2%}")

    c5, c6, c7 = st.columns(3)
    c5.metric("🔴 Con diabetes", positivos, f"{positivos/len(df):.1%}")
    c6.metric("🟢 Sin diabetes", negativos, f"{negativos/len(df):.1%}")
    c7.metric("🎂 Edad promedio", f"{df['Age'].mean():.1f} años")

    st.markdown("### Estado del sistema")
    s1, s2, s3, s4 = st.columns(4)
    s1.success("✔ Dataset cargado")
    s2.success("✔ Datos limpiados")
    s3.success("✔ Modelos entrenados")
    s4.success("✔ Predicción disponible")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Distribución de diagnósticos")
        conteo = df[TARGET].value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.bar(["Sin diabetes", "Con diabetes"], conteo.values)
        ax.set_ylabel("Número de pacientes")
        ax.set_title("Casos registrados")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Variables más importantes")
        rf = modelos["Random Forest"]
        importancia = pd.Series(
            rf.feature_importances_, index=X.columns
        ).sort_values(ascending=True)

        fig, ax = plt.subplots()
        ax.barh(importancia.index, importancia.values)
        ax.set_xlabel("Importancia")
        ax.set_title("Importancia según Random Forest")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.info(
        "Usa el menú lateral para explorar los datos, comparar modelos, "
        "realizar predicciones o revisar el modelo de regresión."
    )

# -----------------------------
# ANÁLISIS
# -----------------------------
elif pagina == "📊 Análisis exploratorio":
    st.title("📊 Análisis exploratorio")

    variable = st.selectbox(
        "Selecciona una variable para analizar",
        X.columns,
        index=list(X.columns).index("Glucose")
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Promedio", f"{df[variable].mean():.2f}")
    c2.metric("Mediana", f"{df[variable].median():.2f}")
    c3.metric("Mínimo", f"{df[variable].min():.2f}")
    c4.metric("Máximo", f"{df[variable].max():.2f}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribuciones", "Relaciones", "Correlación", "Valores atípicos"
    ])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.hist(df[variable], bins=20, edgecolor="black")
            ax.set_title(f"Distribución de {variable}")
            ax.set_xlabel(variable)
            ax.set_ylabel("Frecuencia")
            st.pyplot(fig)

        with col2:
            grupos = [
                df.loc[df[TARGET] == 0, variable],
                df.loc[df[TARGET] == 1, variable]
            ]
            fig, ax = plt.subplots()
            ax.boxplot(grupos, tick_labels=["Sin diabetes", "Con diabetes"])
            ax.set_title(f"{variable} según diagnóstico")
            ax.set_ylabel(variable)
            st.pyplot(fig)

    with tab2:
        eje_x = st.selectbox("Eje X", X.columns, index=list(X.columns).index("Age"))
        eje_y = st.selectbox("Eje Y", X.columns, index=list(X.columns).index("Glucose"))

        fig, ax = plt.subplots()
        for clase, etiqueta in [(0, "Sin diabetes"), (1, "Con diabetes")]:
            datos_clase = df[df[TARGET] == clase]
            ax.scatter(
                datos_clase[eje_x], datos_clase[eje_y],
                alpha=0.6, label=etiqueta
            )
        ax.set_xlabel(eje_x)
        ax.set_ylabel(eje_y)
        ax.legend()
        ax.set_title(f"{eje_x} vs {eje_y}")
        st.pyplot(fig)

    with tab3:
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(corr, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        fig.colorbar(im, ax=ax)
        ax.set_title("Mapa de correlación")
        st.pyplot(fig)

    with tab4:
        q1 = df[variable].quantile(0.25)
        q3 = df[variable].quantile(0.75)
        iqr = q3 - q1
        inferior = q1 - 1.5 * iqr
        superior = q3 + 1.5 * iqr
        atipicos = df[(df[variable] < inferior) | (df[variable] > superior)]

        st.metric("Valores atípicos detectados", len(atipicos))
        st.dataframe(atipicos[[variable, TARGET]], use_container_width=True)

# -----------------------------
# PREDICCIÓN
# -----------------------------
elif pagina == "🤖 Predicción":
    st.markdown("""
    <div class="hero-box">
        <div class="main-title">🤖 Evaluación clínica</div>
        <div class="subtitle">
            Herramienta académica de apoyo para personal médico.
        </div>
    </div>
    """, unsafe_allow_html=True)

    

    with st.expander("ℹ️ Guía de captura"):
        st.markdown("""
        - **Glucosa:** concentración plasmática en mg/dL.
        - **Presión arterial:** presión diastólica en mmHg.
        - **Insulina:** concentración sérica en μU/mL.
        - **Peso y estatura:** se utilizan para calcular el IMC automáticamente.
        - **Antecedentes familiares:** se contestan mediante preguntas; la aplicación
          genera un índice aproximado para poder utilizar el modelo Pima.
        """)

    st.markdown("### 👤 Datos generales")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Edad (años)", min_value=18, max_value=100,
                              value=int(round(df["Age"].median())), step=1)
    with c2:
        sexo = st.selectbox("Sexo", ["Femenino", "Masculino"])
    with c3:
        if sexo == "Femenino":
            pregnancies = st.number_input(
                "Número de embarazos", min_value=0, max_value=20,
                value=int(round(df["Pregnancies"].median())), step=1
            )
        else:
            pregnancies = 0
            st.number_input("Número de embarazos", value=0, disabled=True,
                            help="Se establece en 0 para pacientes masculinos.")

    st.markdown("### 🧪 Mediciones clínicas")
    c1, c2, c3 = st.columns(3)
    with c1:
        glucose = st.number_input("Glucosa (mg/dL)", min_value=40, max_value=300,
                                  value=int(round(df["Glucose"].median())), step=1)
    with c2:
        blood_pressure = st.number_input(
            "Presión diastólica (mmHg)", min_value=30.0, max_value=150.0,
            value=float(round(df["BloodPressure"].median(), 1)), step=1.0
        )
    with c3:
        insulin = st.number_input("Insulina (μU/mL)", min_value=10.0, max_value=900.0,
                                  value=float(round(df["Insulin"].median(), 1)), step=1.0)

    st.markdown("### ⚖️ Cálculo automático del IMC")
    c1, c2, c3 = st.columns(3)
    with c1:
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0,
                               value=70.0, step=0.5)
    with c2:
        estatura_cm = st.number_input("Estatura (cm)", min_value=120.0, max_value=220.0,
                                      value=165.0, step=1.0)
    bmi = peso / ((estatura_cm / 100) ** 2)
    with c3:
        st.metric("IMC calculado", f"{bmi:.1f}")
        if bmi < 18.5:
            st.caption("Clasificación descriptiva: bajo peso")
        elif bmi < 25:
            st.caption("Clasificación descriptiva: rango normal")
        elif bmi < 30:
            st.caption("Clasificación descriptiva: sobrepeso")
        else:
            st.caption("Clasificación descriptiva: obesidad")

    st.markdown("### 🧬 Antecedentes familiares")
    familiar = st.radio(
        "¿El paciente tiene familiares con diabetes?",
        ["No", "Sí"], horizontal=True
    )

    familiares = []
    edad_diagnostico = "No se conoce"
    if familiar == "Sí":
        familiares = st.multiselect(
            "Seleccione los familiares con diabetes",
            ["Madre", "Padre", "Hermana o hermano", "Abuela o abuelo", "Otros familiares"]
        )
        edad_diagnostico = st.selectbox(
            "¿Alguno fue diagnosticado antes de los 45 años?",
            ["No se conoce", "No", "Sí"]
        )

    # Índice aproximado para adaptar respuestas comprensibles a la variable
    pedigree = 0.10
    pesos_familia = {
        "Madre": 0.25,
        "Padre": 0.25,
        "Hermana o hermano": 0.20,
        "Abuela o abuelo": 0.10,
        "Otros familiares": 0.05,
    }
    for fam in familiares:
        pedigree += pesos_familia[fam]
    if edad_diagnostico == "Sí":
        pedigree += 0.15
    pedigree = min(pedigree, 2.50)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Índice familiar estimado", f"{pedigree:.2f}")
    with c2:
        st.caption(
            "Este valor se genera automáticamente."
        )

    enviado = st.button("🔍 Analizar paciente", use_container_width=True, type="primary")

    if enviado:
        nuevo = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age
        }])

        prob = float(mejor_modelo.predict_proba(nuevo)[0][1])

        if prob < 0.30:
            nivel = "Riesgo bajo"
            icono = "🟢"
            clase_css = "risk-low"
            mensaje = "Los datos muestran una probabilidad baja dentro de este modelo."
        elif prob < 0.60:
            nivel = "Riesgo moderado"
            icono = "🟡"
            clase_css = "risk-medium"
            mensaje = "Existen algunos indicadores que requieren seguimiento."
        else:
            nivel = "Riesgo alto"
            icono = "🔴"
            clase_css = "risk-high"
            mensaje = "Los datos presentan varios indicadores asociados con mayor riesgo."

        st.markdown("---")
        st.markdown(
            f"""
            <div class="result-card {clase_css}">
                <div class="label-muted">RESULTADO DEL PACIENTE</div>
                <div class="big-number">{icono} {nivel}</div>
                <p>{mensaje}</p>
                <b>Modelo utilizado:</b> {mejor_nombre}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Probabilidad estimada")
        st.progress(int(prob * 100))
        st.markdown(f"## {prob:.1%}")

        factores = [
            ("Glucosa", f"{glucose} mg/dL", "Elevada" if glucose >= 140 else "Intermedia" if glucose >= 100 else "Baja"),
            ("IMC", f"{bmi:.1f}", "Elevado" if bmi >= 30 else "Intermedio" if bmi >= 25 else "Bajo"),
            ("Edad", age, "45 años o más" if age >= 45 else "Menor de 45 años"),
            ("Antecedentes", ", ".join(familiares) if familiares else "Ninguno reportado", f"Índice automático: {pedigree:.2f}")
        ]
        st.markdown("### 🔎 Resumen de factores")
        st.dataframe(pd.DataFrame(factores, columns=["Indicador", "Valor", "Interpretación"]),
                     use_container_width=True, hide_index=True)

        st.markdown("### 🩺 Recomendación general")
        if prob < 0.30:
            st.success("Mantener controles preventivos y hábitos saludables.")
        elif prob < 0.60:
            st.warning("Valorar seguimiento clínico y pruebas complementarias según criterio médico.")
        else:
            st.error("Se recomienda valoración médica integral y pruebas diagnósticas confirmatorias.")

# -----------------------------
# COMPARACIÓN DE MODELOS
# -----------------------------
elif pagina == "📈 Comparación de modelos":
    st.title("📈 Comparación de modelos de clasificación")

    tabla = resultados.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        tabla[col] = tabla[col].map(lambda x: f"{x:.2%}")

    st.dataframe(tabla, use_container_width=True, hide_index=True)
    st.success(f"Mejor modelo según F1: {mejor_nombre}")

    metricas_plot = resultados.set_index("Modelo")[[
        "Accuracy", "Precision", "Recall", "F1"
    ]]
    fig, ax = plt.subplots(figsize=(10, 5))
    metricas_plot.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Puntuación")
    ax.set_title("Comparación de métricas")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)

    st.markdown("### ¿Cómo mejoran las predicciones con más datos?")
    st.write(
        """
        Representa cómo cambia el desempeño del modelo cuando aumenta la cantidad de
        datos disponibles para entrenarlo.
        """
    )

    tamanos, train_scores, val_scores = learning_curve(
        RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        ),
        X, y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tamanos, train_mean, marker="o", label="Entrenamiento")
    ax.plot(tamanos, val_mean, marker="o", label="Validación")
    ax.set_xlabel("Cantidad de registros de entrenamiento")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Curva de aprendizaje del Random Forest")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# -----------------------------
# REGRESIÓN
# -----------------------------
elif pagina == "📈 Tendencia histórica":
    st.markdown("""
    <div class="hero-box">
        <div class="main-title">📈 Regresión histórica</div>
        <div class="subtitle">
            Evolución y proyección de la prevalencia de diabetes por país.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df_historico is None:
        st.error("No fue posible cargar el dataset histórico.")
        st.write("Descarga el archivo CSV oficial y guárdalo como `diabetes_historico_who.csv` en la misma carpeta de `app.py`.")
        st.code(URL_HISTORICO)
        st.stop()

    paises = sorted(df_historico["Pais"].dropna().unique().tolist())
    pais_default = paises.index("Mexico") if "Mexico" in paises else 0
    pais = st.selectbox("🌎 Selecciona un país", paises, index=pais_default)

    datos_pais = (
        df_historico[df_historico["Pais"] == pais][["Anio", "Prevalencia"]]
        .drop_duplicates(subset=["Anio"])
        .sort_values("Anio")
    )

    if len(datos_pais) < 10:
        st.warning(
            f"{pais} solo tiene {len(datos_pais)} años disponibles. "
            "Selecciona otro país con al menos 10 registros históricos."
        )
        st.stop()

    X_hist = datos_pais[["Anio"]]
    y_hist = datos_pais["Prevalencia"]

    # Separamos cronológicamente:
    # los primeros 80% de años se usan para entrenamiento
    # y los últimos 20% para prueba.
    corte = max(3, int(len(datos_pais) * 0.8))

    Xh_train = X_hist.iloc[:corte]
    Xh_test = X_hist.iloc[corte:]

    yh_train = y_hist.iloc[:corte]
    yh_test = y_hist.iloc[corte:]

    # Modelo utilizado para evaluar el rendimiento
    modelo_hist = LinearRegression()
    modelo_hist.fit(Xh_train, yh_train)

    pred_test = modelo_hist.predict(Xh_test)

    # Métricas de evaluación
    mae_hist = mean_absolute_error(yh_test, pred_test)
    rmse_hist = np.sqrt(mean_squared_error(yh_test, pred_test))

    if len(yh_test) >= 2:
        r2_hist = r2_score(yh_test, pred_test)
    else:
        r2_hist = float("nan")

    # Información general de la serie histórica
    anio_actual = datetime.now().year
    ultimo_anio = int(datos_pais["Anio"].max())

    ultimo_valor = float(
        datos_pais.loc[
            datos_pais["Anio"].idxmax(),
            "Prevalencia"
        ]
    )

    primer_valor = float(datos_pais.iloc[0]["Prevalencia"])
    cambio = ultimo_valor - primer_valor

    # Tarjetas informativas
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Primer año",
        int(datos_pais["Anio"].min())
    )

    c2.metric(
        "Último año con datos",
        ultimo_anio
    )

    c3.metric(
        "Última prevalencia",
        f"{ultimo_valor:.2f}%"
    )

    c4.metric(
        "Cambio histórico",
        f"{cambio:+.2f} puntos"
    )

    # Métricas del modelo
    st.markdown("### Evaluación del modelo")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "MAE",
        f"{mae_hist:.3f}"
    )

    m2.metric(
        "RMSE",
        f"{rmse_hist:.3f}"
    )

    m3.metric(
        "R²",
        "No disponible"
        if np.isnan(r2_hist)
        else f"{r2_hist:.3f}"
    )

    # Entrenamos otro modelo usando todos los datos históricos.
    # Este modelo se utiliza para realizar la proyección futura.
    modelo_final = LinearRegression()
    modelo_final.fit(X_hist, y_hist)

    # Proyección desde el año actual hasta los siguientes cuatro años.
    # Por ejemplo, si estamos en 2026:
    # 2026, 2027, 2028, 2029 y 2030.
    anios_futuros = np.arange(
        anio_actual,
        anio_actual + 5
    ).reshape(-1, 1)

    pred_futuro = modelo_final.predict(
        pd.DataFrame(anios_futuros, columns=["Anio"])
    )

    # Evitamos valores negativos, porque una prevalencia
    # no puede ser menor que cero.
    pred_futuro = np.maximum(pred_futuro, 0)

    # Tabla de proyección
    futuro = pd.DataFrame({
        "Año": anios_futuros.flatten(),
        "Prevalencia estimada (%)": np.round(pred_futuro, 2)
    })

    # Gráfica
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        datos_pais["Anio"],
        datos_pais["Prevalencia"],
        marker="o",
        label="Datos históricos"
    )

    ax.plot(
        Xh_test["Anio"],
        pred_test,
        marker="o",
        linestyle="--",
        label="Predicción sobre datos de prueba"
    )

    ax.plot(
        anios_futuros.flatten(),
        pred_futuro,
        marker="o",
        linestyle="--",
        label="Proyección futura"
    )

    # Línea que marca el inicio de la proyección actual
    ax.axvline(
        anio_actual,
        linestyle=":",
        alpha=0.7,
        label=f"Inicio de proyección ({anio_actual})"
    )

    ax.set_xlabel("Año")
    ax.set_ylabel("Prevalencia (%)")
    ax.set_title(f"Evolución de la diabetes en {pais}")
    ax.grid(alpha=0.25)
    ax.legend()

    st.pyplot(fig)

    # Tabla con los años futuros
    st.markdown("### Proyección para los próximos cinco años")

    st.dataframe(
        futuro,
        use_container_width=True,
        hide_index=True
    )

    # Interpretación de la tendencia
    pendiente = modelo_final.coef_[0]

    if pendiente > 0:
        tendencia = "creciente"
    elif pendiente < 0:
        tendencia = "decreciente"
    else:
        tendencia = "estable"

    st.info(
        f"Entre {int(datos_pais['Anio'].min())} y {ultimo_anio}, "
        f"{pais} presenta una tendencia lineal {tendencia}. "
        f"La pendiente estimada es de {pendiente:+.3f} "
        f"puntos porcentuales por año."
    )

    # Advertencia por la distancia entre el último dato y el año actual
    if ultimo_anio < anio_actual:
        st.warning(
            f"El último dato histórico disponible corresponde a {ultimo_anio}. "
            f"Las estimaciones de {anio_actual} a {anio_actual + 4} son "
            "proyecciones académicas obtenidas al extender la tendencia histórica. "
            "No representan datos observados ni un pronóstico epidemiológico oficial."
        )
    else:
        st.warning(
            "La proyección es educativa y supone que la tendencia histórica continúa. "
            "No es un pronóstico médico ni epidemiológico oficial."
        )

    # Descarga del archivo CSV
    st.download_button(
        "📥 Descargar proyección",
        data=futuro.to_csv(index=False).encode("utf-8"),
        file_name=f"proyeccion_diabetes_{pais.replace(' ', '_')}.csv",
        mime="text/csv"
    )

# -----------------------------
# DATASET
# -----------------------------
elif pagina == "📋 Dataset":
    st.title("📋 Dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", df.shape[0])
    c2.metric("Columnas", df.shape[1])
    c3.metric("Nulos después de ETL", int(df.isna().sum().sum()))
    c4.metric("Duplicados", int(df.duplicated().sum()))

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Descargar dataset limpio",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="diabetes_limpio.csv",
        mime="text/csv"
    )

    st.markdown("### Estadísticas descriptivas")
    st.dataframe(df.describe().T, use_container_width=True)

# -----------------------------
# ACERCA
# -----------------------------
else:
    st.title("ℹ️ Acerca del proyecto")

    st.markdown("""
### 🩺 Diabetes Intelligence

**Descripción del proyecto**

Diabetes Intelligence es una aplicación desarrollada con Python y Streamlit que utiliza algoritmos de aprendizaje supervisado para apoyar el análisis de pacientes con riesgo de diabetes.

El sistema permite capturar información clínica del paciente, calcular automáticamente el índice de masa corporal (IMC), generar un índice aproximado de antecedentes familiares y estimar la probabilidad de presentar diabetes mediante diferentes modelos de clasificación.

Además, incluye herramientas para explorar el conjunto de datos, comparar el rendimiento de distintos algoritmos de aprendizaje supervisado y analizar la evolución histórica de la prevalencia de diabetes por país utilizando un modelo de regresión lineal.

### Funciones principales

- Evaluación clínica de pacientes.
- Cálculo automático del IMC.
- Estimación automática del índice de antecedentes familiares.
- Predicción del riesgo de diabetes.
- Comparación de modelos de aprendizaje supervisado.
- Análisis exploratorio del conjunto de datos.
- Tendencia histórica y proyección de la prevalencia de diabetes.
- Descarga del dataset limpio y de las proyecciones generadas.

### Tecnologías utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

### Modelos implementados

**Clasificación**
- Árbol de Decisión
- Random Forest
- Regresión Logística
- KNN

**Regresión**
- Regresión Lineal

### Proceso general

1. Carga y limpieza del conjunto de datos.
2. Entrenamiento de los modelos.
3. Comparación de métricas.
4. Evaluación clínica del paciente.
5. Generación de predicciones y visualización de resultados.

""")

    st.warning(
        "Proyecto académico. No sustituye evaluación, diagnóstico ni tratamiento médico."
    )