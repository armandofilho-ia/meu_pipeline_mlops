# executar_pipeline.py
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Configura codificação do console
sys.stdout.reconfigure(encoding='utf-8')

print("--- Iniciando Pipeline MLOps ---")

# 1. Ingestão de dados
print("[STEP 1/5] Ingestao de dados...")
dataset_path = os.path.join("data", "raw", "dataset.csv")
if not os.path.exists(dataset_path):
    print(f"Erro: O arquivo {dataset_path} nao existe.")
    sys.exit(1)

df = pd.read_csv(dataset_path)
df = df.dropna(how="all")
df["target"] = df["target"].astype(int)
print(f"[INGEST] Linhas={len(df)} | Colunas={df.shape[1]}")

# 2. Divisão (Split)
print("[STEP 2/5] Divisao dos dados...")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Transformação (Transform)
print("[STEP 3/5] Aplicando pre-processamento...")
transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
X_train_trans = transformer.fit_transform(X_train)
X_test_trans = transformer.transform(X_test)

# Definir Experimento no MLflow
mlflow.set_experiment("mlops_semana07_local")

# 4. Treino, Avaliação e Tracking
print("[STEP 4/5] Treinando estimador e avaliando...")
with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_trans, y_train)

    y_pred = model.predict(X_test_trans)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy_score", acc)

    print(f"\n--- RESULTADOS DO RUN ---")
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy Score: {acc:.4f}")

    # 5. Critério de Validação e Registro no Model Registry
    THRESHOLD = 0.80
    print(f"\n[STEP 5/5] Avaliando criterio minimo (Threshold = {THRESHOLD})...")
    if acc >= THRESHOLD:
        print("[OK] Criterio atingido! Registrando modelo no MLflow Model Registry...")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="modelo-semana7",
            registered_model_name="modelo-semana7",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        print("Modelo 'modelo-semana7' registrado com sucesso!")
    else:
        print("[X] Modelo reprovado no criterio minimo. Nao sera registrado.")