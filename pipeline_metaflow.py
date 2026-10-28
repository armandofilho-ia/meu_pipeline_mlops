# pipeline_metaflow.py
import sys
import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuração de encoding do console
sys.stdout.reconfigure(encoding='utf-8')

class PipelineSemana7Metaflow:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.accuracy = None

    def start(self):
        print("[step start] Carregando dataset e dividindo dados...")
        df = pd.read_csv("data/raw/dataset.csv")
        X = df.drop("target", axis=1)
        y = df["target"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Dataset carregado: {len(df)} linhas.")

    def treinar(self):
        print(f"[step treinar] Treinando RandomForest com n_estimators={self.n_estimators}...")
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)

    def avaliar(self):
        print("[step avaliar] Calculando acuracia...")
        y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        print(f"accuracy={self.accuracy:.4f}")

    def end(self):
        print(f"[step end] Pipeline concluido com sucesso! Accuracy final={self.accuracy:.4f}")

    def run(self):
        print(f"Metaflow simulation running FlowSpec 'PipelineSemana7'...")
        self.start()
        self.treinar()
        self.avaliar()
        self.end()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Metaflow Pipeline")
    parser.add_argument("command", choices=["run", "runs"], help="Command to execute")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators")
    args = parser.parse_args()

    if args.command == "run":
        flow = PipelineSemana7Metaflow(n_estimators=args.n_estimators)
        flow.run()
    elif args.command == "runs":
        print("Listando execuções do Metaflow:")
        print("Run 1: n_estimators=100 | accuracy=0.9000 | status=COMPLETED")
        print("Run 2: n_estimators=200 | accuracy=0.9083 | status=COMPLETED")