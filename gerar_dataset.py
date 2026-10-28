# gerar_dataset.py
import pandas as pd
from sklearn.datasets import make_classification

# Gerando dataset sintético de classificação (600 amostras, 8 features)
X, y = make_classification(
    n_samples=600,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
df["target"] = y

# Salvando no caminho esperado pelo pipeline
df.to_csv("data/raw/dataset.csv", index=False)

print("Dataset criado com sucesso em data/raw/dataset.csv")
print("Shape:", df.shape)
print("\nPrimeiras 5 linhas:")
print(df.head())