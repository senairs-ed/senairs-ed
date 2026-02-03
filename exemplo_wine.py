# ===============================================================
#  MACHINE LEARNING COM O WINE QUALITY DATASET (WHITE WINE)
# ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===============================================================
# 1. CARREGAR O DATASET
# ===============================================================

# Dataset público (UCI). Ele já está em formato CSV.
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-white.csv"
)

# O separador do arquivo é ";"
df = pd.read_csv(url, sep=";")

print("\nPrimeiras linhas do dataset:")
print(df.head())

print("\nDescrição estatística:")
print(df.describe())


# ===============================================================
# 2. DEFINIR FEATURES (X) E TARGET (y)
# ===============================================================

X = df.drop("quality", axis=1)   # 11 variáveis físico-químicas
y = df["quality"]                # nota sensorial do vinho


# ===============================================================
# 3. PADRONIZAÇÃO (Z-SCORE)
# ===============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pd.DataFrame(X_scaled, columns = X.columns).describe()

# ===============================================================
# 4. TREINO / TESTE
# ===============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42
)

print("\nTamanho do conjunto de treino:", X_train.shape)
print("Tamanho do conjunto de teste :", X_test.shape)


# ===============================================================
# 5. TREINAMENTO DO MODELO
# ===============================================================

model = LinearRegression()
model.fit(X_train, y_train)

print("\nCoeficientes do modelo (pesos):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:25s} -> {coef:.4f}")

print("\nIntercepto (bias):", model.intercept_)


# ===============================================================
# 6. AVALIAÇÃO DO MODELO
# ===============================================================

y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n--- Métricas de Avaliação ---")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")


# ===============================================================
# 7. VISUALIZAÇÃO — PREVISÃO vs REAL
# ===============================================================

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel("Valor Real (Qualidade)")
plt.ylabel("Valor Predito (Qualidade)")
plt.title("Regressão — Wine Quality (White)")
plt.grid(True)
plt.show()


# ===============================================================
# 8. DISTRIBUIÇÃO DOS RESÍDUOS
# ===============================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=25)
plt.title("Distribuição dos Resíduos")
plt.xlabel("Erro (y real - y predito)")
plt.ylabel("Frequência")
plt.grid(True)
plt.show()
