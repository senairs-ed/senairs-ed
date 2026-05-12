# ========================================================
# Introdução aos Testes e Validação de Modelos - Regressão
# Exemplo completo com Random Forest Regressor
# Dataset: California Housing
# ========================================================

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------
# 1. Carregar e explorar o dataset
# -----------------------------------------------
dados = fetch_california_housing()
X = pd.DataFrame(dados.data, columns=dados.feature_names)
y = dados.target

print("Amostras do dataset:")
print(X.head(), "\n")
print("Descrição das variáveis:\n", dados.DESCR.split("\n")[0:10], "\n")
print(f"Número total de amostras: {len(X)}")

# -----------------------------------------------
# 2. Divisão dos dados
# -----------------------------------------------
# Divisão em treino+validação (80%) e teste (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Divisão interna em treino (60%) e validação (20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)
# (Resultado final: 60% treino, 20% validação, 20% teste)

print(f"Tamanho treino: {len(X_train)}")
print(f"Tamanho validação: {len(X_val)}")
print(f"Tamanho teste: {len(X_test)}\n")

# -----------------------------------------------
# 3. Treinamento do modelo
# -----------------------------------------------
modelo = RandomForestRegressor(random_state=42, n_estimators=150)
modelo.fit(X_train, y_train)

# -----------------------------------------------
# 4. Avaliação no conjunto de validação
# -----------------------------------------------
y_pred_val = modelo.predict(X_val)
mse_val = mean_squared_error(y_val, y_pred_val)
rmse_val = np.sqrt(mse_val)

print("=== RESULTADOS NA VALIDAÇÃO ===")
print(f"MSE (Validação): {mse_val:.4f}")
print(f"RMSE (Validação): {rmse_val:.4f}\n")

# -----------------------------------------------
# 5. Avaliação no conjunto de teste
# -----------------------------------------------
y_pred_test = modelo.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)

print("=== RESULTADOS NO TESTE ===")
print(f"MSE (Teste): {mse_test:.4f}")
print(f"RMSE (Teste): {rmse_test:.4f}\n")

# -----------------------------------------------
# 6. Visualizações
# -----------------------------------------------

# Comparação real vs predito - Validação
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_val, y=y_pred_val, color="blue", alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("Valores Reais")
plt.ylabel("Valores Preditos")
plt.title("Validação - Real vs Predito")
plt.show()

# Comparação real vs predito - Teste
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred_test, color="green", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valores Reais")
plt.ylabel("Valores Preditos")
plt.title("Teste - Real vs Predito")
plt.show()