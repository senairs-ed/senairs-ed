# ===============================================
# Introdução aos Testes e Validação de Modelos
# Exemplo completo com Random Forest no dataset Iris
# ===============================================

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------
# 1. Carregar e explorar o dataset
# -----------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names

df = pd.DataFrame(X, columns=iris.feature_names)
df["classe"] = [labels[i] for i in y]

print("Amostras do dataset:")
print(df.head(), "\n")
print("Distribuição das classes:")
print(df["classe"].value_counts(), "\n")

# -----------------------------------------------
# 2. Divisão dos dados
# -----------------------------------------------
# Primeiro dividimos em treino+validação e teste
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15, stratify=y
)

# Depois dividimos o conjunto temporário em treino e validação
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=15, stratify=y_temp
)
# (Resultado: 60% treino, 20% validação, 20% teste)

print(f"Tamanho treino: {len(X_train)}")
print(f"Tamanho validação: {len(X_val)}")
print(f"Tamanho teste: {len(X_test)}\n")

# -----------------------------------------------
# 3. Treinamento do modelo
# -----------------------------------------------
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# -----------------------------------------------
# 4. Avaliação no conjunto de validação
# -----------------------------------------------
y_pred_val = modelo.predict(X_val)
acc_val = accuracy_score(y_val, y_pred_val)
print("=== RESULTADOS NA VALIDAÇÃO ===")
print(f"Accuracy: {acc_val:.4f}")
print(classification_report(y_val, y_pred_val, target_names=labels))

# Matriz de confusão (validação)
cm_val = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(5,4))
sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Matriz de Confusão - Validação")
plt.ylabel("Verdadeiro")
plt.xlabel("Previsto")
plt.show()

# -----------------------------------------------
# 5. Avaliação no conjunto de teste
# -----------------------------------------------
y_pred_test = modelo.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)
print("=== RESULTADOS NO TESTE ===")
print(f"Accuracy: {acc_test:.4f}")
print(classification_report(y_test, y_pred_test, target_names=labels))

# Matriz de confusão (teste)
cm_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(5,4))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens",
            xticklabels=labels, yticklabels=labels)
plt.title("Matriz de Confusão - Teste")
plt.ylabel("Verdadeiro")
plt.xlabel("Previsto")
plt.show()