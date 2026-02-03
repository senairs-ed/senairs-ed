# ============================================================
# 1. Imports
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# ============================================================
# 2. Carregando o CIFAR-10
# ============================================================
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Formato do X_train:", X_train.shape)
print("Formato do y_train:", y_train.shape)

# Exibir exemplo
plt.imshow(X_train[0])
plt.title(f"Exemplo CIFAR-10 — classe: {y_train[0][0]}")
plt.axis("off")
plt.show()

# ============================================================
# 3. Normalização e One-Hot Encoding
# ============================================================
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test, 10)

# ============================================================
# 4. Construindo a CNN passo a passo
# ============================================================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ============================================================
# 5. Treinamento
# ============================================================
history = model.fit(X_train, y_train_cat,
                    epochs=10,
                    batch_size=64,
                    validation_split=0.2)

# ============================================================
# 6. Curvas de treinamento
# ============================================================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy (Treino vs Val)")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.legend(["Treino", "Validação"])

plt.subplot(1,2,2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss (Treino vs Val)")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend(["Treino", "Validação"])

plt.show()

# ============================================================
# 7. Avaliação final
# ============================================================
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Accuracy no teste: {acc*100:.2f}%")
print(f"Loss no teste: {loss:.4f}")

# ============================================================
# 9. Predições incorretas
# ============================================================
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = y_test.flatten()

erros = np.where(y_pred != y_true)[0]

print("Total de erros:", len(erros))

# Mostrar 9 erros
plt.figure(figsize=(10,10))
for i, idx in enumerate(erros[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[idx])
    plt.title(f"Pred: {y_pred[idx]} / True: {y_true[idx]}")
    plt.axis("off")
plt.show()
