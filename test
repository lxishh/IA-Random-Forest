import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Paso 1: Cargar el conjunto de datos de Titanic
data = pd.read_csv('/content/train.csv')


# Paso 2: Preprocesamiento de los datos

# Seleccionar las columnas relevantes
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]

# Convertir 'Sex' a valores numéricos: male = 0, female = 1
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Convertir 'Embarked' a valores numéricos: C = 0, Q = 1, S = 2
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Rellenar los valores nulos en 'Age' con la mediana
data['Age'] = data['Age'].fillna(data['Age'].median())

# Eliminar las filas con valores nulos en 'Embarked' (puedes usar otros métodos si lo prefieres)
data.dropna(subset=['Embarked'], inplace=True)


# Paso 3: Separar las características y la variable objetivo
X = data.drop('Survived', axis=1)
y = data['Survived']


# Paso 4: Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Paso 5: Inicializar y entrenar el clasificador
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Paso 6: Hacer predicciones
y_pred = model.predict(X_test)


# Paso 7: Generar un reporte de clasificación
report = classification_report(y_test, y_pred, target_names=['No Sobrevive', 'Sobrevive'])
#Imprimimos pero en español
print(report.replace("precision", "Precisión")
              .replace("recall", "Sensibilidad")
              .replace("f1-score", "Puntuación F1")
              .replace("support", "Soporte")
              .replace("accuracy", "Exactitud"))

# Paso 8: Generar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicho')
plt.title('Matriz de Confusión')
plt.show()


# Paso 9: Realizar una predicción con un nuevo dato (ejemplo de pasajero)
nuevo_dato = pd.DataFrame({
    'Pclass': [3],
    'Sex': [0],  # Hombre
    'Age': [22],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [7.25],
    'Embarked': [2]  # S (Southampton)
})

# Hacer la predicción
prediccion = model.predict(nuevo_dato)

# Mostrar resultado
resultado = "Sobrevive" if prediccion[0] == 1 else "No Sobrevive"
print("El nuevo dato indica:", resultado)
