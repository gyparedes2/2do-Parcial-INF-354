import numpy as np
import pandas as pd

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide
def sigmoid_derivada(x):
    return x * (1 - x)

# Lectura del conjunto de datos Iris
datos_iris = pd.read_csv('iris.csv')

# Normalización de los datos
datos = datos_iris.iloc[:, :-1]
datos = (datos - datos.mean()) / datos.std()

# Agregamos una columna de unos para el sesgo
datos['sesgo'] = 1

# Convertimos los datos a matriz numpy
X = datos.values

# Etiquetas (target)
y = datos_iris['target'].values

# Definición de parámetros
tamano_entrada = X.shape[1] #número de columnas de X
tamano_salida = 3  # 3 clases en el conjunto de datos Iris
tamano_oculta1 = 6 #capa1
tamano_oculta2 = 5 #capa2
tasa_aprendizaje = 0.1
epocas = 1000

# Inicialización de pesos
np.random.seed(37)
pesos_entrada_oculta1 = np.random.rand(tamano_entrada, tamano_oculta1)
pesos_oculta1_oculta2 = np.random.rand(tamano_oculta1, tamano_oculta2)
pesos_oculta2_salida = np.random.rand(tamano_oculta2, tamano_salida)

#mostrar todas las filas y columnas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Entrenamiento de la red neuronal
for epoca in range(epocas):
    # Forward pass (producto punto y activación)
    entrada_oculta1 = np.dot(X, pesos_entrada_oculta1)
    salida_oculta1 = sigmoid(entrada_oculta1)

    entrada_oculta2 = np.dot(salida_oculta1, pesos_oculta1_oculta2)
    salida_oculta2 = sigmoid(entrada_oculta2)

    entrada_salida = np.dot(salida_oculta2, pesos_oculta2_salida)
    salida_final = sigmoid(entrada_salida)

    # Calcular el error
    error = y.reshape(-1, 1) - salida_final
    #print('Error:',error)

    # Backpropagation (errores y actualización de pesos)
    delta_salida = error * sigmoid_derivada(salida_final)
    error_oculta2 = delta_salida.dot(pesos_oculta2_salida.T)
    delta_oculta2 = error_oculta2 * sigmoid_derivada(salida_oculta2)
    error_oculta1 = delta_oculta2.dot(pesos_oculta1_oculta2.T)
    delta_oculta1 = error_oculta1 * sigmoid_derivada(salida_oculta1)

    # Actualizar pesos
    pesos_oculta2_salida += salida_oculta2.T.dot(delta_salida) * tasa_aprendizaje
    pesos_oculta1_oculta2 += salida_oculta1.T.dot(delta_oculta2) * tasa_aprendizaje
    pesos_entrada_oculta1 += X.T.dot(delta_oculta1) * tasa_aprendizaje

    # Imprimir la pérdida
    if (epoca + 1) % 10 == 0:
        perdida = np.mean(np.abs(error))
        print(f'Época {epoca + 1}, Pérdida: {perdida}')

# Predicciones
predicciones = np.argmax(salida_final, axis=1)

# Imprimir las predicciones
resultados = pd.DataFrame({
    'Etiqueta Real': y,
    'Predicción': predicciones
})

print(resultados)
