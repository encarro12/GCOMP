import numpy as np
import random
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

# Función de separación que pretendemos simular con el perceptrón simple
def f(x, y):
  return 3*x + 2*y > 2

# Diseñamos el perceptrón simple
class Perceptron_simple:
  # Factor de aprendizaje
  e = 0.5

  # Función de activación
  def sigmoid_function(self, x: float) -> float:
    """
    Logistic or sigmoid function, used as the activation function
    """
    return 1./(1 + np.exp(-x))

  def salida_sensor(self, E: np.array) -> float:
    """
    Función de predicción para un punto E, que consiste en aplicar
    la función sigmoide a c(E) = <self.weights, E>.
    """
    # Función de conexión, c(E)
    c = np.dot(self.weights, E)
    # Aplicamos la función sigmoide a c(E)
    output_sensor = self.sigmoid_function(c)
    # Devolvemos la salida del sensor
    return output_sensor


  # Función de aprendizaje
  def fit(self, X_entrenamiento: np.array, S_entrenamiento: np.array, n_it: int = 1):
    """
    Función de aprendizaje basada en la Regla Delta generalizada, donde
    Args:
        - E_entrenamiento: El conjunto de datos de entrenamiento.
        - S_entrenamiento: Salidas reales del conjunto de entrenamiento.
        - n_it: Número de iteraciones con el conjunto de entrenamiento.
    """
    # Asignamos valores aleatorios y pequeños, tanto positivos como negativos
    # a los pesos sinápticos. Normalmente estos pesos están comprendidos entre
    # [-1, 1].
    self.weights = np.random.uniform(-1, 1, X_entrenamiento.shape[1])
    self.b_weights = self.weights.copy()
    # Para cada uno de los patrones de entrada, realizamos "n_it" iteraciones.
    for it in range(n_it):
      for E, s in zip(X_entrenamiento, S_entrenamiento):
        # Feedforward: Conexión entre la entrada
        s_predecido = self.salida_sensor(E)
        # Backpropagation: Actualizamos los pesos
        error =  s - s_predecido
        delta_w = error * s_predecido * (1 - s_predecido) * E
        self.weights += (1 - self.e)*delta_w
  
  # Función predicción
  def prediccion(self, X_prueba: np.array) -> np.array:
    """
    Una vez entrenado el perceptrón simple, devuelve salidas posibles para
    un conjunto de prueba que se pasa por argumento
    """
    return np.array([self.salida_sensor(E) for E in X_prueba]) > 0.5
  
  def beginning_weights(self) -> np.array:
    return self.b_weights

# Definimos el conjunto de prueba que va a ser el mismo siempre
X_prueba = np.array([[1,1], [1,0], [0,1]])
S_prueba = np.array([f(E[0], E[1]) for E in X_prueba])

# Creamos los dataframes donde se recoge la información
columnas = ['N', 'Precision PS', 'Precision Keras', 'Acierta E1', 'Acierta E2', 'Acierta E3']
df = pd.DataFrame(columns=columnas)

# Definimos una función que realiza la comparación entre los dos modelos, fijando la región R donde
# se coge el conjunto de entrenamiento y N el número de ejemplos
def comparacion(N):
    """
    Función que compara el perceptrón simple diseñado en la clase de arriba y el diseñado por 
    la librería keras.

    Args:
        R: Región donde se encuentran el conjunto de entrenamiento.
        N: Cardinal del conjunto de entrenamiento.
        df: Dataframe donde se recoge la información de las pruebas.
    """
    # Obtenemos el conjunto de entrenamiento junto a sus etiquetas
    X_entrenamiento = np.array([[random.randint(-R, R), random.randint(-R, R)] for n in range(N)])
    S_entrenamiento = np.array([f(E[0], E[1]) for E in X_entrenamiento])
    
    # Entrenamos y evaluamos el perceptrón simple
    perceptron = Perceptron_simple()
    perceptron.fit(X_entrenamiento, S_entrenamiento)
    S_predecida = perceptron.prediccion(X_prueba)
    
    # Entrenamos y evaluamos el perceptrón simple de Keras
    weights = [np.reshape(perceptron.beginning_weights(), (2,1))]
    modelo = keras.models.Sequential()
    modelo.add(keras.layers.Dense(1, input_dim=2, use_bias=False, activation='sigmoid', weights=weights)) # Una neurona con sigmoide
    modelo.compile(optimizer=keras.optimizers.SGD(learning_rate=0.5), loss='mean_squared_error') # Regla Delta generalizada
    modelo.fit(x=X_entrenamiento, y=S_entrenamiento, epochs=1, shuffle=True, verbose=0) # Una iteración
    S_predecida_keras = np.reshape(modelo.predict(x=X_prueba, verbose=0), -1) > 0.5
    
    # Comparamos los resultados con la salida real
    return {
        'N': N,
        'Precision PS': len(S_predecida[S_predecida == S_prueba]) * 100 / len(S_predecida),
        'Precision Keras': np.all(S_predecida == S_predecida), # Comparacion con Keras
        'Acierta E1': S_predecida[0] == S_prueba[0],
        'Acierta E2': S_predecida[1] == S_prueba[1],
        'Acierta E3': S_predecida[2] == S_prueba[2]
    }



# Obtención de los datos variando la N
values_N = np.arange(1, 10000 + 100, 100)
R = 700
for N in values_N:
    df = df.append(comparacion(N), ignore_index=True)
    
# Veces que acierta E1, E2, E3
fig = plt.figure(figsize = (6, 6))
Ei = np.array(['E1', 'E2', 'E3'])
PEi = np.array([len(df[df['Acierta E1']]), len(df[df['Acierta E2']]), len(df[df['Acierta E3']])]) / len(df)
plt.bar(Ei, PEi, color=['r','b', 'g'])
plt.ylabel('Porcentaje de acierto')
plt.ylim(0, 1)
plt.savefig("Diagrama de barras N.png")

# Porcentaje de acierto del perceptrón
fig = plt.figure(figsize = (6, 6))
PN = np.array([df['Precision PS']])
plt.scatter(values_N, PN, color='r')
plt.xlabel('N')
plt.ylabel('Porcentaje de acierto')
plt.savefig("Precision PS N.png")

# Comprobar que predicen lo mismo
print("Porcentaje de acierto de ambos:", len(df[df['Precision Keras']])*100 / len(df))