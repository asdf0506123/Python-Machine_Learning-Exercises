import tensorflow as tf
import numpy as np

centimetros = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
pulgadas = np.array([0.393701, 0.787402,1.1811 ,1.5748 ,1.9685 ,2.3622 ,2.75591 ], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer= tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print('Comenzando entrenamiento...')
historial = modelo.fit(centimetros, pulgadas, epochs=50, verbose=False)
print('Modelo entrenado!')

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])

print('Realizando una prediccion.')
inputUsuario = float(input("Introduce una cantidad en centimetros: "))
input_data = np.array([[inputUsuario]])
resultado = modelo.predict(input_data)
print(f'El resultado es: {resultado[0][0]:.2f} pulgadas')

print("Variables internas del modelo")
print(capa.get_weights())