# main.py
# Importa las bibliotecas necesarias
import matplotlib.pyplot as plt

import numpy as np

from modelos.gan_modelo import crear_generador_bueno, crear_generador_malo, crear_discriminador
from modelos.cnn_modelo import crear_modelo  # Si es necesario
from datos.generador_datos_gan import generar_generador_datos_gan
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.optimizers import Adam
from modelos.cnn_modelo import crear_modelo as crear_modelo_cnn
from modelos.gan_modelo import crear_generador_bueno, crear_generador_malo, crear_discriminador
from datos.generador_datos_cnn import generar_generador_datos
from datos.generador_datos_gan import generar_generador_datos_gan
from entrenamiento.entrenamiento import entrenar_cnn
from entrenamiento.entrenamiento_conjunto import entrenar_gan_cnn_conjunto

# Configuración para los generadores de datos
ruta_datos_cnn = 'dataset/'  # Ajusta la ruta según la ubicación real de tus datos para la CNN
ruta_datos_gan = 'dataset_gan/'  # Ajusta la ruta según la ubicación real de tus datos para la GAN

def main():
    # Configuración para la CNN
    num_clases_cnn = 1  # Ajusta según el número de clases en tu CNN
    modelo_cnn = crear_modelo_cnn(num_clases_cnn)

    # Configuración para la GAN
    input_dim_gan = 100  # Ajusta según las dimensiones de entrada de tu GAN
    generador_bueno = crear_generador_bueno(input_dim_gan)
    generador_malo = crear_generador_malo(input_dim_gan)
    discriminador = crear_discriminador(input_dim_gan)

    generador_datos_cnn = generar_generador_datos(ruta_datos_cnn)
    generador_datos_gan = generar_generador_datos_gan(ruta_datos_gan)
    
    # Entrenamiento de la CNN
    entrenar_cnn(modelo_cnn, generador_datos_cnn, num_epochs=2)  # Ajusta según tus necesidades

    # Entrenamiento de la GAN
    entrenar_gan_cnn_conjunto(generador_bueno, generador_malo, discriminador, generador_datos_gan, input_dim_gan, num_epochs=2)  # Ajusta según tus necesidades

    # Obtén un lote de datos del generador
    batch = generador_datos_cnn.next()

    # Visualiza algunas imágenes del lote
    for i in range(len(batch[0])):
        plt.imshow(batch[0][i])
        plt.show()

if __name__ == "__main__":
    main()
