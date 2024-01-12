# entrenamiento_conjunto.py
import numpy as np
from tensorflow.keras.layers import concatenate
from datos.generador_datos_cnn import generar_generador_datos

from tensorflow.keras.layers import Input
from datos.generador_datos_gan import generar_generador_datos_gan
from modelos.gan_modelo import crear_generador_bueno, crear_generador_malo, crear_discriminador
from modelos.cnn_modelo import crear_modelo
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def entrenar_gan_cnn_conjunto(generador_datos_gan, input_dim, num_epochs=10):
    # Crear y compilar modelos
    generador_bueno = crear_generador_bueno(input_dim)
    generador_malo = crear_generador_malo(input_dim)
    discriminador = crear_discriminador(input_dim)

    entrada_generador = Input(shape=(input_dim,))
    salida_generador_bueno = generador_bueno(entrada_generador)
    salida_generador_malo = generador_malo(entrada_generador)
    salida_concatenada = concatenate([salida_generador_bueno, salida_generador_malo])
    modelo_completo = Model(inputs=entrada_generador, outputs=salida_concatenada)
    
    discriminador.trainable = False
    modelo_completo.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss=['binary_crossentropy', 'binary_crossentropy'])

    tamano_lote = generador_datos_gan.batch_size
    pasos_por_epoca = generador_datos_gan.n // tamano_lote

    for epoch in range(num_epochs):
        for paso in range(pasos_por_epoca):
            ruido = np.random.normal(0, 1, (tamano_lote, input_dim))

            datos_generados_buenos = generador_bueno.predict(ruido)
            datos_generados_malos = generador_malo.predict(ruido)

            etiquetas_reales = np.ones((tamano_lote, 1))
            etiquetas_falsas = np.zeros((tamano_lote, 1))

            # Entrenar el discriminador
            perdida_real = discriminador.train_on_batch(datos_generados_buenos, etiquetas_reales)
            perdida_fake = discriminador.train_on_batch(datos_generados_malos, etiquetas_falsas)

            # Imprimir progreso, guardar modelos, etc.
            print(f'Epoch {epoch + 1}/{num_epochs}, Pérdida Real: {perdida_real[0]}, Pérdida Fake: {perdida_fake[0]}')


            # Entrenar el modelo completo (generadores)
            modelo_completo.train_on_batch(ruido, [etiquetas_reales, etiquetas_falsas])


        # Guardar modelos después de cierto número de épocas si es necesario
        if (epoch + 1) % 1 == 0:
            generador_bueno.save('generador_bueno_modelo.keras')
            generador_malo.save('generador_malo_modelo.keras')
            discriminador.save('discriminador_modelo.keras')


# Puedes ejecutar la función para comenzar el entrenamiento
ruta_datos_gan = 'dataset_gan/'
generador_datos_gan = generar_generador_datos_gan(ruta_datos_gan)
entrenar_gan_cnn_conjunto(generador_datos_gan, input_dim=2, num_epochs=2)
