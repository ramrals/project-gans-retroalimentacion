# generador_datos_gan.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generar_generador_datos_gan(ruta_datos_gan, tamano_lote=2):
    # Configurar el generador de datos para la GAN
    datagen = ImageDataGenerator(rescale=1./255)

    # Configurar el generador de flujo de datos desde el directorio
    generator = datagen.flow_from_directory(
        ruta_datos_gan,
        target_size=(224, 224),
        batch_size=tamano_lote,
        class_mode=None,  # No se utiliza class_mode para la GAN
        shuffle=True
        #seed=None
    )

    return generator
