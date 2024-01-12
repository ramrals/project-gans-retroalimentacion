# generador_datos.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generar_generador_datos(ruta_datos, tamano_lote=2):
    # Configurar el generador de datos para la CNN
    datagen = ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 validation_split=0.2)  # Puedes ajustar estos parámetros según tus necesidades

    # Configurar el generador de flujo de datos desde el directorio
    generator = datagen.flow_from_directory(
        ruta_datos,
        target_size=(224, 224),
        batch_size=tamano_lote,
        class_mode='categorical',  # Ajusta según el tipo de problema (categorical para clasificación)
        subset='training',
        shuffle=True,  # Puedes ajustar según tus necesidades
        seed=42  # Puedes ajustar según tus necesidades
    )

    return generator
