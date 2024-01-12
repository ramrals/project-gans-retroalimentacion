# entrenamiento.py
def entrenar_cnn(modelo, generador_datos, num_epochs=2):
    pasos_por_epoca = generador_datos.n // generador_datos.batch_size

    modelo.fit(
        generador_datos,
        steps_per_epoch=pasos_por_epoca,
        epochs=num_epochs,
    )

# Ejemplo de uso:
# entrenar_cnn(modelo_cnn, generador_datos_cnn, num_epochs=10)
