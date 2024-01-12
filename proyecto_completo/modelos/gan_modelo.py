# gan_modelo.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam

def crear_generador_bueno(input_dim):
    modelo = Sequential()
    modelo.add(Dense(256, input_dim=input_dim))
    modelo.add(LeakyReLU(alpha=0.2))
    modelo.add(BatchNormalization(momentum=0.8))
    modelo.add(Dense(512))
    modelo.add(LeakyReLU(alpha=0.2))
    modelo.add(BatchNormalization(momentum=0.8))
    modelo.add(Dense(1024))
    modelo.add(LeakyReLU(alpha=0.2))
    modelo.add(BatchNormalization(momentum=0.8))
    modelo.add(Dense(3, activation='tanh'))  # Ajusta el número de neuronas según el tamaño de salida y la tarea
    return modelo

def crear_generador_malo(input_dim):
    modelo = Sequential()
    modelo.add(Dense(256, input_dim=input_dim))
    modelo.add(LeakyReLU(alpha=0.2))
    modelo.add(BatchNormalization(momentum=0.8))
    modelo.add(Dense(512))
    modelo.add(LeakyReLU(alpha=0.2))
    modelo.add(BatchNormalization(momentum=0.8))
    modelo.add(Dense(1023))
    modelo.add(LeakyReLU(alpha=0.2))
    modelo.add(BatchNormalization(momentum=0.8))
    modelo.add(Dense(3, activation='tanh'))  # Ajusta el número de neuronas según el tamaño de salida y la tarea
    
    # Similar al generador bueno, puedes experimentar con la arquitectura
    # ...
    return modelo

def crear_discriminador(input_dim):
    modelo = Sequential()
    modelo.add(Dense(2, input_dim=input_dim))
    modelo.add(LeakyReLU(alpha=0.2))
    modelo.add(Dense(512))
    modelo.add(LeakyReLU(alpha=0.2))
    modelo.add(Dense(256))
    modelo.add(LeakyReLU(alpha=0.2))
    modelo.add(Dense(1, activation='sigmoid'))
    modelo.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return modelo
