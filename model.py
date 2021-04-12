# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:31:26 2021
@author: freddyalc
Detalle: Predicción de Spam utilizando recurrent neural network
Dataset: https://www.kaggle.com/team-ai/spam-text-message-classification
Fuentes: 
    - https://laptrinhx.com/nlp-detecting-spam-messages-with-tensorflow-2612233361/
    - https://github.com/MGCodesandStats/tensorflow-nlp
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split

################## Pre-procesamiento ##################
# Cargar dataset
dataframe = pd.read_csv( 'dataset/spam.csv' , encoding='iso8859' , usecols=[ 'v1' , 'v2' ] )
# v1 = Ground Truth (Y)
# v2 = Característica 'texto' (X)

# labels antes de codificar en números: 
print(np.unique(dataframe.v1))

# Crear lista para almacenar los labels (Y):

labels = list()
for line in dataframe.v1:
	labels.append( 0 if line == 'ham' else 1 ) # 0 = ham/no spam - 1 = Spam

# labels después de codificar: 
print(np.unique(labels)) #0 = ham/no spam 1 = spam

# Crear lista para almacenar los textos (X):
texts = list()
for line in dataframe.v2:
	texts.append( line )

# Lista para guardar la longitud de cada texto 
lengths = list()

for text in texts:
	lengths.append( len( text.split() ) ) # Guardar longitudes

# Obtener la longitud maxima de todos los textos para el tokenizer
maxlen = max( lengths ) #Este maxlenght se utilizará en la aplicación android para secuenciar los comentarios.

# Convertir las listas de los datos Y y X en arrays
labels = np.array( labels )
texts = np.array( texts )


# Tokenize the dataset, including padding and OOV
#Set super parameters
vocab_size = 600 #tambien puede ser 1000
embedding_dim = 16
#max_length = maxlen #Con un valor de 60 aqui se obtiene una precision de 98% para predecir spam. Con maxLen=171 se obtiene el mismo resultado con un random_state=11 en el train_Teste_split
max_length = maxlen
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Crear objeto tokenizer de keras para tokenizar los mensajes

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
# Entrenar el tokenizador con los textos
tokenizer.fit_on_texts(texts)

word_index = tokenizer.word_index

# #Crear una lista de sencuencias de numeros enteros
sequences = tokenizer.texts_to_sequences(texts)

# Aplicar el método de relleno 'padded' para que que todas las secuencias en la lista tengan la misma longitud
padded_messages = pad_sequences(sequences,
                                maxlen=max_length,
                                padding=padding_type, 
                                truncating=trunc_type)

# Review a las secuencias:
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded_messages[1]))
print(padded_messages[1])

X = padded_messages
Y = labels

# El Tokenizer mantiene un objeto 'dict' de Python que tiene pares de palabras
# y sus índices. Convertimos este diccionario a un archivo JSON usando:

with open( 'android/word_dict.json' , 'w' ) as file:
	json.dump( tokenizer.word_index , file )
# Este archivo word_dict.json se utilizara en la aplicación android.


# Dividir los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split( X , Y , test_size=0.2, random_state=11)

#Crear modelo

# Note the embedding layer is first, 
# and the output is only 1 node as it is either 0 or 1 (negative or positive)
model = tf.keras.Sequential([
    #Capa de incrustación o embebida de Keras: https://unipython.com/como-aprender-y-cargar-incrustaciones-de-palabras-word-embbeddings-en-keras/
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    #La capa Embedding devuelve un vector 2d se debe aplanar a 1d para la capa densa.
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Compilar modelo
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# Definir callback para guardar solo el mejor modelo con menor perdida
from tensorflow.keras.callbacks import ModelCheckpoint
carpeta_logs = "logs/modelo.h5"
checkpoint = ModelCheckpoint(carpeta_logs, monitor="val_loss", save_best_only=True, verbose=1)
callbacks = [checkpoint]


num_epochs = 50
history=model.fit(x_train,
                  y_train,
                  verbose=1,
                  epochs=num_epochs,
                  callbacks=callbacks,
                  validation_data=(x_test, y_test))

# Cargar el modelo guardado con la menor perdida:

model = tf.keras.models.load_model('logs/modelo.h5')

# Obtener perdida y precisión en los datos de prueba:
loss , accuracy = model.evaluate( x_test , y_test )
print( "Perdida de prueba: {}".format( loss ) , "Accuracy: {} %".format( accuracy * 100 ) )

# Resultados de perdida y precision con max_lenght = 60 (Reducido al original)
# Perdida de prueba: 0.04714813761312865 Accuracy: 99.01345372200012 %

# Resultados de perdida y precision con maxlen = 171 (Obtenido de los textos original)
# Perdida de prueba: 0.03035859611937818 Accuracy: 99.19282793998718 %

# Plotear la precisión y la pérdida de entrenamiento y validación en cada época

from matplotlib import pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Hacer el clasification report:

from sklearn.metrics import classification_report
# Evaluar la red
# Nota: 0 = No Spam, 1 = Spam
y_preds = model.predict(x_test, batch_size=64)
y_preds = (y_preds > 0.5)

print(classification_report(y_test, y_preds))
# con max_lenght=60

#  0       0.99% precision
#  1       0.98% precision

# con max_lenght=171

#  0       0.99% precision
#  1       0.98% precision

# Hacer Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_preds)

# sns.heatmap(cm, annot=True) Asi se muestra la cm con not. cientifica.
# https://stackoverflow.com/questions/29647749/seaborn-showing-scientific-notation-in-heatmap-for-3-digit-numbers
sns.heatmap(cm, annot=True, fmt='d') #Mostrar enteros

# Usar el modelo para predecir si es spam:
text_messages = ['England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/Ì¼1.20 POBOXox36504W45WQ 16+',
                'Congrats on your new iPhone! Click here to claim your prize...', 
                'Really like that new photo of you',
                'Did you hear the news today? Terrible what has happened...',
                'Attend this free COVID webinar today: Book your session now...',
                'Are you coming to the party tonight?',
                'Your parcel has gone missing',
                'Do not forget to bring friends!',
                'You have won a million dollars! Fill out your bank details here...',
                'Looking forward to seeing you again']

# Crear las secuencias
padding_type='post'
sample_sequences = tokenizer.texts_to_sequences(text_messages)
fakes_padded = pad_sequences(sample_sequences, padding=padding_type, maxlen=max_length)           

classes = model.predict(fakes_padded)

# The closer the class is to 1, the more likely that the message is spam
for x in range(len(text_messages)):
  print(text_messages[x])
  print(classes[x])
  print('\n')

# Resultados con max_length=60 :
"""
England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/Ì¼1.20 POBOXox36504W45WQ 16+
[0.9688195]

Congrats on your new iPhone! Click here to claim your prize...
[0.98583084]

Really like that new photo of you
[0.00128229]

Did you hear the news today? Terrible what has happened...
[6.2555237e-06]

Attend this free COVID webinar today: Book your session now...
[0.07541651]

Are you coming to the party tonight?
[2.1808735e-05]

Your parcel has gone missing
[0.00157868]

Do not forget to bring friends!
[1.4570647e-05]

You have won a million dollars! Fill out your bank details here...
[0.71377903]

Looking forward to seeing you again
[0.00334883]
"""

# Resultados con maxlen=171
"""
England v Macedonia - dont miss the goals/team news.
Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/Ì¼1.20 POBOXox36504W45WQ 16+
[0.84690726]

Congrats on your new iPhone! Click here to claim your prize...
[0.9396416]

Really like that new photo of you
[0.00131759]

Did you hear the news today? Terrible what has happened...
[0.00562917]

Attend this free COVID webinar today: Book your session now...
[0.6351438]

Are you coming to the party tonight?
[0.00028302]

Your parcel has gone missing
[0.00778051]

Do not forget to bring friends!
[0.00016722]

You have won a million dollars! Fill out your bank details here...
[0.54084766]

Looking forward to seeing you again
[0.01155215]
"""
# Cargar el model entrenado y guardar el modelo en formato tflite:
keras_model = tf.keras.models.load_model('logs/modelo.h5', compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tfmodel = converter.convert()
file = open('android/spamv3.tflite','wb') 
file.write(tfmodel)






