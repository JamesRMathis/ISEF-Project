file = open(r'C:\Users\James\Documents\Code\School\ISEF Project\functions.txt', 'r').read()

functions = file.split('\n\n')
results = []
for i in range(len(functions)):
    for line in functions[i].split('\n'):
        if line.startswith('#'):
            results.append(int(line[1:].strip()))
            functions[i] = functions[i].replace(line, '')
# print(functions, results, sep='\n\n')

# functions = list(zip(functions, results))

import random

# Split the data into training and testing sets
random.shuffle(functions)
X_train = []
y_train = []
X_test = []
y_test = []

# for i in range(len(functions) - 2):
#     X_train.append(functions[i][0])
#     y_train.append(functions[i][1])

# for i in range(len(functions) - 2, len(functions)):
#     X_test.append(functions[i][0])
#     y_test.append(functions[i][1])

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Combine the comments and results into a single list
data = [entry[0] + ' ' + str(entry[1]) for entry in zip(functions, results)]

# Tokenize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)

vocab_size = tokenizer.num_words
print(vocab_size)

# Pad the sequences to a fixed length
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split the data into training and testing sets
X_train = padded_sequences[:-2]
y_train = np.array(results[:-2])
X_test = padded_sequences[-2:]
y_test = np.array(results[-2:])

# print(X_train, y_train)


from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

# Define the model architecture
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=10, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=5)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)