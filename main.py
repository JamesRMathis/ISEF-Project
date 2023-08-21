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

# for i in range(len(functions) - 2):
#     X_train.append(functions[i][0])
#     y_train.append(functions[i][1])

# for i in range(len(functions) - 2, len(functions)):
#     X_test.append(functions[i][0])
#     y_test.append(functions[i][1])

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random

# Combine the comments and results into a single list
data = list(zip(functions, results))

# Shuffle the data
random.shuffle(data)
# print(data)

functions, results = zip(*data)








# Tokenize the data
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(functions)
sequences = tokenizer.texts_to_sequences(functions)

vocab_size = tokenizer.num_words
# print(vocab_size)

# Pad the sequences to a fixed length
max_length = 50
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split the data into training and testing sets
split_index = int(len(padded_sequences) * 2 / 3)
X_train = padded_sequences[:split_index]
y_train = np.array(results[:split_index])
X_test = padded_sequences[split_index:]
y_test = np.array(results[split_index:])

# print(X_test, y_test)


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
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=5, verbose=0)

# Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print('Test accuracy:', accuracy)

p = model.predict(X_test)
# p = (p > 0.5).astype(int)
# print(p)

# Create a reverse word index
reverse_word_index = dict([(value, key) for (key, value) in tokenizer.word_index.items()])

# Convert a sequence of integers back into text
for ind, sequence in enumerate(X_test):
    text = ' '.join([reverse_word_index.get(i, '?') for i in sequence])
    print(f'real: {y_test[ind]}, predicted: {p[ind]}, text: {text}')