file = open(r'C:\Users\James\Documents\Code\School\ISEF Project\functions.txt', 'r').read()

functions = file.split('\n<sep>\n')
results = []
for i in range(len(functions)):
    for line in functions[i].split('\n'):
        if line.startswith('#'):
            results.append(int(line[1:].strip()))
            functions[i] = functions[i].replace(line, '')

# Shuffle the data
import random

data = list(zip(functions, results))
random.shuffle(data)
functions, results = zip(*data)

# Tokenize the data
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(functions)
sequences = tokenizer.texts_to_sequences(functions)

# Pad the sequences to a fixed length
from keras.preprocessing.sequence import pad_sequences

max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split the data into training and testing sets
import numpy as np

split_index = int(len(padded_sequences) * 2 / 3)
X_train = padded_sequences[:split_index]
y_train = np.array(results[:split_index])
X_test = padded_sequences[split_index:]
y_test = np.array(results[split_index:])

# Define the model architecture
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=10, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compile and train the model
import pickle
training = 0
if training:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=10, verbose=0)
    model.save('model.keras')

    pickle.dump(X_test, open('X_test.pkl', 'wb'))
    pickle.dump(y_test, open('y_test.pkl', 'wb'))
else:
    from keras.models import load_model

    model = load_model('model.keras')
    X_test = pickle.load(open('X_test.pkl', 'rb'))
    y_test = pickle.load(open('y_test.pkl', 'rb'))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
# print('Test accuracy:', accuracy)

p = model.predict(X_test)
# p = (p > 0.5).astype(int)
# print(p)

# Create a reverse word index
reverse_word_index = dict([(value, key) for (key, value) in tokenizer.word_index.items()])

# Convert a sequence of integers back into text
for ind, sequence in enumerate(X_test):
    text = ' '.join([reverse_word_index.get(i, '\0') for i in sequence])
    print(f'real: {y_test[ind]}, predicted: {p[ind]}, text: {text}')