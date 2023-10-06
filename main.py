def parseData():
    file = open(r'C:\Users\James\Documents\Code\School\ISEF Project\functions.txt', 'r').read()

    functions = file.split('\n<sep>\n')
    results = []
    for i in range(len(functions)):
        for line in functions[i].split('\n'):
            if line.startswith('#'):
                results.append(int(line[1:].strip()))
                functions[i] = functions[i].replace(line, '')
    
    return functions, results

def shuffleData(functions, results):
    # Shuffle the data
    import random

    data = list(zip(functions, results))
    random.shuffle(data)
    funcs, res = zip(*data)
    return funcs, res

def tokenizeData(functions):
    
    # Tokenize the data
    from keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(functions)
    sequences = tokenizer.texts_to_sequences(functions)

    # Pad the sequences to a fixed length
    from keras.preprocessing.sequence import pad_sequences

    max_length = 100
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    return sequences, padded_sequences, max_length, tokenizer

def splitData(padded_sequences, results):
    # Split the data into training and testing sets
    import numpy as np

    split_index = int(len(padded_sequences) * 2 / 3)
    X_train = padded_sequences[:split_index]
    y_train = np.array(results[:split_index])
    X_test = padded_sequences[split_index:]
    y_test = np.array(results[split_index:])

    return X_train, y_train, X_test, y_test

def createModel(max_length):


    # Define the model architecture
    from keras.models import Sequential
    from keras.layers import Embedding, Flatten, Dense
    from keras.metrics import Recall, Precision

    model = Sequential()
    model.add(Embedding(input_dim=1000, output_dim=10, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

def trainModel(model, X_train, y_train, X_test, y_test, training=1, optimizer='adam', epochs=1000, batch_size=50):
    
    # Compile and train the model
    import pickle
    training = 1
    if training:
        # recall = Recall()
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', 'Recall', 'Precision'])
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0)
        model.save('model.keras')

        pickle.dump(X_test, open('X_test.pkl', 'wb'))
        pickle.dump(y_test, open('y_test.pkl', 'wb'))
    else:
        from keras.models import load_model

        model = load_model('model.keras')
        X_test = pickle.load(open('X_test.pkl', 'rb'))
        y_test = pickle.load(open('y_test.pkl', 'rb'))

    # Evaluate the model
    loss, accuracy, recall, precision = model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    return y_pred

def createConfusionMatrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from matplotlib import pyplot as plt
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=['doesnt halt', 'halts'])
    disp.plot(cmap=plt.cm.Reds)
    plt.show()

def seePredictions(X_test, y_test, y_pred,  tokenizer):

    # Create a reverse word index
    reverse_word_index = dict([(value, key) for (key, value) in tokenizer.word_index.items()])

    # Convert a sequence of integers back into text
    for ind, sequence in enumerate(X_test):
        text = ' '.join([reverse_word_index.get(i, '\0') for i in sequence])
        print(f'real: {y_test[ind]}, predicted: {y_pred[ind]}, text: {text}')

def main():
    functions, results = parseData()
    functions, results = shuffleData(functions, results)
    sequences, padded_sequences, max_length, tokenizer = tokenizeData(functions)
    X_train, y_train, X_test, y_test = splitData(padded_sequences, results)
    model = createModel(max_length)
    y_pred = trainModel(model, X_train, y_train, X_test, y_test, optimizer='sgd', epochs=1000, batch_size=10)
    createConfusionMatrix(y_test, y_pred)
    seePredictions(X_test, y_test, y_pred, tokenizer)

def optimize():
    functions, results = parseData()
    functions, results = shuffleData(functions, results)
    sequences, padded_sequences, max_length, tokenizer = tokenizeData(functions)
    X_train, y_train, X_test, y_test = splitData(padded_sequences, results)

    # Define the grid search parameters
    optimizers = ['adam', 'sgd', 'rmsprop']
    epochs = [100, 500, 1000, 10000]
    batch_sizes = [10, 20, 50, 100]

    # Perform the grid search
    best_accuracy = 0
    best_params = {}

    model = createModel(max_length)
    for optimizer in optimizers:
        for epoch in range(100, 10000, 100):
            for batch_size in batch_sizes:
                print(f'optimizer: {optimizer}, epochs: {epoch}, batch_size: {batch_size}')
                y_pred = trainModel(model, X_train, y_train, X_test, y_test, optimizer=optimizer, epochs=epoch, batch_size=batch_size)
                loss, accuracy, recall, precision = model.evaluate(X_test, y_test)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'optimizer': optimizer, 'epochs': epoch, 'batch_size': batch_size}

if __name__ == '__main__':
    main()
    # optimize()