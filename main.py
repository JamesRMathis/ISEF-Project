from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import pad_sequences
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import random
import numpy as np
import pickle

from matplotlib import pyplot as plt

def parseData():
    file = open('functions.txt', 'r').read()

    functions = file.split('\n<sep>\n')
    results = []
    for i in range(len(functions)):
        for line in functions[i].split('\n'):
            if line.startswith('#'):
                results.append(int(line[1:].strip()))
                functions[i] = functions[i].replace(line, '')

    print(len(functions))
    return functions, results

def shuffleData(functions, results):
    data = list(zip(functions, results))
    random.shuffle(data)
    funcs, res = zip(*data)
    return funcs, res

def tokenizeData(functions):
    # Tokenize the data
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(functions)
    sequences = tokenizer.texts_to_sequences(functions)

    # Pad the sequences to a fixed length
    max_length = 500
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    return sequences, padded_sequences, max_length, tokenizer

def splitData(padded_sequences, results):
    # Split the data into training and testing sets

    split_index = int(len(padded_sequences) * 2 / 3)
    X_train = padded_sequences[:split_index]
    y_train = np.array(results[:split_index])
    X_test = padded_sequences[split_index:]
    y_test = np.array(results[split_index:])

    return X_train, y_train, X_test, y_test

def createModel(optimizer='adam', output_dim=5, activation='sigmoid'):
    # Define the model architecture
    model = Sequential()
    model.add(Embedding(input_dim=1000, output_dim=output_dim, input_length=500, input_length=500))
    model.add(Flatten())
    model.add(Dense(1, activation=activation))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Recall', 'Precision'])

    return model

def trainModel(model, X_train, y_train, X_test, y_test, training=1, optimizer='adam', epochs=1000, batch_size=50):

    # Compile and train the model
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
    # return loss, accuracy, recall, precision

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    return y_pred

# def createConfusionMatrix(y_test, y_pred):
#     cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
#     disp = ConfusionMatrixDisplay(cm, display_labels=['doesnt halt', 'halts'])
#     disp.plot(cmap=plt.cm.Reds)
#     plt.show()

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

    # lossList, accList, recList, precList = [], [], [], []
    # for i in range(100):
    #   model = createModel(max_length, 20)
    #   loss, acc, rec, prec = trainModel(model, X_train, y_train, X_test, y_test, optimizer='rmsprop', epochs=200, batch_size=64)
    #   with open('output.csv', 'a') as f:
    #     f.write(f'{loss}, {acc}, {rec}, {prec}\n')
    #   lossList.append(loss)
    #   accList.append(acc)
    #   recList.append(rec)
    #   precList.append(prec)
    # avgLoss = sum(lossList) / len(lossList)
    # avgAcc = sum(accList) / len(accList)
    # avgRec = sum(recList) / len(recList)
    # avgPrec = sum(precList) / len(precList)
    # print(f'avgLoss: {avgLoss}, avgAcc: {avgAcc}, avgRec: {avgRec}, avgPrec: {avgPrec}')
    model = createModel(max_length, 20)
    y_pred = trainModel(model, X_train, y_train, X_test, y_test, optimizer='rmsprop', epochs=200, batch_size=64)
    createConfusionMatrix(y_test, y_pred)
    # createConfusionMatrix(y_train, y_pred)
    # seePredictions(X_test, y_test, y_pred, tokenizer)

if __name__ == '__main__':
    main()
    # optimize()

    # functions, results = parseData()
    # functions, results = shuffleData(functions, results)
    # sequences, padded_sequences, max_length, tokenizer = tokenizeData(functions)
    # X, Y = padded_sequences, np.array(results)
    # print(X, Y)

    # model = KerasClassifier(createModel, output_dim=5, verbose=0)
    # print('model created')
    # param_grid = {
    #     'optimizer': ['adam', 'sgd', 'rmsprop'],
    #     'epochs': [val for val in range(100, 501, 100)],
    #     # 'batch_size': [val for val in range(20, 201, 20)],
    #     'batch_size': [2**i for i in range(5, 8)],
    #     'output_dim': [val for val in range(20, 101, 20)]
    # }
    # print(param_grid)
    # # smaller grid for debug
    # param_grid = {
    #     'optimizer': ['adam', 'sgd', 'rmsprop'],
    #     'epochs': [val for val in range(100, 201, 100)],
    #     'batch_size': [val for val in range(5, 11, 5)],
    #     'output_dim': [val for val in range(5, 11, 5)]
    # }
    # grid = GridSearchCV(model, param_grid, cv=3, verbose=1, scoring='accuracy', n_jobs=-1)
    # grid.fit(X, Y)

    print(grid.best_params_)
    print(grid.best_score_)
    print(grid.best_estimator_)