import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Download necessary NLTK data
nltk.download('punkt')

import numpy as np
import tensorflow as tf
import random
import json

# Load intents file
with open('intents.json') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

# Tokenize and stem words
for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Stem and lower each word, remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

# Create training data
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Build the neural network with TensorFlow Keras
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(training[0]),)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(output[0]), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training, output, epochs=1000, batch_size=8, verbose=1)
model.save("model.h5")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return np.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict(np.array([bag_of_words(inp, words)]))
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
        
        print(random.choice(responses))

chat()
