import nltk
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

# Use pathlib for handling file paths
from pathlib import Path

# Load intents from a JSON file
intents_file = Path("intents.json")
with intents_file.open() as file:
    data = json.load(file)

# Process and preprocess the intents data
words = []
labels = []
docs_1 = []
docs_2 = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        words_1 = nltk.word_tokenize(pattern)
        words.extend(words_1)
        docs_1.append(words_1)
        docs_2.append(intent['tag'])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Stemming and preparing words and labels
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_1):
    bag = []

    words_1 = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in words_1:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_2[x])] = 1

    training.append(bag)
    output.append(output_row)

# Convert training and output into numpy arrays
training = np.array(training)
output = np.array(output)

# Save processed data using pickle
data_pickle_file = Path("data.pickle")
with data_pickle_file.open("wb") as f:
    pickle.dump((words, labels, training, output), f)

# Reset TensorFlow graph
tf.compat.v1.reset_default_graph()

# Define the neural network model using tflearn
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

# Load or train the model
model_file = Path("model.tflearn")
if model_file.exists():
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# Function to create a bag of words from a given sentence
def words_bag(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

# Function to start the chat with the bot
def chat():
    print("Start talking with the bot (to stop type 'quit')!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Get predictions from the model based on user input
        results = model.predict([words_bag(inp, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        # Retrieve appropriate responses from intents based on predicted tag
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        # Display a random response from the identified tag
        print(random.choice(responses))

# Start the chatbot
chat()
