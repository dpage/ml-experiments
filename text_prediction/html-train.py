import argparse
import json
import glob
import string

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dropout


# Helper to plot graphs
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()


# Get command line options
parser = argparse.ArgumentParser(description='Create a Tensorflow model based on a directory of HTML')
parser.add_argument("--debug", action='store_true', help="enable debug mode")
parser.add_argument("-d", "--data", required=True, help="the file to save data to")
parser.add_argument("-i", "--input", required=True, help="the input directory containing the HTML files")
parser.add_argument("-m", "--model", required=True, help="the file to save the model to")

args = parser.parse_args()
DEBUG = args.debug

if DEBUG:
    print('Building corpus from {}:'.format(args.input + '/*.html'))

# Create the corpus
corpus = []
for path in glob.iglob(args.input + '/*.html'):
    if DEBUG:
        print('Loading text from {}...'.format(path))

    file = open(path, "r")
    html = file.read()
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()

    # Get the text
    for p in soup.find_all('p'):
        # Extract the <p> tags, and convert to lower case.
        para = p.getText().lower()

        # Remove any line feeds
        para = para.replace('\n', ' ')

        # Break up into one line per sentence
        para = para.replace('. ', '.\n')

        # Replace punctuation with a space
        para = para.translate(str.maketrans(string.punctuation + '“' + '”', ' ' * len(string.punctuation + '“' + '”')))

        # Ensure the data is split into clean lines
        lines = para.split('\n')

        # Add the lines to the corpus... but only if there's more than one word
        for line in lines:
            if line.find(' ') > -1:
                corpus.append(line.strip())


# Check that we loaded something
if len(corpus) == 0:
    print("No corpus could be loaded from {}. Exiting.".format(args.directory + '/*.html'))
    exit(1)

# Tokenize the corpus. This will create the tokenizer, and extract and index
# all the words in the corpus, creating a dictionary of words and their IDs
tokenizer = Tokenizer(num_words=3000)
tokenizer.fit_on_texts(corpus)
total_words = tokenizer.num_words

print("Created a corpus of {} lines with {} words.".format(
    len(corpus),
    len(tokenizer.index_word)))

# Create a list of sequences; that is, a list of lists of word IDs,
# representing each sequential substring, up to the full line (sentence) in
# the corpus. In other words, if the sentence is represented by [1, 2, 3, 4],
# we create:
# [[1, 2],
#  [1, 2, 3],
#  [1, 2, 3, 4]]
sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        sequences.append(n_gram_sequence)

# Pre-pad all the sequences so they're the same length, as the input to the
# model requires that, creating:
# [[0, 0, 1, 2],
#  [0, 1, 2, 3],
#  [1, 2, 3, 4]]
max_sequence_len = max([len(seq) for seq in sequences])
sequences = np.array(
    pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))

# Split sequences between the "input" sequence (the first N elements of each
# sequence), and "output" label or predicted word (the last element of each
# sequence, creating:
# [[0, 0, 1],
#  [0, 1, 2],
#  [1, 2, 3]]
# and:
#  [2, 3, 4]
input_sequences, labels = sequences[:, :-1], sequences[:, -1]

# One-hot encode the labels. This creates a matrix with one row for each
# sequence, and a 1 in the column corresponding to the word ID of the next word
# Given the example sequences above, this would create:
# [[0, 1, 0, 0],
#  [0, 0, 1, 0],
#  [0, 0, 0, 1]]
# Representing word 2 in sequence 1, word 3 in sequence 2, word 4 in sequence 3
# and so on.
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

epochs = 100
if DEBUG:
    epochs = 5

# Build the model
model = Sequential()

# Embedding layer - Turns positive integers (indexes) into dense vectors of fixed size.
# Params:
#   input dimension: words in the training set
#   output dimension: size of the embedding vectors (i.e the number of cells in the next layer)
#   input_length: the size of the sequences (i.e. the longest sentence)
model.add(Embedding(total_words, 256, input_length=max_sequence_len - 1))

# Add a bi-directional LTSM layer, of the appropriate size. LSTMs are Long Term
# Short Memory cells, which have the ability to remember (and forget) values to
# carry forward in the sequence.
# Params:
#   dimension: the size of the layer
#   return_sequences: return the entire sequence, not just the last value, so
#                     it can be fed into another LSTM layer.
model.add(Bidirectional(LSTM(256, return_sequences=True)))

# More of the same, except the LSTM layer doesn't return sequences this time.
model.add(Bidirectional(LSTM(256)))

# Randomly set some input units to zero, to help prevent overfitting. Note that
# we don't do this before any of the LTSM layers because it may cause them to
# forget things that should not be forgotten.
# Params:
#   rate: frequency of the dropouts
model.add(Dropout(0.2))

# Finish up with a dense (fully connected) layer. The softmax activation
# gives us a vector of probabilities for each word in the index.
model.add(Dense(total_words, activation='softmax'))

# Compile the mode.
# Params:
#    loss: the function used to calculate loss
#    optimizer: the optimiser function, for adjusting the learning rate. Adam
#               is generally a good choice and performs well.
#    metrics: the metric(s) to monitor during training.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if DEBUG:
    print(model.summary())

# We're going to save a checkpoint each time our loss metric improves.
checkpoint = ModelCheckpoint('checkpoint.h5',
                             monitor='loss',
                             save_best_only=True,
                             mode='min')

# Train the model.
# Params:
#   input data: the data to learn from
#   target data: the expected output (e.g. a 1 in the row corresponding to the
#                input sequence, indicating the next word.
#   epochs: the number of epochs to train for
#   callbacks: a list of callbacks to execute
history = model.fit(input_sequences,
                    one_hot_labels,
                    epochs=epochs,
                    callbacks=[checkpoint])

# We're done training, so load the checkpoint which contains the best model
# Training is done, so load the best model from the last checkpoint
model = load_model("checkpoint.h5")

# Save the model to the final file
model.save(args.model)
print('Model saved to {}.'.format(args.model))

# Save the tokenizer and max_sequence_length
data = {'max_sequence_len': max_sequence_len,
        'tokenizer': tokenizer.to_json()}
with open(args.data, 'w', encoding='utf-8') as f:
    f.write(json.dumps(data, ensure_ascii=False))
print('Data saved to {}.'.format(args.data))

# All done, but if we're in debug mode, dump some interesting info.
if DEBUG:
    plot_graphs(history, 'accuracy')

    # Dump some basic test info
    text = "trigger dialog"
    next_words = 5

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1,
                                   padding='pre')

        # Get a list of the predicted probabilities corresponding to each word
        # in the index
        predictions = model.predict(token_list)

        # Get the top 5 results
        indices = np.argpartition(predictions, -5)[0][-5:]

        # Create a dict of the results and their probabilities
        results = {}
        for index in indices:
            key = [k for (k, v) in tokenizer.word_index.items() if v == index]
            results.update({key[0]: predictions[0, index]})

        results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

        # Add the top result to the string, if it's not there already
        for result in results:
            if result not in text:
                text = text + " " + result
                break

        print("{}".format(results))

    print(text)