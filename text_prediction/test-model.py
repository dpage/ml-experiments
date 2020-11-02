import argparse
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

DEBUG = False

# Get command line options
parser = argparse.ArgumentParser(description='Test a pre-trained Tensorflow model with text data')
parser.add_argument("-d", "--data", required=True, help="the file to load data from")
parser.add_argument("-m", "--model", required=True, help="the file to load the model from")

args = parser.parse_args()

# Load the model
model = tf.keras.models.load_model(args.model)

# Load the data
f = open(args.data, "r")
data = json.load(f)
max_sequence_len = data['max_sequence_len']
tokenizer = tokenizer_from_json(data['tokenizer'])

text = input("Enter text (blank to quit): ")
while text != '':
    words = input("Number of words to generate (default: 1): ")
    if words == '':
        words = 1
    else:
        words = int(words)

    for _ in range(words):
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

        if DEBUG:
            print("    {}".format(results))

    print("Results: {}".format(text))
    text = input("Enter text (blank to quit): ")