# -*- coding: utf-8 -*-
"""
Created on Sun May 10 12:42:17 2020

@author: Aaditya Chopra
"""


import tensorflow_datasets as tfds
import numpy as np

imdb,info = tfds.load("imdb_reviews", with_info = True, as_supervised = True)

train_data = imdb['train']
test_data = imdb['test']

training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []

for sentence, label in train_data:
    training_sentences.append(str(sentence.numpy()))
    training_labels.append(label.numpy())

for sentence, label in test_data:
    testing_sentences.append(str(sentence.numpy()))
    testing_labels.append(label.numpy())

training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

import string

def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    s = sentence.split()
    s = [word for word in s if len(word)>1]
    s = [word for word in s if word.isalpha()]
    sentence = ' '.join(s)
    return sentence


for i in range(0,len(training_sentences)):
    training_sentences[i] = clean_sentence(training_sentences[i])

for i in range(0,len(testing_sentences)):
    testing_sentences[i] = clean_sentence(testing_sentences[i])


vocab = set()
max_len = 0
for sentence in training_sentences:
    s = sentence.split()
    max_len = max(max_len, len(s))
    for word in s:
        vocab.add(word)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
embedding_dim = 16
max_length = 120
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

def make_sequence(sentence):
    sequence = tokenizer.texts_to_sequences(sentence)
    sequence = pad_sequences(sequence, maxlen = max_length, truncating = 'post')
    return sequence
    
sequences = make_sequence(training_sentences)
testing_sequences = make_sequence(testing_sentences)
    

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length = max_length))
model.add(Flatten())
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

checkpoint_filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(
    checkpoint_filepath, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min'
    )

model.fit(sequences,
          training_labels,
          validation_data=(testing_sequences, testing_labels),
          epochs = 30,
          verbose = 1,
          callbacks=[checkpoint]
          )

from tensorflow.keras.models import load_model

model_weights = "model_weights.h5"
model = load_model(model_weights)

def prediction(review):
    review = clean_sentence(review)
    review = make_sequence(review)
    result = model.predict(review)
    print(result)
    """if result > 0.5:
        print("POSITIVE REVIEW\n")
    else:
        print("NEGATIVE REVIEW\n")"""

review = input()
prediction(review)