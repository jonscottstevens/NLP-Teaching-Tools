"""
This is mostly meant for fun---to see if we can get a neural network to learn Pig Latin from a small number of completely regular examples.  The accompanying data file, 'PigLatin.txt' gives just 65 training examples.
"""

import os
import numpy as np

# set seed for randomization

np.random.seed(7)

# set path to directory containing script

os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""
We need to featurize our data.  We'll try using a time series approach to represent sequences of letters in words, as well as representing simple bags of letters
"""

alphabet = 'abcdefghijklmnopqrstuvwxyz '
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

# convert a word to time series format; maxlength is the fixed number of characters to consider... after the end of the word, spaces are added to make every word of length maxlength

def sequentialize(word, maxlength):
    output = np.zeros((maxlength,len(alphabet)))
    for n in xrange(0,maxlength):
        try:
            output[n,char_to_int[word[n]]] = 1
        except IndexError:
            output[n,len(alphabet)-1] = 1
    return output

# bags of letters

def bag(word, maxlength):
    output = np.zeros((maxlength,len(alphabet)))
    for letter in word:
        output[0,char_to_int[letter]] = 1
    return output

# let's fix the maxlength parameter at 20

maxlen = 20

# create first set of training and testing data

with open('PigLatin.txt', 'r') as training:
        insouts = []
        for line in training:
            insouts.append(line.split(","))
        ins = []
        outs = []
        for element in insouts:
            ins.append(element[0])
            outs.append(element[1])
        outs = [word[0:len(word)-1] for word in outs]
        X = np.array([sequentialize(word, maxlen) for word in ins]) # training data (time series of letters)
        XX = np.array([bag(word, maxlen) for word in ins]) # secondary training data (bags of letters)
        Y = np.array([sequentialize(word, maxlen) for word in outs]) # testing data (time series of letters)

# train LSTM model (long short-term memory; a kind of recurrent neural net for learning from sequences)

from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Input, concatenate
from keras.layers.core import Dropout

"""
The first model uses sequences to predict sequences.
The second model (model2) uses sequences and bags of letters to predict sequences
The training algorithm runs 500 times, so training takes a little while
"""

try:
    model = load_model('MODEL.h5') # load model if it exists
except:
    print("Creating sequential LSTM model")
    model = Sequential()
    model.add(LSTM(100, input_shape=(maxlen, len(alphabet)), init = 'uniform', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(len(alphabet), activation='sigmoid', init = 'uniform', return_sequences=True))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, nb_epoch=500, batch_size=1)
    model.save('MODEL.h5')

try:
    model2 = load_model('MODEL2.h5')
except:
    print("Creating merged LSTM model")
    main_input = Input((maxlen, len(alphabet)))
    auxiliary_input = Input((maxlen, len(alphabet)))
    hidden_layer1 = LSTM(100, input_shape=(maxlen, len(alphabet)), init = 'uniform', return_sequences=True)(main_input)
    hidden_layer2 = LSTM(100, input_shape=(1, len(alphabet)), init = 'uniform', return_sequences=True)(auxiliary_input)
    merged_layer = concatenate([hidden_layer1, hidden_layer2])
    dropout_layer = Dropout(0.2)(merged_layer)
    output_layer = LSTM(len(alphabet), activation='sigmoid', init = 'uniform', return_sequences=True)(dropout_layer)
    model2 = Model(inputs=[main_input, auxiliary_input], outputs=[output_layer])
    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model2.fit([X,XX], Y, nb_epoch=500, batch_size=1)
    model2.save('MODEL2.h5')
    
# a function to take an output tensor and convert it back to a word

def interpret(prediction):
    output = ''
    for row in prediction:
        max_value = max(row)
        maxcolumn = np.where(row==max_value)[0][0]
        output = output + int_to_char[maxcolumn]
    return output

# translate1 uses model #1 to try to translate a word into Pig Latin; translate2 uses model #2

def translate1(word):
    return interpret(model.predict(np.array([sequentialize(word, maxlen)]))[0]).strip()

def translate2(word):
    return interpret(model2.predict([np.array([sequentialize(word, maxlen)]), np.array([bag(word, maxlen)])])[0])

"""
For the third model, we convert our bag-of-letters features to n-grams, considering both unigrams and bigrams...
This changes the dimensionality of the inputs, but not of the output
"""

ngram_alphabet = list(alphabet)
bigrams = []
for unigram1 in ngram_alphabet:
    for unigram2 in ngram_alphabet:
        bigrams.append(unigram1+unigram2)
        
ngram_alphabet = ngram_alphabet + bigrams

ngram_to_int = dict((n, i) for i, n in enumerate(ngram_alphabet))
int_to_ngram = dict((i, n) for i, n in enumerate(ngram_alphabet))

def ngram_sequentialize(word, maxlength):
    output = np.zeros((maxlength,len(ngram_alphabet)))
    for n in xrange(0,maxlength):
        try:
            output[n,ngram_to_int[word[n]]] = 1
        except IndexError:
            output[n,ngram_to_int[' ']] = 1
    return output

def ngram_bag(word, maxlength):
    output = np.zeros((maxlength,len(ngram_alphabet)))
    ngrams = [word[0]]
    for i in xrange(1,len(word)):
        ngrams.append(word[i])
        ngrams.append(word[i-1]+word[i])
    for n in ngrams:
        output[0,ngram_to_int[n]] = 1
    return output

# new training and secondary training input

XXX = np.array([ngram_sequentialize(word, maxlen) for word in ins])
XXXX = np.array([ngram_bag(word, maxlen) for word in ins])

"""
The third model (model3) uses n-gram features in conjunction with letter sequences to predict Pig Latin representations...
The function translate3 uses this model to translate individual words
"""

try:
    model3 = load_model('MODEL3.h5')
except:
    print("Creating merged LSTM model with bigram features")
    ngram_main_input = Input((maxlen, len(ngram_alphabet)))
    ngram_auxiliary_input = Input((maxlen, len(ngram_alphabet)))
    ngram_hidden_layer1 = LSTM(100, input_shape=(maxlen, len(ngram_alphabet)), init = 'uniform', return_sequences=True)(ngram_main_input)
    ngram_hidden_layer2 = LSTM(100, input_shape=(1, len(ngram_alphabet)), init = 'uniform', return_sequences=True)(ngram_auxiliary_input)
    ngram_merged_layer = concatenate([ngram_hidden_layer1, ngram_hidden_layer2])
    ngram_dropout_layer = Dropout(0.2)(ngram_merged_layer)
    ngram_output_layer = LSTM(len(alphabet), activation='sigmoid', init = 'uniform', return_sequences=True)(ngram_dropout_layer)
    model3 = Model(inputs=[ngram_main_input, ngram_auxiliary_input], outputs=[ngram_output_layer])
    model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model3.fit([XXX,XXXX], Y, nb_epoch=500, batch_size=1)
    model3.save('MODEL3.h5')

def translate3(word):
    return interpret(model3.predict([np.array([ngram_sequentialize(word, maxlen)]), np.array([ngram_bag(word, maxlen)])])[0]).strip()
