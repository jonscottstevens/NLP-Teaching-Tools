""""""""""""""""""""""""""""""
Learning to generate Pig Latin
""""""""""""""""""""""""""""""

PigLatin.py provides code for a fun little exercise where we see whether we can use long short-term memory networks (LSTMs) to learn Pig Latin from a small number of examples.

PigLatin.txt provides just 65 examples of words and their Pig Latin counterparts.  It's in a simple comma-separated format with English on the left and Pig Latin on the right, like so:

pig,igpay
dog,ogday
...

Three models are provided, as well as the code to train them, all implemented using keras.

Model #1:  Input features are sequences of letters, treated like time series, with a fixed sequence length of 20.  For example, the word 'pig' would be featurized as a sequence 'p' < 'i' < 'g' < 17 spaces.  Here the alphabet is of length 27 (letters plus space), and there are 65 training examples, and so the training data is a tensor with dimensionality 65x20x27.  There is a 100-dimensional hidden layer and an output layer of the same dimensionality as the input layer, where the input features have binary values, and the output features are between 0 and 1.

The 'translate1' function takes a word, featurizes it, uses model #1 to predict its Pig Latin output, and transforms that output back into word format.

Model #2:  This model utilizes two input layers, each feeding into a hidden layer, where the hidden layers are merged and fed into the output layer.  The two input layers are:  (i) the same features used by model #1, and (ii) a "bag of letters" vector for each word, not representing sequence information.

The 'translate2' function uses model #2 to translate words into Pig Latin.

Model #3:  This model is the same as model #2 except that instead of a bag of letters as secondary input, it is a bag of n-grams (both unigrams and bigrams).

The 'translate3' function uses model #3 to translate words into Pig Latin.

Results:

It is immediately apparent that model #1 is not great.  Even on words in the training set, i.e., words it has seen before, it tends to fail:

>>> translate1('black')
'aaeblay'

Models #2 and #3 do better on the training examples.

>>> translate2('black')
'ackblay'
>>> translate3('black')
'ackblay'

Testing on examples it hasn't seen, model #1 seems to have learned something about Pig Latin, e.g., how long words should be, and the fact that they should end in -ay, but the translations are not at all faithful to what letters were in the input word.

>>> translate1('frack')
'onghray'

This is the motivation for model #2, which provides a dual input including a "bag of letters" aimed at enforcing this faithfulness.  We see that the output is not as wild, but still a bit wild:

>>> translate2('frack')
'acgbtay'

The final refinement is to add bigrams to the bags of letters, and this has a positive effect:

>>> translate3('frack')
'ackfray'

Note that 'frack' is pretty similar to 'black', which was in the training set.  We'll find in general that for this very small data set, the model can only correctly generate translations of words that are similar to words it has seen.  This sheds some light on what the neural network is doing:  It's not learning the simple general rule(s) of Pig Latin, but rather making more specific connections between n-grams and letter sequences in the input and output.

Here's another example of how each successive model does better on familiar-looking words than the last one, where "store" was in the training set, but "gore" and "yore" were not.

>>> translate1('store')
'anestay'
>>> translate2('store')
'orestay'
>>> translate3('store')
'orestay'
>>> 
>>> translate1('gore')
'igaaay'
>>> translate2('gore')
'oragay'
>>> translate3('gore')
'oregay'
>>> 
>>> translate1('yore')
'earaay'
>>> translate2('yore')
'oaroay'
>>> translate3('yore')
'oreyay'

Only model #3 seems to learn any Pig Latin.  But again, it really only works when minimal changes to training-set words are made, e.g. "blink" --> "brink" and "pig" --> "fig":

>>> translate3('blink')
'inkblay'
>>> translate3('brink')
'inkbray'
>>> 
>>> translate3('pig')
'igpay'
>>> translate3('fig')
'igfay'

A word like "stop", which happens not to be very similar to any training-set word, fares worse:

>>> translate3('stop')
'oatsay'

It happens that this data set is small enough that the letters 'w' and 'x' never occur!  We can see that this is detrimental to the model, as it will not output these letters:

>>> translate3('wax')
'aaaay'

Finally, even model #3 fails quite spectacularly on longer and more complex words:

>>> translate3('algorithm')
'iaaalaaaaay'

So all in all, we have not taught our LSTM model to reliably translate words into Pig Latin.  However, some generalizations were indeed learned from just 65 examples.  This suggests that more data would make more robust learning possible.

Comparing model outputs and training the models on different data sets can be used as a teaching tool to better understand what these models are doing and why, and to illustrate strengths and weaknesses of deep learning.

Discussion questions:

1.  What other sorts of features could improve the model?

2.  How much data would we need to match the performance of a simple algorithm that chops off initial consonants, adds them to the end and suffixes -ay?

3.  The simple chopping off algorithm described above will not work perfectly for English words as they are written.  Why not?  What does this say about the merits of using orthographic vs. phonemic represenations of words?

4.  What are some practical tasks that similar models could be leveraged for?