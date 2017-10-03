""""""""""""""""""
Sentiment analysis
""""""""""""""""""

Sentiment.py contains code to be used for hands-on demos for teaching a crash course in sentiment analysis.  The code trains a series of neural networks with two hidden layers, implemented in keras, on the IMDB large reviews dataset.  The variant models differ only in what features are used as input.  Tf-Idf features, created using sklearn, serve as the basis for the input, from which four variants are created by adding linguistic information using the SpaCy package:

1) Tf-Idf features only
2) Tf-Idf features with document vectors (obtained by averaging SpaCy word vectors for content words)
3) Tf-Idf features with negated words added as separate tokens (obtained using SpaCy's dependency parser)
4) Tf-Idf features with both document vectors and negated words

This demonstration can be used as an aid to better understand how to do several important tasks, including:

-Preprocess messy data
-Use a machine learning pipeline to create a data featurizer and transform the data into machine-readable form
-Implement a neural network for binary classification
-Use off-the-shelf NLP tools
-Evaluate and compare model variants
-Apply linguistic insights to machine learning and model design

The final model comparison shows two important trends:

A.  Models 1--4 differ in their accuracy on the testing data, in the way you'd expect, though only slightly.  This suggests a ceiling of about 88--89% accuracy for this model architecture, a bit above that of the original 2011 ACL paper for which this dataset was constructed, but below the accuracy that more complex algorithms have since achieved.

Model		Accuracy
Tf-Idf (1)	0.879
+Vectors (2)	0.886
+Negation (3)	0.881
Everything (4)	0.887

B.  Error analysis and testing the model's predictions on single short sentences and phrases suggests that adding additional linguistic features makes the model more robust to shorter inputs.  Likely a ceiling effect on the testing data is achieved because the testing data consist of paragraph-long reviews.  But performance on shorter chunks of sentences is clearly different between the models!  This underscores the need for different types of evaluations.  Just a few examples:

Example				Tf-Idf (1)	+Vectors (2)	+Negation (3)	Everything (4)
'not good' (NEG)		wrong		wrong		right		right
'gnarly dude' (POS)		wrong		right		wrong		right
'never boring' (POS)		wrong		wrong		wrong		right

Students of sentiment analysis will be encouraged to further explore differences between the models, and to test the following hypotheses:  (i) adding word/document vectors boosts performance in cases where low-frequency words are used to convey sentiment, and (ii) adding information about negation dependencies boost performance where strongly sentiment-laden words appear in the scope of negation.

""""""""""""""""""
SENTIMENTOR demo
""""""""""""""""""

SENTIMENTOR.py contains a stripped down demo of the 'choose', 'sentiment' and 'emoji' functions, with no evaluation data loaded.  It's quicker to load, so you can jump right in.  Here are some example commands you can use after importing SENTIMENTOR:

from SENTIMENTOR import *
choose('full')
sentiment('this is a nice demo')
emoji('this is a nice demo')