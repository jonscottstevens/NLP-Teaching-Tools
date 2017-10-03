"""
this script trains neural net classifiers for binary seniment analysis on the IMDB large review dataset, stores the featurizers and data objects as pickle files and persists the neural net models as HDF5 files... if those objects already exist, they are loaded so that they can be evaluated, compared and played around with...
"""

from __future__ import division
import re
import spacy
import os
from sklearn.externals import joblib
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from numpy import matrix, hstack, random

# set seed for randomization

random.seed(7)

# set path to directory containing script

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# load SpaCy's medium-vocabulary model of English

english = spacy.load('en_core_web_md')

# uc function guarantees unicode; needed for Python 2, because SpaCy needs unicode and Python 2 gives ascii

def uc(string):
	if isinstance(string, unicode):
		return string
	else:
		return unicode(string, errors='ignore')

# take out html tags and convert word tokens to lowercase

def clean(string):
	stripped = re.sub('<.*?>', '', string)
	return stripped.lower()

# add dummy words encoding words that have a negation dependency

def add_negs(string):
	nlp = english(uc(string))
	negs = []
	for n in nlp:
		if n.dep_ == 'neg':
			negs.append('negneg'+n.head.orth_)
	output = uc(string)
	for neg in negs:
		output = output + ' ' + neg + ' ' + neg
	return output

# extract document vectors by averaging word vectors for content words; if negated words have been added to the document, penalize them by adding negated vectors

def docvec(nlp):
	doc = []
	for w in nlp:
		if w.pos_ in ["NOUN","ADJ","VERB","ADV"]:
			doc.append(w.vector)
		if "negneg" in w.orth_:
			word = english(w.orth_[6:len(w.orth_)])
			doc.append(word.vector * -1)
	return sum(doc)/len(doc)

"""
this script shares some variables across model creation functions; a quick and dirty way to handle these variables is to check whether they already exist, and create them as needed...

the check_load function does this for raw training and testing data...

check_create checks whether sklearn objects like featurizers and featurized data sets have been stored as pickle files, and creates them if not...

check_create_model checks whether keras models have been stored...

this way, different objects can be modified/replaced as needed, without running the entire script
"""

def check_load(varname, datatype): # load train or test data
	try:
		exec(varname, globals())
	except NameError:
		exec("global {}".format(varname), globals())
		exec("{} = load_files('{}', categories=['neg', 'pos'], load_content=True)".format(varname, datatype), globals())

def check_create(varname): # create objects only if not pickled
	try:
		exec("{} = joblib.load('{}.pkl')".format(varname, varname), globals())
	except IOError:
		exec("global {}".format(varname), globals())
		exec("{} = create_{}()".format(varname, varname), globals())
		exec("joblib.dump({}, '{}.pkl')".format(varname, varname), globals())

def check_create_model(varname): # train models only if not persisted
	try:
		exec("{} = load_model('{}.h5')".format(varname, varname), globals())
	except IOError:
		exec("global {}".format(varname), globals())
		exec("{} = create_{}()".format(varname, varname), globals())
		exec("{}.save('{}.h5')".format(varname, varname), globals())

"""
below are the functions to create the relevant model objects:
-the target values for training and testing data,
-a Tf-Idf featurizer and associated featurized data (both train and test)
-a simple dense neural net model for binary classification using Tf-Idf inputs
-featurized data with document vectors added
-a classifier using both Tf-Idf and doc vector inputs
-a Tf-Idf featurizer with negated words added as features, and associated featurized data
-a classifier that uses Tf-Idfs with negation
-the 'full' model that uses Tf-Idfs with negation and document vectors
"""

def create_train_targets():
	check_load('tfidf1', 'train')
	return tfidf1['target']

def create_test_targets():
	check_load('tfidf2', 'test')
	return tfidf2['target']

def create_tfidf_featurizer():
	return Pipeline([('counts', CountVectorizer()),('tfidf', TfidfTransformer())]).fit(tfidf1['data'])

def create_tfidf_train():
	check_load('tfidf1', 'train')
	for n in xrange(0, len(tfidf1['data'])):
		tfidf1['data'][n] = clean(tfidf1['data'][n])
	check_create('tfidf_featurizer')
	return tfidf_featurizer.transform(tfidf1['data']).todense()

def create_tfidf_test():
	check_load('tfidf2', 'test')
	for n in xrange(0, len(tfidf2['data'])):
		tfidf2['data'][n] = clean(tfidf2['data'][n])
	check_create('tfidf_featurizer')
	return tfidf_featurizer.transform(tfidf2['data']).todense()

def create_tfidf_model():
	model = Sequential()
	model.add(Dense(32, input_dim=(tfidf_train.shape[1]), activation='relu', init='uniform'))
	model.add(Dense(16, activation='relu', init='uniform'))
	model.add(Dense(1, activation='sigmoid', init='uniform'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(tfidf_train, train_targets, nb_epoch=2, batch_size=100)
	return model

def create_docvec_train():
	check_load('tfidf1', 'train')
	train_vectors = []
	for n in xrange(0,len(tfidf_train)):
		train_vectors.append(docvec(english(uc(tfidf1['data'][n]))))
	return hstack((tfidf_train, train_vectors))

def create_docvec_test():
	check_load('tfidf2', 'test')
	test_vectors = []
	for n in xrange(0,len(tfidf_test)):
		test_vectors.append(docvec(english(uc(tfidf2['data'][n]))))
	return hstack((tfidf_test, test_vectors))

def create_docvec_model():
	model = Sequential()
	model.add(Dense(32, input_dim=(docvec_train.shape[1]), activation='relu', init='uniform'))
	model.add(Dense(16, activation='relu', init='uniform'))
	model.add(Dense(1, activation='sigmoid', init='uniform'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(docvec_train, train_targets, nb_epoch=2, batch_size=100)
	return model

def create_negation_featurizer():
	return Pipeline([('counts', CountVectorizer()),('tfidf', TfidfTransformer())]).fit(negation1['data'])

def create_full_train():
	check_load('tfidf1', 'train')
	global negation1
	negation1 = tfidf1
	for n in xrange(0, len(negation1['data'])):
		negation1['data'][n] = add_negs(clean(negation1['data'][n]))
	check_create('negation_featurizer')
	neg_tfidf_train = negation_featurizer.transform(negation1['data']).todense()
	train_vectors = []
	for n in xrange(0,len(neg_tfidf_train)):
		train_vectors.append(docvec(english(negation1['data'][n])))
	return hstack((neg_tfidf_train, train_vectors))
	

def create_full_test():
	check_load('tfidf2', 'test')
	global negation2
	negation2 = tfidf2
	for n in xrange(0, len(negation2['data'])):
		negation2['data'][n] = add_negs(clean(negation2['data'][n]))
	check_create('negation_featurizer')
	neg_tfidf_test = negation_featurizer.transform(negation2['data']).todense()
	test_vectors = []
	for n in xrange(0,len(neg_tfidf_test)):
		test_vectors.append(docvec(english(negation2['data'][n])))
	return hstack((neg_tfidf_test, test_vectors))

def create_negation_model():
	model = Sequential()
	model.add(Dense(32, input_dim=(full_train.shape[1]-300), activation='relu', init='uniform'))
	model.add(Dense(16, activation='relu', init='uniform'))
	model.add(Dense(1, activation='sigmoid', init='uniform'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(full_train[:,0:full_train.shape[1]-300], train_targets, nb_epoch=2, batch_size=100)
	return model

def create_full_model():
	model = Sequential()
	model.add(Dense(32, input_dim=(full_train.shape[1]), activation='relu', init='uniform'))
	model.add(Dense(16, activation='relu', init='uniform'))
	model.add(Dense(1, activation='sigmoid', init='uniform'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(full_train, train_targets, nb_epoch=2, batch_size=100)
	return model

"""
the check_create functions below need to be executed if the models and featurizers are not already stored...

comment out data files you don't need to load in order to save memory...

creating everything from scratch at once takes some time (~1 hour on a standard laptop)... as a teaching demo, it is practical to only create the first model from scratch 
"""

#check_create('train_targets')
#check_create('test_targets')
#check_create('tfidf_train')
#check_create('tfidf_test')
#check_create_model('tfidf_model')
#check_create('docvec_train')
#check_create('docvec_test')
#check_create_model('docvec_model')
#check_create('full_train')
#check_create('full_test')
#check_create_model('negation_model')
#check_create_model('full_model')

"""
load up the featurizers, featurized test data and models if they exist

NOTE:  this takes a few seconds, because test data is loaded, and it's big... it would be quicker to cache the evaluation scores and only have the REPL evaluate novel sentences on the fly; however, it can be nice as a teaching tool to see the evaluation happen in real time
"""

try:
	tfidf_featurizer = joblib.load('tfidf_featurizer.pkl')
	negation_featurizer = joblib.load('negation_featurizer.pkl')
	tfidf_test = joblib.load('tfidf_test.pkl')
	docvec_test = joblib.load('docvec_test.pkl')
	full_test = joblib.load('full_test.pkl')
	test_targets = joblib.load('test_targets.pkl')
	tfidf_model = load_model('tfidf_model.h5')
	docvec_model = load_model('docvec_model.h5')
	negation_model = load_model('negation_model.h5')
	full_model = load_model('full_model.h5')
	chosen_model = tfidf_model
except:
	print("featurizers, featurized test data, and/or models need to be created!")

print("use \'choose\' command to choose \'tfidf\', \'docvec\', \'negation\' or \'full\' models")
print("use \'evaluate\' command to evaluate chosen model on test data")
print("use \'sentiment\' command to score sentences with chosen model")
print("use \'emoji\' command for binary judgments")

"""
these commands can be executed in the REPL to demonstrate the effect of adding different kinds of linguistic information to the sentiment analysis model
"""

# pick which model you're using

def choose(model_name):
	exec("chosen_model = {}_model".format(model_name), globals())

# determine what the right featurized data object is for evaluation

def test_data(model):
	if model == tfidf_model:
		return tfidf_test
	if model == docvec_model:
		return docvec_test
	if model == negation_model:
		return full_test[:,0:full_test.shape[1]-300]
	else:
		return full_test

# evaluate accuracy

def evaluate():
	return chosen_model.evaluate(test_data(chosen_model), test_targets)

# return sentiment score for a given sentence

def sentiment(sentence):
	if chosen_model == tfidf_model:
		features = tfidf_featurizer.transform([clean(sentence)]).todense()

	if chosen_model == docvec_model:
		tfidfs = tfidf_featurizer.transform([clean(sentence)]).todense()
		vec = matrix(docvec(english(uc(sentence))))
		features = hstack((tfidfs, vec))

	if chosen_model == negation_model:
		features = negation_featurizer.transform([add_negs(clean(sentence))]).todense()

	if chosen_model == full_model:
		neg_tfidfs = negation_featurizer.transform([add_negs(clean(sentence))]).todense()
		vec = matrix(docvec(english(add_negs(uc(sentence)))))
		features = hstack((neg_tfidfs, vec))

	return chosen_model.predict(features)

# binary classification of a sentence in the form of an emoji

def emoji(sentence):
	score = sentiment(sentence)
	if score > 0.5:
		return ':-)'
	else:
		return ':-('
