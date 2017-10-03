"""
a simple sentiment analysis demo
"""

from __future__ import division
import re
import spacy
import os
from sklearn.externals import joblib
from numpy import matrix, hstack, random
from keras.models import load_model

os.chdir(os.path.dirname(os.path.abspath(__file__)))

english = spacy.load('en_core_web_md')

def uc(string):
	if isinstance(string, unicode):
		return string
	else:
		return unicode(string, errors='ignore')

def clean(string):
	stripped = re.sub('<.*?>', '', string)
	return stripped.lower()

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

def docvec(nlp):
	doc = []
	for w in nlp:
		if w.pos_ in ["NOUN","ADJ","VERB","ADV"]:
			doc.append(w.vector)
		if "negneg" in w.orth_:
			word = english(w.orth_[6:len(w.orth_)])
			doc.append(word.vector * -1)
	return sum(doc)/len(doc)

tfidf_featurizer = joblib.load('tfidf_featurizer.pkl')
negation_featurizer = joblib.load('negation_featurizer.pkl')
tfidf_model = load_model('tfidf_model.h5')
docvec_model = load_model('docvec_model.h5')
negation_model = load_model('negation_model.h5')
full_model = load_model('full_model.h5')
chosen_model = tfidf_model

print("use \'choose\' command to choose \'tfidf\', \'docvec\', \'negation\' or \'full\' models")
print("use \'sentiment\' command to score sentences with chosen model")
print("use \'emoji\' command for binary judgments")

def choose(model_name):
	exec("chosen_model = {}_model".format(model_name), globals())

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

def emoji(sentence):
	score = sentiment(sentence)
	if score > 0.5:
		return ':-)'
	else:
		return ':-('
