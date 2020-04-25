import nltk
nltk.download('brown')

from nltk.corpus import brown

brown.words() # Returns a list of strings
len(brown.words()) # No. of words in the corpus
brown.sents() # Returns a list of list of strings 