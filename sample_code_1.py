import numpy as np
import nltk
from nltk import tokenize, word_tokenize
from spellchecker import SpellChecker

def num_sentences(txt):
    #Get base number of sentences through sentence tokenizer
    tokenized_sentences = tokenize.sent_tokenize(txt)
    base_num = len(tokenized_sentences)

    #Check for any missed "implied sentences"
    tokenized_words = word_tokenize(txt)
    POS_tags = nltk.pos_tag(tokenized_words)
    print(POS_tags)
    



def spelling_mistakes(txt):
    pass


test = "Hello, I am Synthia"
print(num_sentences(test))