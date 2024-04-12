import numpy as np
import nltk
from nltk import word_tokenize

def agreement():
    pass

def verbMistakes(tokenized_sentences):
    
    notfiniteVerbTags = ['VB', 'VBG', 'VBN'] #Verb tags not included here are finite

    num_mistakes = 0
    for s in tokenized_sentences:
        tokenized_words = word_tokenize(s)
        POS_tags = nltk.pos_tag(tokenized_words)

        finite_verb_count = 0
        for (word,tag) in POS_tags:
            
            if tag not in notfiniteVerbTags and tag[0] == "V":
                finite_verb_count += 1
        
        #If theres no main verb, add to mistakes
        if finite_verb_count != 1:
            num_mistakes += 1

