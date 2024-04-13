import numpy as np
import nltk
from nltk import word_tokenize
import spacy

def agreement():
    pass

def verbMistakes(tokenized_sentences):
    
    notfiniteVerbTags = ['VB', 'VBG', 'VBN'] #Verb tags not included here are finite
    present = ['VBP', 'VBG','VBZ']
    past = ['VBD', 'VBN']
    precede_VBG = ['VBZ', 'VBD', 'VBP', 'VBP', 'IN', 'PRP$']
    processor = spacy.load("en_core_web_sm")

    all_finite_verbs = []

    num_mistakes = 0
    for s in tokenized_sentences:
        POS_tags = processor(s)

        verbs = [v for v in POS_tags if v.pos_ == "VERB"]
        for token in verbs:
            print(f"{token.text} {token.tag_} {token.dep_}")

        finite_verb_count = 0
        for i, token in enumerate(POS_tags):
            if i != 0:
                prev_tag = POS_tags[i-1]
                #Preceding gerund mistakes, missing auxilary
                if i != 0 and token.tag_ == "VBG" and prev_tag.pos_ != "AUX" or prev_tag.tag_ != "IN":
                    num_mistakes += 1


            #Find finite verbs
            if token.pos_ not in notfiniteVerbTags and token.pos_[0] == "V":
                all_finite_verbs.append(token.tag_)
                finite_verb_count += 1

        
        #If theres no main verb, add to mistakes
        if finite_verb_count < 1:
            num_mistakes += 1
        elif finite_verb_count > 1:
            num_mistakes += finite_verb_count
        print()

    for i, v in enumerate(all_finite_verbs):
        #Skip last verb
        if i == len(all_finite_verbs) - 1:
            continue
        
        #Check for verbs in consecutive sentences being different tense (not ideal solution for now)
        if (v in past and all_finite_verbs[i+1] in present) or (v in present and all_finite_verbs[i+1] in past):
            num_mistakes += 1
    print(num_mistakes)


ddd = [('')]
sentences = ['I remember that I go to see the eclipse', 'I remember that I went to see the eclipse']

verbMistakes(sentences)
