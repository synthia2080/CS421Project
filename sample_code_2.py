import numpy as np
import nltk
from nltk import word_tokenize
import spacy
import pandas as pd
import os
from sample_code_1 import num_sentences
import math

def agreement(tokenized_sentences):
    nlp = spacy.load("en_core_web_sm")
    errors = 0
    auxiliary_verbs = ["has", "do"]

    for sentence in tokenized_sentences:
        if sentence.strip():
            text = nlp(sentence.strip())
            
            has_subject = False
            singular_subject = False
            plural_subject = False
            singular_verb = False
            plural_verb = False
            personal_pronoun = ""

            for token in text:
                if token.dep_ == "nsubj":
                    has_subject = True
                    if token.tag_ in ["NN", "NNP"]:
                        singular_subject = True
                    elif token.tag_ in ["NNS", "NNPS"]:
                        plural_subject = True
                    elif token.tag_ == "PRP$":
                        personal_pronoun = "possessive"
                    elif token.tag_ == "PRP":
                        personal_pronoun = token.text.lower()

                if token.pos_ == "VERB" and token.lemma_.lower() not in auxiliary_verbs:
                    if token.tag_ in ["VBZ", "VBD"]:
                        singular_verb = True
                    if token.tag_ == "VBP":
                        plural_verb = True
                    if personal_pronoun in ["he", "she", "it"]:
                        if token.tag_ == "VBP":
                            errors += 1
                            print("Error:", sentence.strip())
                    elif personal_pronoun in ["i", "you", "we", "they"]:
                        if token.tag_ == "VBZ":
                            errors += 1
                            print("Error:", sentence.strip())
                
                if token.pos_ == "VERB" and token.lemma_.lower() in auxiliary_verbs:
                    if token.tag_ in "VBZ":
                        singular_verb = True
                    if personal_pronoun in ["he", "she", "it"]:
                        if token.tag_ == "VBP":
                            errors += 1
                            print("Error:", sentence.strip())
                    elif personal_pronoun in ["i", "you", "we", "they"]:
                        if token.tag_ == "VBZ":
                            errors += 1
                            print("Error:", sentence.strip())

            if not has_subject:
                errors += 1
                print("Error:", sentence.strip())
    
    return errors

def verbMistakes(tokenized_sentences):
    numSentences = len(tokenized_sentences)
    notfiniteVerbTags = ['VB', 'VBG', 'VBN'] #Verb tags not included here are finite
    present = ['VBP', 'VBG','VBZ', 'VB']
    past = ['VBD', 'VBN', 'VB']
    processor = spacy.load("en_core_web_sm")

    all_root_verbs = []
    num_mistakes = 0

    for s in tokenized_sentences:
        POS_tags = processor(s)

        finite_verb_count = 0
        for i, token in enumerate(POS_tags):
            if i == len(POS_tags)-1:
                continue
            
            #Incorrect verb tense following infinitive
            if token.tag_ == "TO" and POS_tags[i+1].tag_ != "VB":
                num_mistakes +=1

            if (token.tag_[0] == "V" or token.tag_ == "MD") and token.dep_ == "ROOT":
                all_root_verbs.append(token.tag_)
                
                children_dep = [c.dep_ for c in token.children]

                #Check if infinite verbs are indeed finite
                if token.tag_ in notfiniteVerbTags and not ("aux" in children_dep or "auxpass" in children_dep or "cop" in children_dep):
                    num_mistakes += 1
                elif token.tag_ not in notfiniteVerbTags and "PART" in [c.pos_ for c in token.children]: #Second auxilary check, 
                    num_mistakes += 1
                else:
                    finite_verb_count += 1


        #If theres no main verb, add to mistakes
        if finite_verb_count < 1:
            num_mistakes += 1
        elif finite_verb_count > 1:
            num_mistakes += finite_verb_count-1 #Subtract 1 since one of the verbs must be right


    #Check for different tenses (through root verbs) in consecutive sentences being different tense (not ideal but best solution for now)
    for i, v in enumerate(all_root_verbs):
        #Skip last verb
        if i == len(all_root_verbs) - 1:
            continue
        
        if v in past and all_root_verbs[i+1] not in past:
            num_mistakes += 1
        elif v in present and all_root_verbs[i+1] not in present:
            num_mistakes += 1
        elif v == "MD" and all_root_verbs[i+1] != "MD" and all_root_verbs[i+1] != 'VB':
            num_mistakes += 1

    normalized_mistakes = (float(num_mistakes) / numSentences)*100

    #Return based on average number of mistakes for essays
    if normalized_mistakes < 47:
        return 5
    elif normalized_mistakes > 57:
        return 1
    else:
        return 1 + 4 * (normalized_mistakes - 47) / (57 - 47)

