import numpy as np
import nltk
import pandas as pd
from nltk import tokenize, word_tokenize
from spellchecker import SpellChecker

import os

def num_sentences(txt):
    #Get base number of sentences through sentence tokenizer
    tokenized_sentences = tokenize.sent_tokenize(txt)

    # Checking for missed sentences
    new_sentences = [] #holds all new sentences minus the last one
    for i, s in enumerate(tokenized_sentences):
        #get POS tags for analyzing
        tokenized_words = word_tokenize(s)
        POS_tags = nltk.pos_tag(tokenized_words)

        #holds the new sentence
        holder_sentence = ""
        
        #Go through each word & its POS tag
        for i, w in enumerate(POS_tags):
            if i == 0:
                holder_sentence += f"{w[0]} "
                continue

            word = w[0]
            tag = w[1]

            #Assume capitalized, non-proper nouns is a marker for new sentence
            if word[0].isupper() and tag != 'NNP':
                new_sentences.append(holder_sentence)
                holder_sentence = ""

            holder_sentence += f"{word} "

    num_sentences = len(tokenized_sentences) + len(new_sentences)

    #Return score based on where number of sentences falls in the range
    if num_sentences < 10:
        return 0
    elif num_sentences in range(10,14):
        return 1
    elif num_sentences in range(13,17):
        return 2
    elif num_sentences in range(16,20):
        return 3
    elif num_sentences in range (20, 24):
        return 4
    else:
        return 5



def spelling_mistakes(txt):
    pass



def main():

    #Simply here for testing
    test = "I want to do well I am sad I am happy. I am not sad"
    print(num_sentences(test))


    #
    #Getting averages of number of sentences for high and low grades
    #
    index = pd.read_csv("essays_dataset/index.csv", delimiter=";")
    pathToEssays = "essays_dataset/essays/"
    essays = []
    #read in essays 
    for filename in os.listdir(pathToEssays):
        file_path = os.path.join(pathToEssays, filename)

        with open(file_path, 'r') as file:
            text = file.read()
            essays.append((filename, text))

    #Count number of sentences for all essays
    low_sentence_num = 0
    low_sentence_totalSentenceNum = 0
    high_sentence_num = 0
    high_sentence_totalSentenceNum = 0
    for name, essay in essays:
        numSentence = num_sentences(essay)

        df = index[index['filename'] == name]
        score = df['grade'].iloc[0]

        if score == "high":
            high_sentence_num += 1
            high_sentence_totalSentenceNum += numSentence
        else:
            low_sentence_num += 1
            low_sentence_totalSentenceNum += numSentence

    #compute averages
    high_average = high_sentence_totalSentenceNum / high_sentence_num
    low_average = low_sentence_totalSentenceNum / low_sentence_num

    print(f"Average number of sentences in 'high' grades: {high_average}")
    print(f"Average number of sentences in 'low' grades: {low_average}")

    #Holding here in case its needed for future use
    # "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


if __name__ == "__main__":
    main()
