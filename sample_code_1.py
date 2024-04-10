import numpy as np
import nltk
import pandas as pd
from nltk import tokenize, word_tokenize
from spellchecker import SpellChecker

import os

def num_sentences(txt):
    """
    Tokenizes essays by sentence, then for each sentence, get POS tags and search for any extra sentences missed by
    the nltk sentence tokenizer

    txt: essay as a string
    returns: score as an int
    """
    #Get base number of sentences through sentence tokenizer
    tokenized_sentences = tokenize.sent_tokenize(txt)

    comma_list = [',', ';']
    not_EOS_pos_tags = ["DT","IN", "CC","DT", "PRP$", "WP$", "IN", "VB", "VBP", "VBZ", "VBD"] #Tags that shouldn't be at end of sentence
    new_sentences = [] #holds all new sentences minus the last one

    # Checking for missed sentences
    for s in tokenized_sentences:
        #get POS tags for analyzing
        tokenized_words = word_tokenize(s)
        POS_tags = nltk.pos_tag(tokenized_words)

        #holds the new sentence
        holder_sentence = ""

        #Go through each word & its POS tag
        for i2, (word, tag) in enumerate(POS_tags):
            if i2 == 0:
                holder_sentence += f"{word} "
                continue
            
            word_before = POS_tags[i2-1][0]
            tag_before = POS_tags[i2-1][1]

            #Assume capitalized, non-proper nouns is a marker for new sentence
            if word[0].isupper() and tag != 'NNP' and word_before not in comma_list and tag_before not in not_EOS_pos_tags:
                new_sentences.append(holder_sentence)
                holder_sentence = ""

            holder_sentence += f"{word} "

        new_sentences.append(holder_sentence)


    notfiniteVerbTags = ['VB', 'VBG', 'VBN']
    sub_coord_tags = ['CC', 'IN', 'WP', 'WDT', 'WP$', 'RB'] #Tags usually indicating subordinate/coordinate clauses
    num_missed_sentences = 0
    num_sentences_offset = 0

    #Check for more missed sentences based on finite verbs
    for s in new_sentences:
        tokenized_words = word_tokenize(s)
        POS_tags = nltk.pos_tag(tokenized_words)

        #Get number of finite verbs in sentence
        num_finite_verbs = sum(1 for item in POS_tags if item[1] not in notfiniteVerbTags and item[1][0] == "V")
        if num_finite_verbs <= 1:
            continue
        
        #Get number of tags relating to subordinate/coordinate clauses
        sum_sub_coord_tags = sum(1 for t in POS_tags if t[1] in sub_coord_tags)
        sum_and_words = sum(1 for t in POS_tags if t[0].lower() == "and") #count number of ands

        #More finite verbs than counted sub/coord clauses, should indicate a missed sentence
        if sum_sub_coord_tags < num_finite_verbs:
            #Prevent extra sentence count for sentences with "and"
            if sum_and_words != 0:
                sum_and_words += 1

            missed = (num_finite_verbs-sum_and_words)
            num_missed_sentences += missed
            if missed > 0:
                num_sentences_offset += 1 #offset used later to not count this current sentence

    num_sentences = num_missed_sentences + (len(new_sentences) - num_sentences_offset)

    #Return score based on where number of sentences falls in the range
    # if num_sentences < 10:
    #     return 0
    # elif num_sentences in range(10,14):
    #     return 1
    # elif num_sentences in range(13,17):
    #     return 2
    # elif num_sentences in range(16,20):
    #     return 3
    # elif num_sentences in range (20, 24):
    #     return 4
    # else:
    #     return 5

    if num_sentences <= 0:
        return 0
    elif num_sentences >= 22:
        return 5
    else:
        return (num_sentences**2 / 22.0**2) * 5



def spelling_mistakes(txt):
    pass


def getAverageSentCount():
    """
    Gets averages of number of sentences for high and low grades
    """
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

def main():

    #Simply here for testing
    # test = "Most people do not walk to work; instead, they drive or take the train. I love trains and I love brains and I love hating cars I am cool therefore you are cool too"
    test = "I want to do well I am sad i am happy. i am cool i am not cool he is dumb. After he and I finished my homework, I went to bed and also brushed my teeth."

    print(num_sentences(test))

    getAverageSentCount()

    #Holding here in case its needed for future use
    # "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


if __name__ == "__main__":
    main()
