import numpy as np
import nltk
import pandas as pd
from nltk import tokenize, word_tokenize
from spellchecker import SpellChecker
import os

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
        numSentence, new_sentences = num_sentences(essay)

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
    start_tags = ["PRP", "DT", "WP", "IN", "RB", "NN", "NNP"]
    new_sentences = [] #holds all new sentences minus the last one

    # Checking for missed sentences based off of capitalization
    for s in tokenized_sentences:
        #get POS tags for analyzing
        tokenized_words = word_tokenize(s)
        tokenized_words = ["I" if word == "i" else word for word in tokenized_words] #Helps POS tagger appropriately label "I"

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


    notfiniteVerbTags = ['VB', 'VBG', 'VBN'] #Verb tags not included here are finite
    sub_coord_tags = ['CC', 'IN', 'WP', 'WDT', 'WP$', 'RB'] #Tags usually indicating subordinate/coordinate clauses
    start_tags = ["PRP", "DT", "WP", "IN", "RB", "NN", "NNP"] #Tags that usually are in beginning of sentences


    new_new_sentences = []
    #Check for more missed sentences based on finite verbs
    for s in new_sentences:
        tokenized_words = word_tokenize(s)
        tokenized_words = ["I" if word == "i" else word for word in tokenized_words] #Helps POS tagger appropriately label "I"
        POS_tags = nltk.pos_tag(tokenized_words)

        #Get index and tuple of finite verbs
        finite_verbs = [value for index, value in enumerate(POS_tags) if value[1] not in notfiniteVerbTags and value[1][0] == "V"]
        num_finite_verbs = len(finite_verbs)

        #Sentence already proper
        if num_finite_verbs <= 1:
            new_new_sentences.append(s)
            continue
        
        #Get number of tags relating to subordinate/coordinate clauses
        sum_sub_coord_tags = [value for index, value in enumerate(POS_tags) if value[1] in sub_coord_tags]
        num_sub_coords = len(sum_sub_coord_tags)

        #More finite verbs than counted sub/coord clauses, should indicate a missed sentence
        if num_sub_coords < num_finite_verbs:
            finite_count = 0
            holder_sentence = ""

            #Go through tags to determine new sentence based on if a finite verb is read and whether the tag is part of possible EOS tag
            for (word, tag) in POS_tags:
                if num_finite_verbs > 1 and (word, tag) in finite_verbs:
                    finite_count = 1
                
                #Indicates most likely start of a new sentence, excluding subordinate/coordinate tags to ensure those clauses remain together
                if finite_count == 1 and tag in start_tags and (word, tag) not in sum_sub_coord_tags:
                    new_new_sentences.append(holder_sentence)
                    holder_sentence = ""
                    finite_count = 0
                    holder_sentence += f"{word} "
                    continue
                holder_sentence += f"{word} "
            new_new_sentences.append(holder_sentence)
        else:
            new_new_sentences.append(s)

    num_sentences = len(new_new_sentences)

    #Calculate score 
    if num_sentences <= 0:
        return 0, new_new_sentences
    elif num_sentences >= 22:
        return 5, new_new_sentences
    else:
        return (num_sentences**2 / 22.0**2) * 5, new_new_sentences



def spelling_mistakes(txt):
    pass

def main():

    #Simply here for testing
    # test = "Most people do not walk to work; instead, they drive or take the train. I love trains and I love brains and I love hating cars I am cool therefore you are cool too"
    # test = "I want to do well I am sad i am happy. i am cool i am not cool he is dumb. After he and I finished my homework, I went to bed and also brushed my teeth."
    # test = "After he and I finished my homework, I went to bed and also brushed my teeth."
    print(num_sentences("that"))

    # getAverageSentCount()

    #Holding here in case its needed for future use
    # "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


if __name__ == "__main__":
    main()
