# CS421Project 
Teammate 1: Synthia Sasulski netid: lsasu2

Teammate 2: Sravani Bhamidipaty netid: sbham3

[link to github](https://github.com/synthia2080/CS421Project.git)

## How to Run
First, uncompress w2v.pkl.zip if it's not already and leave it in root directory (Had to be compressed due to git size limitations)(Same w2v.pkl as in hw3)

python run_project.py

This assumes a folder hierarchy of:
<br>
/CS421PROJECT
<br>- essays
<br>&nbsp;&nbsp;&nbsp;&nbsp;- 1234.txt
<br>&nbsp;&nbsp;&nbsp;&nbsp;- ...
<br>- index.csv
<br>- run_project.py
<br>- w2v.pkl
<br>- ...
<br>

## Example Output:
*Some functions to take long so do not worry <br><br>
52951.txt:
<br>&nbsp;&nbsp;&nbsp;&nbsp;a-score: 3.347107438016529
<br>&nbsp;&nbsp;&nbsp;&nbsp;b-score: 4
<br>&nbsp;&nbsp;&nbsp;&nbsp;ci-score: 5
<br>&nbsp;&nbsp;&nbsp;&nbsp;cii-score: 5
<br>&nbsp;&nbsp;&nbsp;&nbsp;ciii-score: 3.304017821503724
<br>&nbsp;&nbsp;&nbsp;&nbsp;di-score: 1
<br>&nbsp;&nbsp;&nbsp;&nbsp;dii-score: 1
<br>&nbsp;&nbsp;&nbsp;&nbsp;Final Score: 23.302250519040506
<br>&nbsp;&nbsp;&nbsp;&nbsp;Final grade: high

## Packages Used
- Numpy
- nltk
- SpaCY
- pandas
- os
- Spellchecker
- pickle

## Functions/Explanations for scoring

*The prompts are simply read through the index.csv file, using a pandas dataframe to find the row of the filename that is currently being read and getting the prompt.

### a-score (num_sentences(), sample_code_1.py)
There were 3 steps in determining where sentences start/end.
1. use nltk sentence tokenizer as a base since its relatively accurate and a good starting off point
2. Get "sub-sentences" based off capitalization. This is done through going through the words and respective POS tags, and checking
if the word is capitlaized. If the word is, and its not a proper noun or part of a manually created list of POS tags that usually can't be at the end of a sentence, we can then assume its a new sentence.
3. Get "sub-sentences" based off POS tag exploitation. For all of the newly created and previous sentences, check for finite verbs. If there are more than one and there are more finite verbs than subordinate/coodinate clause tags, it should indicate multiple sub-sentences, otherwise if theres only 1 finite verb, we're done since its most likely just one sentence. If the first is true, then we look where the finite verbs and subordinate/coordinate clauses are. Then, if one of the tokens dependency is marked as "mark", it indicates a subordinate/coordinate clause so simply append the sentence, otherwise we go until a finite verb is found, continue until a token is found after that can possbily mark the end of  of sentence, if it can then we found another sub-sentence. Simply continue for the sentence until we passed the final finite verb.

The scoring was calculated through simply getting the average number of sentences for high/low essays and using interpolation to get a score from 0-5 based on the manually found averages.

### b-score (spelling_mistakes(), sample_code_1.py)
There are 3 steps to look for spelling mistakes.
1. We utilize a spell checking library: pyspellchecker
2. Then we take the input text and then tokenize the text into individual words using word_tokenize function which is imported from the nltk library.
3. We utilize unknown function from the SpellChecker library to see if there are any spelling mistakes in the words array created from the previous step.

The scoring was calculated through simply getting the average number of misspelled words for high/low essays and using interpolation to get a score from 0-5 based on the manually found averages.

### ci-score (agreement(), sample_code_2.py)
There are a few checks to look for subject-verb agreement.
After passing in the newly tokenized sentences from num_sentences(), we check through the POS tags again. First, we iterate through each tokenized sentence to identify the subject using POS tags to determine whether the subject is singular of plural. Then for each sentence, we consider two major verb groups: is the verb auxiliary or not. If the verb is not auxiliary: it is not in the auxiliary verbs list (we have only considered the verb has and do since this are the most common ones, but we can add other common ones), then we ensure that its form matches the subject. There are two cases, for singular subjects ("he", "she", "it"), the verb should not be in plural form. For singular subjects ("I", "you") and plural subjects ("we", "they"), the verb should not be in singular form. This is determined using POS tags for verbs. If there is an error, the error counter increments by 1.

The scoring for the agreement function is determined by comparing the frequency of grammatical errors related to subject-verb agreement and tense consistency within the provided tokenized sentences and using interpolation to get a score from 0-5 based on the manually found averages of agreement errors from the high/low essays.

### cii-score (verbMistakes(), sample_code_2.py)
There are a few checks to look for verb mistakes.
After passing in the newly tokenized sentences from num_sentences(), we check through the POS tags again. First, we check for a correct verb tense following infinitive. Then we check for missing auxilary verbs through looking at the word's children's dependency. After all that, and counting the number of finite verbs, we check to see if theres only 1 finite verb (from the previous missing verb checks we also checked for the root verb). Once these checks are done, we check for discrepencies between root verb tenses in consecutive sentences.

The scoring was done similarly to the scoring in num_sentences(), however, the number of mistakes as well as the number of verb tense changes between sentences are normalized. This is to account for longer essays possibly having more mistakes simply because theres more room for error. Normalizing the verb tense changes also helps account for correct tense changes between sentences, which was proven difficult to determine reliably.

### ciii-score (syntacticWellFormedness(), sample_code_3.py)
There are a few checks to look for sentence formation.
After passing in the sentence, it is parsed through the NLTK's CoreNLPParser. Then it checks for 3 conditions:
1. The sentence should not start with a verb, or the sentence should start with an auxiliary or with a wh-word. If any of these conditions fail based on pos tagging, the count for mistakes increments by 1.
2.  For each subtree, the program iterates through its children, and then checks if it is missing any determiners, prepositional phrases, and missing prepositions. If there are any missing words or consitituents based on the pos tagging then the count for mistakes increments by 1.
3. For subordinate clauses, the program checks if there is a missing main verb or subordinating conjuction, and then the count for mistakes increments by 1. 

The scoring for the syntacticWellFormedness function involves analyzing the syntactic structure and grammatical correctness of each sentence within the tokenized sentences and using interpolation to get a score from 0-5 based on the manually found averages of syntactic structure errors from the high/low essays.

### di-score/dii-score (semanticsPragmatics(), sample_code_4.py)
This function, semanticsPragmatics(), handles both the di and dii score.
It takes in the tokenized sentences and the corresponding prompt as parameters, gets the embedding for each sentence/the prompt, for words in w2v that are content, non-stop, and non-punct. Then it gets the average for the sentence and stores it in a list.
From there we use the averages to get the cosine similarity between each sentence averages and the prompt, then getting the average similarty for the essay to map into a di score from high/low thresholds we got after seperate experimentation.

For dii score we run through the averaged sentence embeddings, and simply checking the cosine similary between the current sentence and the one in front, except for the final sentence. Then we calculate the standard deviation, and after finding the average deviations between high/low grades in a seperate experiment, we map those thresholds to get our final dii score. 

