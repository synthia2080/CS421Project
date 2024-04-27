import numpy as np
import nltk
from sample_code_1 import num_sentences, spelling_mistakes
from sample_code_2 import verbMistakes, agreement
from sample_code_3 import syntacticWellFormedness
from sample_code_4 import semanticsPragmatics
import argparse
import os
import pandas as pd

def main():
    # #parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--folder_path", help="Path to essay folder")
    # parser.add_argument("--single_essay", help="path to single essay")
    # args = parser.parse_args()

    # #Errors for arguments
    # if args.folder_path and args.single_essay:
    #     parser.error("Please only use one of the two options: --folder_path or --single_essay")
    # if not args.folder_path and not args.single_essay:
    #     parser.error("Please use at least one of the two options: --folder_path or --single_essay")

    essays = []

    essays_path = "essays"
    for filename in os.listdir(essays_path):
        file_path = os.path.join(essays_path, filename)

        with open(file_path, 'r') as file:
            text = file.read()
            essays.append((filename, text))

    #Get scores for all essays
    high_sentence_num = 0
    high_sentence_totalCosineSim = 0
    low_sentence_num = 0
    low_sentence_totalCosineSim = 0
    index = pd.read_csv("index.csv", delimiter=";")
    for (filename, essay) in essays:
        df = index[index['filename'] == filename]
        prompt = df['prompt'].iloc[0]
        human_score = df['grade'].iloc[0]

        a_score, new_sentences = num_sentences(essay)

        b_score = spelling_mistakes(essay)
        ci_score = agreement(new_sentences)
        cii_score = verbMistakes(new_sentences)
        ciii_score = syntacticWellFormedness(new_sentences)
        di_score, dii_score = semanticsPragmatics(prompt, new_sentences)
        final_score = 2 * a_score - b_score + ci_score + cii_score + 2 * ciii_score + 3 * di_score + dii_score

        highThresh = 23.1

        if final_score >= highThresh:
            final_grade = 'high'
        else:
            final_grade = 'low'
        
        print(f"{filename}:")
        print(f"    a-score: {a_score}")
        print(f"    b-score: {b_score}")
        print(f"    ci-score: {ci_score}")
        print(f"    cii-score: {cii_score}")
        print(f"    ciii-score: {ciii_score}")
        print(f"    di-score: {di_score}")
        print(f"    dii-score: {dii_score}")
        print(f"    Final Score: {final_score}")
        print(f"    Final grade: {final_grade}\n")
        
        if human_score == "high":
            high_sentence_num += 1
            high_sentence_totalCosineSim += final_score
        else:
            low_sentence_num += 1
            low_sentence_totalCosineSim += final_score

    correctPrompt_high_average = high_sentence_totalCosineSim / high_sentence_num
    correctPrompt_low_average = low_sentence_totalCosineSim / low_sentence_num

    print(f"Average final score for 'high' grades: {correctPrompt_high_average}")
    print(f"Average final score for 'low' grades: {correctPrompt_low_average}")

if __name__ == "__main__":
    main()