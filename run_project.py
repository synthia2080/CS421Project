import numpy as np
import nltk
from sample_code_1 import num_sentences, spelling_mistakes
from sample_code_2 import verbMistakes, agreement
import argparse
import os

def main():
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", help="Path to essay folder")
    parser.add_argument("--single_essay", help="path to single essay")
    args = parser.parse_args()

    #Errors for arguments
    if args.folder_path and args.single_essay:
        parser.error("Please only use one of the two options: --folder_path or --single_essay")
    if not args.folder_path and not args.single_essay:
        parser.error("Please use at least one of the two options: --folder_path or --single_essay")

    essays = []

    #Read in files based on args
    if args.folder_path:
        for filename in os.listdir(args.folder_path):
            file_path = os.path.join(args.folder_path, filename)

            with open(file_path, 'r') as file:
                text = file.read()
                essays.append((filename, text))

    elif args.single_essay:
        with open(args.single_essay, 'r') as file:
            text = file.read()
            essays.append((os.path.basename(args.single_essay), text))

    #Get scores for all essays
    for (filename, essay) in essays:
        a_score, new_sentences = num_sentences(essay)

        b_score = spelling_mistakes(essay)
        ci_score = agreement(new_sentences)

        cii_score = verbMistakes(new_sentences)

        print(f"{filename}:")
        print(f"    a-score: {a_score}")
        print(f"    b-score: {b_score}")
        print(f"    ci-score: {ci_score}")
        print(f"    cii-score: {cii_score}")
        print()

if __name__ == "__main__":
    main()