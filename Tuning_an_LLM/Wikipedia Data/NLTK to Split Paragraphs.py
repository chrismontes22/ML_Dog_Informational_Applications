"""This script takes a text file as input and splits long paragraphs into shorter ones based on a word count threshold of 150 words. 
It uses the NLTK punkt tokenizer to detect sentence boundaries and recombines sentences into new paragraphs that meet the word count limit. 
Used to make them fit into trainable data length. Iterates through all text files in a folder"""
#I used this after combining all of the text

import nltk
import re

def split_paragraphs(file_path):
    # Load the punkt tokenizer
    nltk.download('punkt', quiet=True)
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the text into paragraphs
    paragraphs = text.split('\n')

    new_paragraphs = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > 150:
            # Split the paragraph into sentences
            sentences = sent_detector.tokenize(paragraph.strip())
            new_paragraph = ''
            word_count = 0
            for sentence in sentences:
                word_count += len(sentence.split())
                if word_count > 150:
                    new_paragraphs.append(new_paragraph.strip())
                    new_paragraph = sentence
                    word_count = len(sentence.split())
                else:
                    new_paragraph += ' ' + sentence
            new_paragraphs.append(new_paragraph.strip())
        else:
            new_paragraphs.append(paragraph)

    # Write the new paragraphs back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(new_paragraphs))

# Replace 'your_file.txt' with the path to your file
split_paragraphs('your_file.txt')
