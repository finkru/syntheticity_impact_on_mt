import os
import string
import re
from typing import Callable, Literal
import spacy


class Text:
    CLEANING_REGEX = re.compile(r'\[[^\]]*\]|\d+')
    SEG_OUTPUT_DIR = 'data/seg_output'
    TRA_OUTPUT_DIR = 'data/tra_output'

    def __init__(self, language: str, text_path: str):
        self.text_path = text_path
        self.language = language
        self.nlp = self.load_spacy_model()

    def load_spacy_model(self):
        '''
        Loads the appropriate spaCy model based on the language.
        First tries news model, falls back to web model if needed.
        '''
        spacy_lang = self.language[:2]
        try:
            return spacy.load(f'{spacy_lang}_core_news_sm')
        except OSError:
            try:
                print(f"News model not found. Attempting to load web model for {spacy_lang}...")
                return spacy.load(f'{spacy_lang}_core_web_sm')
            except OSError:
                print(f"No spaCy models found for language: {spacy_lang}. Please download using:")
                print(f"python -m spacy download {spacy_lang}_core_web_sm")
                return None

    def preprocess_text(self, preprocess_type: Literal['for_translating', 'for_segmenting']) -> None:
        '''
        Process the input text file by cleaning and normalizing it.
        Removes punctuation (except apostrophes), numbers, and bracketed content.
        Writes one word per line to the output file (for_segmenting) or one cleaned line (for_translating).
        '''
        if preprocess_type == 'for_segmenting':
            os.makedirs(self.SEG_OUTPUT_DIR, exist_ok=True)
            with open(self.text_path, 'r', encoding='utf-8') as infile, \
                 open(f'{self.SEG_OUTPUT_DIR}/{self.language}-processed.txt', 'w', encoding='utf-8') as outfile:
                for line in infile:
                    cleaned_line = (
                        line.strip()
                        .translate(str.maketrans("", "", string.punctuation.replace("'", "")))
                    )
                    cleaned_line = self.CLEANING_REGEX.sub('', cleaned_line)
                    for word in cleaned_line.split():
                        if word:
                            outfile.write(f'{word}\n')
        elif preprocess_type == 'for_translating':
            os.makedirs(self.TRA_OUTPUT_DIR, exist_ok=True)
            with open(self.text_path, 'r', encoding='utf-8') as infile, \
                 open(f'{self.TRA_OUTPUT_DIR}/{self.language}-processed.txt', 'w', encoding='utf-8') as outfile:
                for line in infile:
                    doc = self.nlp(line)
                    for sentence in doc.sents:
                        cleaned_sent = (
                            sentence.text.strip()
                            .translate(str.maketrans("", "", string.punctuation.replace("'", "")))
                        )
                        cleaned_sent = self.CLEANING_REGEX.sub('', cleaned_sent)
                        if cleaned_sent:
                            outfile.write(f'{cleaned_sent}\n')


import argparse


class TextSegmenter(Text):
    def __init__(self, source_text: Text):
        super().__init__(source_text.language, source_text.text_path)
        self.word_count = None
        self.text_syntheticity = None
        self.segmented_text_path = None

    def segment_with_CLUZH(self, cluzh: Callable[[argparse.Namespace], None], save_segments: bool = None) -> None:
        '''
        Segments a processed text using CLUZH models.

        Takes a processed text as input, segments it using CLUZH models trained for the SIGMORPHON 2022 shared tasks,
        and writes one word and its segments per line to the output file (test_greedy.predictions).
        Updates the instance with the number of words and the syntheticity index.
        If save_segments=True, saves the path to the segmented text file.
        '''
        args = argparse.Namespace(
            model_folder=f'cluzh_segment/morpheme_segmentation/word_level_p1/{self.language}',
            test=f'{self.SEG_OUTPUT_DIR}/{self.language}-processed.txt',
            output= self.SEG_OUTPUT_DIR,
            features=False,
            nfd=True,
            batch_size=5,
            beam_width=-1,
            device='cpu'
        )
        cluzh(args)

        with open(f'{self.SEG_OUTPUT_DIR}/test_greedy.predictions', 'r') as inpt:
            lines = inpt.readlines()
            words_count = len(lines)

            segments = [line.strip().split('\t')[1].split('_') for line in lines]
            segment_count = sum(len(seg_list) for seg_list in segments)

            if save_segments:
                segmented_path = f'{self.SEG_OUTPUT_DIR}/{self.language}_segmented_text.txt'
                with open(segmented_path, 'w') as outpt:
                    for seg_list in segments:
                        outpt.write('-'.join(seg_list) + ' ')
                self.segmented_text_path = segmented_path
            else:
                self.segmented_text_path = None

        self.word_count = words_count
        self.text_syntheticity = segment_count / words_count if words_count > 0 else 0.0

    def describe(self) -> None:
        print(f'This text has {self.word_count} words and its syntheticity is {self.text_syntheticity} segments per word')

    def __str__(self) -> str:
        if self.segmented_text_path and os.path.exists(self.segmented_text_path):
            with open(self.segmented_text_path, 'r') as f:
                return f.read()
        else:
            return f"No segmented text available for {self.language}."


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TextTranslator(Text):
    '''A class for translating text files, inheriting from Text class.'''
    def __init__(self, source_text: Text, target_lang: str):
        super().__init__(source_text.language, source_text.text_path)
        self.nlp = source_text.nlp
        self.target_lang = target_lang
        self.model_name = f"Helsinki-NLP/opus-mt-{self.language.lower()[:2]}-{target_lang.lower()[:2]}"

    def translate_with_opus(self) -> None:
        '''
        Translates preprocessed text using OPUS MT model.
        Creates a new file with translations in the TRA_OUTPUT_DIR.
        '''
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        with open(f'{self.TRA_OUTPUT_DIR}/{self.language}-processed.txt', 'r', encoding='utf-8') as inpt, \
                open(f'{self.TRA_OUTPUT_DIR}/translated-{self.language}-{self.target_lang}.txt', 'w', encoding='utf-8') as outpt:
            for line in inpt:
                if line.strip():
                    tokens = tokenizer(line, return_tensors="pt", padding=True, truncation=True)
                    translated_tokens = model.generate(**tokens)
                    translated_line = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                    outpt.write(f'{translated_line}\n')


from nltk.translate.chrf_score import corpus_chrf
import pandas as pd