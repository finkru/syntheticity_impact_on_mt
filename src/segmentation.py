import os
import string
import re
import typing
from typing import Callable
import argparse


class TextWithSegments:
    CLEANING_PATTERN = re.compile(r'\[[^\]]*\]|\d+')

    def __init__(self, text_path: str, language: str):
        self.text_path = text_path
        self.language = language
        self.word_count = None
        self.text_syntheticity = None
        self.segmented_text_path = None

    def process_text(self) -> None:
        '''
        Process the input text file by cleaning and normalizing it.
        Removes punctuation (except apostrophes), numbers, and bracketed content.
        Writes one word per line to the output file.
        '''
        with open(self.text_path, 'r', encoding='utf-8') as inpt, \
             open(f'data/seg_output/{self.language}-processed.txt', 'w', encoding='utf-8') as outpt:
            for line in inpt:
                cleaned_line = (
                    line.strip()
                    .translate(str.maketrans("", "", string.punctuation.replace("'", "")))
                )
                cleaned_line = self.CLEANING_PATTERN.sub('', cleaned_line)
                for word in cleaned_line.split():
                    if word:
                        outpt.write(f'{word}\n')

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
            test=f'data/seg_output/{self.language}-processed.txt',
            output='data/seg_output',
            features=False,
            nfd=True,
            batch_size=5,
            beam_width=-1,
            device='cpu'
        )
        cluzh(args)

        with open('data/seg_output/test_greedy.predictions', 'r') as inpt:
            lines = inpt.readlines()
            words_count = len(lines)

            segments = [line.strip().split('\t')[1].split('_') for line in lines]
            segments_count = sum(len(seg_list) for seg_list in segments)

            if save_segments:
                segmented_path = f'data/seg_output/{self.language}_segmented_text.txt'
                with open(segmented_path, 'w') as outpt:
                    for seg_list in segments:
                        outpt.write('-'.join(seg_list) + ' ')
                self.segmented_text_path = segmented_path
            else:
                self.segmented_text_path = None

        self.word_count = words_count
        self.text_syntheticity = segments_count / words_count if words_count > 0 else 0.0

    def describe(self) -> None:
        print(f'This text has {self.word_count} and its syntheticity is {self.text_syntheticity} segments per word')

    def __str__(self) -> str:
        if self.segmented_text_path and os.path.exists(self.segmented_text_path):
            with open(self.segmented_text_path, 'r') as f:
                return f.read()
        else:
            return f"No segmented text available for {self.language}."



