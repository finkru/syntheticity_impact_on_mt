import string
import re
import typing 
import argparse


class TextWithSegments:
    CLEANING_PATTERN = re.compile(r'\[[^\]]*\]|\d+')

    def __init__(self, text_path: str, language: str):
        self.text_path = text_path
        self.language = language
        self.word_count = None
        self.text_syntheticity = None

    def process_text(self) -> None:
        """
        Process the input text by cleaning and normalizing it.
        Removes punctuation (except apostrophes), numbers, and bracketed content.
        Writes one word per line to the output file.
        """
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
                        outpt.write(f"{word}\n")

    def segment_with_CLUZH(self, cluzh) -> None:
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
        with open(f'data/seg_output/test_greedy.predictions', 'r') as f:
            lines = f.readlines()
            words_count = len(lines)
            segmentations_count = 0
            for line in lines:
                segmentations_count += len(line.strip().split('\t')[1].split('_'))
        
        self.word_count = words_count
        self.text_syntheticity = segmentations_count / words_count

    def __str__(self):
        return f'This text has {self.word_count} and its syntheticity is {self.text_syntheticity} segments per word'

