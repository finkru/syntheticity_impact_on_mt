import os
import string
import re
from typing import Callable, Literal
import spacy
import tempfile

class Text:
    CLEANING_REGEX = re.compile(r'\[[^\]]*\]')
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

    def preprocess_text(self, preprocess_type: Literal['sentence_per_line', 'word_per_line']) -> None:
        '''
        Process the input text file by cleaning and normalizing it.
        Removes punctuation (except apostrophes), numbers, and bracketed content.
        Writes one word per line to the output file (word_per_line) or one cleaned line (sentence_per_line).
        '''
        if preprocess_type == 'word_per_line':
            os.makedirs(self.SEG_OUTPUT_DIR, exist_ok=True)
            with open(self.text_path, 'r', encoding='utf-8') as infile, \
                 open(f'{self.SEG_OUTPUT_DIR}/{self.language}-preprocessed.txt', 'w', encoding='utf-8') as outfile:
                for line in infile:
                    cleaned_line = (
                        line.strip()
                        .translate(str.maketrans("", "", string.punctuation.replace("'", "")))
                    )
                    cleaned_line = self.CLEANING_REGEX.sub('', cleaned_line)
                    for word in cleaned_line.split():
                        if word:
                            outfile.write(f'{word}\n')
        elif preprocess_type == 'sentence_per_line':
            os.makedirs(self.TRA_OUTPUT_DIR, exist_ok=True)
            with open(self.text_path, 'r', encoding='utf-8') as infile, \
                 open(f'{self.TRA_OUTPUT_DIR}/{self.language}-preprocessed.txt', 'w', encoding='utf-8') as outfile:
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
    def __init__(self, source_Text: Text):
        super().__init__(source_Text.language, source_Text.text_path)
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
                # Write segmented text to a temporary file first for safe writing.
                temp_file = tempfile.NamedTemporaryFile('w', delete=False, dir=self.SEG_OUTPUT_DIR, encoding='utf-8')
                for seg_list in segments:
                    temp_file.write('-'.join(seg_list) + ' ')
                temp_file.close()
                final_path = f'{self.SEG_OUTPUT_DIR}/{self.language}_segmented_text.txt'
                os.rename(temp_file.name, final_path)
                self.segmented_text_path = final_path
            else:
                self.segmented_text_path = None

        self.word_count = words_count
        self.text_syntheticity = segment_count / words_count if words_count > 0 else 0.0
    
    def segment_string_with_CLUZH(self, text: str, cluzh: Callable[[argparse.Namespace], None]) -> str:
        '''
        Segments a provided text string using CLUZH models, without writing permanent files.
        Returns the segmented text as a string.
        '''
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, f'{self.language}-processed.txt')
            output_dir = temp_dir
            # Write the provided text into a temporary input file.
            with open(input_file, 'w', encoding='utf-8') as f:
                for word in text.split():
                    f.write(f'{word}\n')
            
            args = argparse.Namespace(
                model_folder=f'cluzh_segment/morpheme_segmentation/word_level_p1/{self.language}',
                test=input_file,
                output=output_dir,
                features=False,
                nfd=True,
                batch_size=5,
                beam_width=-1,
                device='cpu'
            )
            cluzh(args)
            
            predictions_file = os.path.join(output_dir, 'test_greedy.predictions')
            if os.path.exists(predictions_file):
                with open(predictions_file, 'r', encoding='utf-8') as pf:
                    lines = pf.readlines()
                words_count = len(lines)
                segments = [line.strip().split('\t')[1].split('_') for line in lines if '\t' in line]
                segment_count = sum(len(seg_list) for seg_list in segments)
                segmented_words = ['-'.join(seg_list) for seg_list in segments]
                segmented_string = ' '.join(segmented_words)
                self.word_count = words_count
                self.text_syntheticity = segment_count / words_count if words_count > 0 else 0.0
                return segmented_string
            else:
                raise FileNotFoundError("Predictions file not found.")

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
    def __init__(self, source_Text: Text, target_lang: str):
        super().__init__(source_Text.language, source_Text.text_path)
        self.nlp = source_Text.nlp
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
             tempfile.NamedTemporaryFile('w', delete=False, dir=self.TRA_OUTPUT_DIR, encoding='utf-8') as tmp_outpt:
            for line in inpt:
                if line.strip():
                    tokens = tokenizer(line, return_tensors="pt", padding=True, truncation=True)
                    translated_tokens = model.generate(**tokens)
                    translated_line = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                    tmp_outpt.write(f'{translated_line}\n')
            tmp_outpt_path = tmp_outpt.name
        final_output = f'{self.TRA_OUTPUT_DIR}/translated-{self.language}-{self.target_lang}.txt'
        os.rename(tmp_outpt_path, final_output)
    
    def translate_string_with_opus(self, text: str) -> str:
        '''
        Translates a provided text string using OPUS MT model.
        Returns the translated text as a string.
        '''
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        output_lines = []
        for line in text.splitlines():
            if line.strip():
                tokens = tokenizer(line, return_tensors="pt", padding=True, truncation=True)
                translated_tokens = model.generate(**tokens)
                translated_line = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                output_lines.append(translated_line)
            else:
                output_lines.append("")
        return "\n".join(output_lines)


from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import corpus_chrf, sentence_chrf
import pandas as pd

class TextEvaluator(Text):
    def __init__(self, source_Text: Text, cluzh: Callable[[argparse.Namespace], None], target_lang: str):
        super().__init__(source_Text.language, source_Text.text_path)
        self.source_Text = source_Text
        self.cluzh = cluzh
        self.target_lang = target_lang

    def aggregate(self) -> pd.DataFrame:
        '''
        Evaluates the text by reading a preprocessed sentence file,
        segmenting each sentence, translating it, computing syntheticity,
        and the character F-score (chrf) between the original and translated sentences.
        Returns a DataFrame with columns:
        "segmented original sentence", "translated sentence", "sytheticity", "chrf_score".
        '''
        if hasattr(self, "preprocessed_file") and self.preprocessed_file:
            preprocessed_file = self.preprocessed_file
        else:
            preprocessed_file = os.path.join(self.TRA_OUTPUT_DIR, f"{self.language}-preprocessed.txt")
        
        with open(preprocessed_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        results = []

        segmenter = TextSegmenter(self.source_Text)
        translator = TextTranslator(self.source_Text, self.target_lang)
        
        for sentence in sentences:
            
            segmented = segmenter.segment_string_with_CLUZH(sentence, self.cluzh)
            
            translated = translator.translate_string_with_opus(sentence)
            
            
            words = segmented.split()
            total_segments = sum(len(word.split('-')) for word in words)
            syntheticity = total_segments / len(words) if words else 0.0
            
            
            
            results.append({
                "segmented original sentence": segmented,
                "original sentence": sentence,
                "translated sentence": translated,
                "sytheticity": syntheticity
            })
        
        return pd.DataFrame(results)
    
    def evaluate(self):
        '''
        Goes through the aggregated DataFrame sentence by sentence, compares each translated sentence
        with the corresponding reference sentence (from data/test/test.txt), after removing punctuation
        (except apostrophes), and computes a sentence-level BLEU and CHRF score.   
        The scores are then appended as new columns to the aggregated DataFrame and the updated DataFrame is returned.
        '''
        # Get the aggregated DataFrame.
        df = self.aggregate()
        
        # Read reference sentences (assumed line-by-line) from the test file.
        # Adjust the test file path if needed.
        test_file = 'data/targets/target.txt'
        with open(test_file, 'r', encoding='utf-8') as f:
            ref_sentences = [line.strip() for line in f if line.strip() and not line.strip().startswith('//')]
        
        # Remove punctuation except apostrophes.
        keep = "'"
        punctuation_to_remove = ''.join(ch for ch in string.punctuation if ch not in keep)
        pattern = re.compile(f'[{re.escape(punctuation_to_remove)}]')
        def remove_punct(text: str) -> str:
            return pattern.sub('', text)
        
        # Preprocess candidate and reference sentences.
        candidate_sentences = [remove_punct(row["translated sentence"]) for _, row in df.iterrows()]
        ref_sentences = [remove_punct(sent) for sent in ref_sentences]
        
        bleu_scores = []
        chrf_scores = []
        # Use smoothing for sentence_bleu to avoid zero scores in short sentences.
        smooth_fn = SmoothingFunction().method1
        
        # Iterate sentence by sentence.
        for i, cand in enumerate(candidate_sentences):
            # If there are fewer reference sentences than candidates, use an empty string as fallback.
            ref = ref_sentences[i] if i < len(ref_sentences) else ""
            
            # Tokenize candidate and reference (simple whitespace split)
            cand_tokens = cand.split()
            ref_tokens = ref.split()
            
            # Compute sentence-level BLEU score.
            bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smooth_fn) if cand_tokens and ref_tokens else 0.0
            # Compute sentence-level CHRF score.
            chrf = sentence_chrf(ref, cand) if ref and cand else 0.0
            
            bleu_scores.append(bleu)
            chrf_scores.append(chrf)
        
        # Append computed scores to the DataFrame.
        df['bleu'] = bleu_scores
        df['chrf_score'] = chrf_scores
        df['targets'] = ref_sentences
        
        return df
    
    def evaluate_with_comet(
        self,
        model_path: str = "wmt20-comet-qe-da/checkpoints/model.ckpt",
        batch_size: int = 8,
        gpus: int | None = 0,
    ) -> pd.DataFrame:
        
        from comet import download_model, load_from_checkpoint

        model_path = download_model("Unbabel/wmt20-comet-qe-da")
        model = load_from_checkpoint(model_path)

        
        df = self.evaluate()      # <-- gives us “targets” column with refs

        
        data = [
            {
                "src": row["original sentence"],
                "mt":  row["translated sentence"],
                "ref": row["targets"],           # reference from evaluate()
            }
            for _, row in df.iterrows()
        ]

        
        scores = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=False)

        
        df["comet_score"] = scores['scores']
        return df

