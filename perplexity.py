import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import kenlm
import numpy as np
import pandas as pd
import sentencepiece


def pp(log_score, length):
    return 10.0 ** (-log_score / length)

class SentencePiece:
    def __init__(self,
                model: Path,
                field: str,
                output_field: str="tokenized",
                ):
        super().__init__()
        self.model = model
        self.field = field
        self.output_field = output_field
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(str(self.model))

    def do(self, document: dict) -> dict:
        text = document[self.field]
        tokenized = self.sp.encode_as_pieces(text)
        document[self.output_field] = " ".join(tokenized)
        return document

class DocLM:
    def __init__(self,
        model: Path,
        field: str,
        output_field: str = "perplexity",
        newline: str = "\n",
        load_method: int = 2,
        ):

        super().__init__()
        self.field = field
        self.output_field = output_field
        self.newline = newline
        self._prefetch: Sequence[str] = []
        self.lm_config = kenlm.Config()
        self.lm_config.load_method = load_method
        start_load = time.time()
        print(f"Loading model...")
        self.lm =  kenlm.Model(str(model), self.lm_config)
        load_time = time.time() - start_load
        print(f"Loaded {self.lm} (took {load_time / 60:.1f}min)")
        self.n_lines = 0

    def get_lines(self, document: dict) -> List[str]:
        content = document.get(self.field)

        if not content:
            return []

        lines = content.split(self.newline)
        self.n_lines = len(lines)
        return lines


    def do(self, document: dict) -> dict:
        lines = self.get_lines(document)
        model = self.lm
        if not lines or not model:
            return document

        doc_log_score, doc_length = 0, 0
        for line in lines:
            log_score = model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        document[self.output_field] = pp(doc_log_score, doc_length) #round(pp(doc_log_score, doc_length), 1)
        return document

    def summary(self):
        delay = time.time() - self.start_time
        h = delay / 3600
        s = self.n_lines / delay

        summ = []
        summ.append(f"Processed {self.n_lines:_} lines in {h:.2}h ({s:.1} lines/s).")
        return summ

class SentenceLM(DocLM):
    """ Returns score for each individual sentence in paragraphs"""
    def do(self, document: dict) -> Optional[str]:
        lines = self.get_lines(document)
        model = self.lm
        if not lines or not model:
            return None

        sentences = []
        for line in lines:
            log_score = model.score(line)
            length = len(line.split()) + 1

            sentences.append(f"{pp(log_score, length)}\t{line}")
        document[self.output_field] = pp(log_score, length)

        return document#"\n".join(sentences)
rng = default_rng(42)

def should_keep_sentence_gaussian(sentence, perplexity, factor=0.78, boundaries=None):
    if boundaries is None:
        boundaries = [536394.99320948, 662247.50212365, 919250.87225178]
    if perplexity <= boundaries[0]:
        quartile_range = boundaries[0]
    elif boundaries[0] < perplexity < boundaries[1]:
        quartile_range = boundaries[1] - boundaries[0]
    elif boundaries[1] < perplexity < boundaries[2]:
        quartile_range = boundaries[2] - boundaries[1]
    elif perplexity >= boundaries[2]:
        quartile_range = 10 * boundaries[2]
    probability = factor / quartile_range

    return rng.uniform() < probability

def perplexity_to_bin(file: Path, output: Path, model_lm: Path, model_sp: Path, keep_jsons: bool=False):
    tok_field="tokenized"
    pp_field = "perplexity"
    sp = SentencePiece(model=model_sp, field="text")
    lm = SentenceLM(model=model_lm, field=tok_field)
    batch_size = 100000
    i = 0
    batch = []
    json_batch = []
    in_file =  open(file, "r")
    out_file = open(output, "wb")
    for sentence in tqdm(in_file.readlines()):
        if sentence != "\n":
            i += 1
            dic_sentence = {"text": sentence}
            dic_sentence = sp.do(dic_sentence)
            pp = lm.do(dic_sentence)
            if keep_jsons:
                pp.pop('tokenized', None)
                val = should_keep_sentence_gaussian(sentence, pp[pp_field])
                ppp = pp
                ppp["keep"] = val
                json_batch.append(ppp)
            batch.append(pp[pp_field])
            if len(batch) >= batch_size:
                np.array(batch, dtype=np.float32).tofile(out_file)
                batch = []
        if len(batch) > 0:
            np.array(batch, dtype=np.float32).tofile(out_file)
    if keep_jsons:
        with open('jsons_files.json', 'w') as fout:
            json.dump(json_batch, fout,  sort_keys=True, indent=4)





if __name__ == "__main__":
    model_sp_path = "data/lm_sp/ro.sp.model"
    model_lm_path = "data/lm_sp/ro.arpa.bin"
    perplexity_to_bin("in_file.txt", "out_file.bin", model_lm=model_lm_path, model_sp=model_sp_path)
