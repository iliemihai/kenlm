import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import kenlm
import numpy as np
import pandas as pd
import sentencepiece

LMDescriptor = Union[Dict[str, Path], Union[Path, str]]


def get_args():
    parser = argparse.ArgumentParser(
        description="Compute the score of each sentences of a document",

    )
    parser.add_argument("--models", type=str)

    parser.add_argument("--sentences", action="store_true", default=False)
    parser.add_argument(
        "--languages", type=str, help="Ignore doc with another language"
    )
    parser.add_argument("--field", type=str, default=None)
    parser.add_argument("--newline", type=str, default="\n")
    return vars(parser.parse_args())

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

# test SP
model_sp_path = "data/lm_sp/ro.sp.model"
doc = {"text": "A fost odata ca niciodata, si de n-ar fi fost, nu s-ar povesti."}
sp = SentencePiece(model=model_sp_path, field="text")
ret = sp.do(doc)
print(ret)

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

# test SP
model_sp_path = "data/lm_sp/ro.sp.model"
doc = {"text": "A fost odata ca niciodata, si de n-ar fi fost, nu s-ar povesti."}
sp = SentencePiece(model=model_sp_path, field="text")
doc = sp.do(doc)
model_lm_path = "data/lm_sp/ro.arpa.bin"
lm = DocLM(model=model_lm_path, field="tokenized")
ret = lm.do(doc)
print(ret)


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

model_sp_path = "data/lm_sp/ro.sp.model"
doc = {"text": "A fost~~~~odata ~==== ca niciodata, si de.................!!!!!!!!!!!! n-ar fi fost, nu s-ar povesti."}
sp = SentencePiece(model=model_sp_path, field="text")
doc = sp.do(doc)
model_lm_path = "data/lm_sp/ro.arpa.bin"
lm = SentenceLM(model=model_lm_path, field="tokenized")
ret = lm.do(doc)
print(ret)

def perplexity_to_bin(file: Path, output: Path, models, tok_field: str):
    pp_field = "perplexity"
    lm = SentenceLM(models, tok_field, output_field=pp_field)
    stats: List[float] = []
    batch_size = 100000
    i = 0
    batch = []
    in_file =  open(file, "r")
    out_file = open(output, "wb")
    for sentence in in_file.readlines():
        i += 1
        pp = lm(sentence)[pp_field]
        batch.append(pp)
        if len(batch) >= batch_size:
            np.array(batch, dtype=np.float32).tofile(out_file)
            batch = []
    if len(batch) > 0:
        np.array(batch, dtype=np.float32).tofile(out_file)


if __name__ == "__main__":
    args = get_args()
    #output = Path(args["output"])
    #perplexity_to_bin(args["file"], output, args["models"], args["field"])
