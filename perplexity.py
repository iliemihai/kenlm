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
                                                                                                                                                                                         1,1           Top
