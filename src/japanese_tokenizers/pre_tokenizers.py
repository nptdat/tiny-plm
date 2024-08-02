# Copyright 2020 The HuggingFace Inc. team.
# Copyright 2021 Masatoshi Suzuki (@singletongue).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copied from https://github.com/cl-tohoku/bert-japanese/blob/main/japanese_tokenizers/pre_tokenizers.py

import os
from typing import List, Optional, Tuple

from tokenizers import NormalizedString, PreTokenizedString

Offsets = Tuple[int, int]


class MeCabPreTokenizer:
    def __init__(
        self,
        mecab_dic: Optional[str] = None,
        mecab_option: Optional[str] = None,
    ) -> None:
        import fugashi

        mecab_option = mecab_option or ""

        if mecab_dic is not None:
            if mecab_dic == "unidic_lite":
                import unidic_lite

                dic_dir = unidic_lite.DICDIR
            elif mecab_dic == "unidic":
                import unidic

                dic_dir = unidic.DICDIR
            elif mecab_dic == "ipadic":
                import ipadic

                dic_dir = ipadic.DICDIR
            else:
                raise ValueError("Invalid mecab_dic is specified.")

            mecabrc = os.path.join(dic_dir, "mecabrc")
            mecab_option = (
                "-d {} -r {} ".format(dic_dir, mecabrc) + mecab_option
            )

        self.mecab = fugashi.GenericTagger(mecab_option)

    def mecab_split(
        self, i: int, text: NormalizedString
    ) -> List[NormalizedString]:
        tokens = []
        cursor = 0
        for word in self.mecab(str(text)):
            token = word.surface
            start = str(text).index(token, cursor)
            end = start + len(token)

            tokens.append(text[start:end])
            cursor = end

        return tokens

    def pre_tokenize(self, text: PreTokenizedString) -> None:
        text.split(self.mecab_split)
