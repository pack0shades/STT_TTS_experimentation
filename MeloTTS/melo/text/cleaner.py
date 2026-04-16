"""Text normalization + G2P entry points.

This file is imported by `melo.api`.

Important: keep imports *lightweight*.
Some language modules (e.g. `chinese_mix`) may pull heavyweight dependencies
and/or trigger model downloads at import time. For common EN-only usage we want
to avoid that and import only the requested language module.
"""

from __future__ import annotations

import copy
import importlib
from typing import Dict

from . import cleaned_text_to_sequence


# Map language codes -> module basename under `melo.text`.
_LANG_TO_MOD: Dict[str, str] = {
    "ZH": "chinese",
    "JP": "japanese",
    "EN": "english",
    "ZH_MIX_EN": "chinese_mix",
    "KR": "korean",
    "FR": "french",
    "SP": "spanish",
    "ES": "spanish",
}

_LANG_MODULE_CACHE: Dict[str, object] = {}


def _get_language_module(language: str):
    language = str(language)
    if language in _LANG_MODULE_CACHE:
        return _LANG_MODULE_CACHE[language]

    mod_name = _LANG_TO_MOD.get(language)
    if not mod_name:
        raise KeyError(f"Unsupported language code: {language!r}")

    # This module lives at `melo.text.cleaner`, so `__package__` is `melo.text`.
    mod = importlib.import_module(f"{__package__}.{mod_name}")
    _LANG_MODULE_CACHE[language] = mod
    return mod


def clean_text(text, language):
    language_module = _get_language_module(language)
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language, device=None):
    language_module = _get_language_module(language)
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)

    word2ph_bak = copy.deepcopy(word2ph)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = language_module.get_bert_feature(norm_text, word2ph, device=device)

    return norm_text, phones, tones, word2ph_bak, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
