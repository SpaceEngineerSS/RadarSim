"""
Interface translations.

English is the source language. ``tr("Text")`` returns the translation for the
active language from ``assets/i18n/<code>.json`` (falling back to the English
text), so a missing entry never breaks the interface. The language is chosen
at start-up from the saved settings.
"""

import json
from functools import cache
from typing import Dict

from .theme import ASSETS

LANGUAGES = {"en": "English", "tr": "Türkçe"}

_language = "en"


@cache
def _catalog(code: str) -> Dict[str, str]:
    path = ASSETS / "i18n" / f"{code}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def set_language(code: str) -> None:
    global _language
    _language = code if code in LANGUAGES else "en"


def language() -> str:
    return _language


def tr(text: str) -> str:
    if _language == "en":
        return text
    return _catalog(_language).get(text, text)
