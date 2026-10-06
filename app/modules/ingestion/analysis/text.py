import re
import unicodedata

_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize(value: str) -> str:
    """Minuscules, sans accents ni ponctuation : « Côte d'Ivoire » devient
    « cote d ivoire ». Sert aux comparaisons multilingues."""
    ascii_value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    )
    return _NON_WORD.sub(" ", ascii_value.lower()).strip()


def contains_phrase(normalized_text: str, phrase: str) -> bool:
    """Recherche d'une expression entière dans un texte déjà normalisé."""
    target = normalize(phrase)
    return bool(target) and f" {target} " in f" {normalized_text} "


def count_phrase(normalized_text: str, phrase: str) -> int:
    target = normalize(phrase)
    if not target:
        return 0
    return f" {normalized_text} ".count(f" {target} ")
