from helpers.env_helper import get_env_variable


_DEFAULT_POSITIVE_TOKENS = "violent,violence,viol"
_DEFAULT_NEGATIVE_TOKENS = "non-violent,nonviolent,non_violent,non violence,non"


def _split_tokens(value: str) -> list[str]:
    return [token.strip().lower() for token in value.split(",") if token.strip()]


def is_violent_label(label: str) -> bool:
    normalized = label.lower()
    positive_tokens = _split_tokens(get_env_variable("CLASSIFICATION_POSITIVE_LABELS", _DEFAULT_POSITIVE_TOKENS))
    negative_tokens = _split_tokens(get_env_variable("CLASSIFICATION_NEGATIVE_LABELS", _DEFAULT_NEGATIVE_TOKENS))
    if any(token in normalized for token in negative_tokens):
        return False
    return any(token in normalized for token in positive_tokens)

