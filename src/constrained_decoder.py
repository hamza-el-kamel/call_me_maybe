from typing import Callable, Optional, TypedDict
from models import FunctionDefinition


class Grammar(TypedDict):
    """
    Define the structure of the function-calling grammar.
    """

    root: list[str]
    function_name: list[str]
    parameters: dict[str, dict[str, dict[str, str]]]


def build_grammar(
        functions: list[FunctionDefinition],
        ) -> Grammar:
    """
    Build the grammar
    """
    grammar: Grammar = {
        "root": [
            "{",
            '"prompt"',
            ":",
            "string",
            ",",
            '"name"',
            ":",
            "function_name",
            ",",
            '"parameters"',
            ":",
            "parameters",
            "}"
        ],

        "function_name": [
            f'"{function.name}"'
            for function in functions
        ],

        "parameters": {
            function.name: function.parameters
            for function in functions
        }
    }

    return grammar


def token_is_allowed(
        partial: str,
        token_text: str,
        allowed_names: list[str]
        ) -> bool:
    """
    Check whether a token can continue a valid function name.
    """
    candidate = partial + token_text

    return any(
        name.startswith(candidate)
        for name in allowed_names
    )


def is_function_name_complete(partial: str, allowed_names: list[str]) -> bool:
    """
    Check whether the partial text is a complete function name.
    """
    return partial in allowed_names


def token_is_allowed_number(partial: str, token_text: str) -> bool:
    """
    Check if token a valid number
    """
    candidate = partial + token_text

    if not candidate:
        return False

    if "-" in candidate:
        if candidate[0] != "-":
            return False

        if candidate.count("-") > 1:
            return False

    if candidate.count(".") > 1:
        return False

    if "." in candidate:
        position = candidate.find(".")
        before_dot = candidate[:position]

        if before_dot == "" or before_dot == "-":
            return False

    for char in candidate:
        if not (char.isdigit() or char in ".-"):
            return False

    return True


def is_number_complete(partial: str) -> bool:
    """
    Check partial text is a complete valid number.
    """
    if not partial:
        return False

    if partial == "-":
        return False

    if partial.endswith("."):
        return False

    try:
        float(partial)
        return True
    except ValueError:
        return False


def token_is_allowed_string(partial: str, token_text: str) -> bool:
    """
    Check the partial text is a allowd token.
    """
    candidate = partial + token_text

    if not candidate.startswith('"'):
        return False

    if candidate.count('"') > 2:
        return False

    if candidate.count('"') == 2:
        return candidate.endswith('"')

    return True


def is_string_complete(partial: str) -> bool:
    """
    Check the partial text is a complete valid string
    """
    return (
        len(partial) >= 2
        and partial.startswith('"')
        and partial.endswith('"')
        and partial.count('"') == 2
    )


def token_is_allowed_boolean(partial: str, token_text: str) -> bool:
    """
    Check a token can continue a valid boolean value.
    """
    candidate = partial + token_text

    allowed_values = [
        "true",
        "false"
    ]

    return any(
        value.startswith(candidate)
        for value in allowed_values
    )


def is_boolean_complete(partial: str) -> bool:
    """
    Check a partial text is a complete boolean value.
    """
    return partial in ("true", "false")


def get_value_checker(
        parameter_type: str
        ) -> Optional[Callable[[str, str], bool]]:
    """
    Return the token checker for a parameter type.
    """

    if parameter_type in ("float", "number", "int"):
        return token_is_allowed_number

    if parameter_type in ("str", "string"):
        return token_is_allowed_string

    if parameter_type in ("bool", "boolean"):
        return token_is_allowed_boolean

    return None


def is_value_complete(partial: str, parameter_type: str) -> bool:
    """
    Check a parameter value is complete and valid.
    """
    if parameter_type in ("float", "number", "int"):
        return is_number_complete(partial)

    if parameter_type in ("str", "string"):
        return is_string_complete(partial)

    if parameter_type in ("bool", "boolean"):
        return is_boolean_complete(partial)

    return False
