import json
import numpy
from concurrent.futures import ThreadPoolExecutor
from models import FunctionCalling
from llm_sdk import Small_LLM_Model
from tokenizer_map import LoadVocab
from pydantic import ValidationError
from typing import Any, Callable, Optional

from constrained_decoder import (
    build_grammar,
    token_is_allowed,
    is_function_name_complete,
    # token_is_allowed_string,
    # is_string_complete,
    get_value_checker,
    is_value_complete,
)

from load_file import functions_definition, function_calling

model = Small_LLM_Model()
path_vocab = model.get_path_to_vocab_file()
vocab = LoadVocab(path_vocab)

_candidate_cache: dict[tuple[str, str], list[int]] = {}


def get_candidate_tokens(
    slot_label: str,
    is_allowed_fn: Callable[[str, str], bool],
    partial: str,
) -> list[int]:
    """
    Return the list of token_ids allowed to follow `partial`.
    """
    cache_key = (slot_label, partial)
    cached = _candidate_cache.get(cache_key)
    if cached is not None:
        return cached

    candidates = [
        token_id
        for token_id, token_string in vocab.swaped_data.items()
        if is_allowed_fn(partial, token_string)
    ]
    _candidate_cache[cache_key] = candidates
    return candidates


def generate_constrained(
    run_token: list[int],
    slot_label: str,
    is_allowed_fn: Callable[[str, str], bool],
    is_complete_fn: Callable[[str], bool],
) -> str:
    """
    Generate text while allowing only tokens accepted by the grammar.
    """
    partial = ""

    while True:
        next_token = model.get_logits_from_input_ids(run_token)

        next_token_array = numpy.array(next_token)

        candidates = get_candidate_tokens(slot_label, is_allowed_fn, partial)

        masked = numpy.full(len(next_token_array), -numpy.inf)
        for token_id in candidates:
            masked[token_id] = next_token_array[token_id]
        next_token_array = masked

        next_token_id = numpy.argmax(next_token_array, axis=-1)

        next_token_id_int = int(next_token_id)

        try:
            next_token_string = vocab.id_to_token(next_token_id_int)
        except KeyError:
            break

        run_token.append(next_token_id_int)
        next_token_string = next_token_string.replace("Ġ", " ")
        partial += next_token_string
        if is_complete_fn(partial):
            break

    return partial


def append_fixed_text(run_token: list[int], output: str, text: str) -> str:
    """
    Appends literal grammar text
    """
    ids = model.encode(text).flatten().tolist()
    run_token.extend(ids)
    return output + text


def generate_parameters_object(
    run_token: list[int], output: str, param_specs: dict[str, dict[str, str]]
) -> str:
    """
    param_specs: dict like {"a": {"type": "number"}, "b": {"type": "number"}}
    Builds: {"a": 1, "b": 2}
    """
    output = append_fixed_text(run_token, output, "{")

    param_names = list(param_specs.keys())

    for i, param_name in enumerate(param_names):
        param_type = param_specs[param_name]["type"]

        output = append_fixed_text(run_token, output, f'"{param_name}":')

        checker = get_value_checker(param_type)
        if checker is None:
            raise ValueError(f"Unsupported parameter type: {param_type}")

        is_allowed_fn = checker

        def is_complete_fn(p: str, t: str = param_type) -> bool:
            return is_value_complete(p, t)

        value_partial = generate_constrained(
            run_token, param_type, is_allowed_fn, is_complete_fn
        )
        output += value_partial

        if i < len(param_names) - 1:
            output = append_fixed_text(run_token, output, ",")

    output = append_fixed_text(run_token, output, "}")
    return output


def user_prompt(text_prompt: str, functions: list[Any]) -> str:
    """
    Build the prompt containing the available functions.
    """
    function_name = [
        f'"{function.name}": "{function.description}".\n' for function in functions
    ]
    function_name_text = "\n".join(function_name)
    text = "You are choosing which function to call for a user's request.\n"
    text += f"Available functions: {function_name_text}\n"
    text += f"{text_prompt}\n"
    return text


def generate_function_call(text: str, functions: list[Any]) -> str:
    """
    Generate a JSON function call for the given user prompt.
    """
    grammar = build_grammar(functions)
    prompt = user_prompt(text, functions)
    tokens_id = model.encode(prompt)
    run_token = tokens_id.flatten().tolist()

    output = ""

    for item in grammar["root"]:
        if item == "string":
            prompt_literal = json.dumps(text)
            output = append_fixed_text(run_token, output, prompt_literal)

        elif item == "function_name":
            allowed_names = grammar["function_name"]

            def is_allowed_fn(
                p: str,
                t: str,
                names: list[str] = allowed_names,
            ) -> bool:
                return token_is_allowed(p, t, names)

            def is_complete_fn(
                p: str,
                names: list[str] = allowed_names,
            ) -> bool:
                return is_function_name_complete(p, names)

            chosen_quoted_name = generate_constrained(
                run_token, "function_name", is_allowed_fn, is_complete_fn
            )
            output += chosen_quoted_name

            chosen_name = chosen_quoted_name.strip('"')

        elif item == "parameters":
            param_specs = grammar["parameters"][chosen_name]
            output = generate_parameters_object(run_token, output, param_specs)

        else:
            output = append_fixed_text(run_token, output, item)

    return output


def single_prompt(text: str, functions: list[Any]) -> Optional[FunctionCalling]:
    """
    Generate and validate a function call for one prompt
    """
    format_json = generate_function_call(text, functions)
    try:
        data = json.loads(format_json)
        ft_calling = FunctionCalling.model_validate(data)

        return ft_calling
    except json.JSONDecodeError:
        print("The JSON is not valid")
        return None

    except ValidationError:
        print("The data is not valid")
        return None


def process_batch(
    prompts: list[Any],
    functions: list[Any],
    max_workers: int = 2,
) -> list[Optional[FunctionCalling]]:
    """
    Process prompts CONCURRENTLY using threads..
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                lambda prom: single_prompt(prom.prompt, functions),
                prompts,
            )
        )
    return results


def process_all_prompts(
    path_result: str = "data/output/function_calling_results.json",
    path_definition: str = "data/input/functions_definition.json",
    path_calling: str = "data/input/function_calling_tests.json",
) -> None:
    """
    Process all prompts and save the generated function calls.
    """
    functions = functions_definition(path_definition)
    if not functions:
        print("No functions were loaded. Cannot process prompts.")
        return
    prompts = function_calling(path_calling)

    results = process_batch(prompts, functions)

    list_of_dicts = [res.model_dump() for res in results if res]

    try:
        with open(path_result, "w") as file:
            json.dump(list_of_dicts, file, indent=2)
    except FileNotFoundError:
        print("The file does not exist")
