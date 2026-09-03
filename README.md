*This project has been created as part of the 42 curriculum by elhamza.*

## Description

This project implements a **function calling** tool that translates natural language
prompts into structured, machine-executable function calls, using a small (0.6B
parameter) local LLM.

## Instructions

### Installation

```bash
make install
```

### Running

```bash
make run
```

Or use:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```


## Resources

- [JSON Schema specification](https://json-schema.org/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [mypy documentation](https://mypy.readthedocs.io/)

**How AI was used:** AI was help to understand the constrained decoding and grammar.

## Algorithm Explanation — Constrained Decoding

At each generation step:

1. The model produces logits over the full vocabulary via
   `get_logits_from_input_ids`.
2. A grammar is built dynamically from the real `functions_definition.json` at
   runtime (never hardcoded), mapping each function name to its own parameter
   schema.
3. Fixed, known parts of the JSON (`{`, key names, `:`, `,`, `}`, and the original
   prompt text itself) are written directly, without asking the model to generate
   them — only the parts that require an actual decision (the function name, and
   each parameter's value) are generated token by token.
4. For each of those "decision" slots, at every step, every token in the vocabulary
   (loaded from `get_path_to_vocab_file`) is checked: would adding this token's text
   keep the partial output either (a) a valid prefix of one of the allowed function
   names, or (b) a syntactically valid partial value of the expected type (number,
   string, or boolean)?
5. Logits for every token that fails this check are set to `-inf`, so they can never
   be selected, no matter how high their original score was.
6. The highest-scoring token among the remaining legal ones is selected (`argmax`).
7. The chosen token is appended to the running input, its text form (with the
   tokenizer's leading-space marker converted to a real space) is appended to the
   output, and the process repeats until the current slot is complete (e.g. the
   function name exactly matches a real function, or a string/number/boolean value
   is properly closed).


## Design Decisions

* **Pydantic** validates all input and output data using clear schemas.
* The **grammar is built dynamically** from `functions_definition.json`, so no function names or parameters are hardcoded.
* **Constrained generation is generic** and reused for function names and parameter values (number, string, boolean).
* **Error handling is defensive**: invalid prompts return `None` and do not stop the remaining batch from processing.


## Performance Analysis

* **JSON validity:** 100% — constrained decoding masks invalid tokens at every step, guaranteeing valid JSON structure and types.
* **Function selection:** Mostly correct, with some misclassification on ambiguous prompts due to the limitations of the 0.6B model.
* **Parameter extraction:** String values were reliable, while numeric and complex regex values were less accurate.
* **Performance & robustness:** All 11 tests complete in roughly 60 seconds on CPU, with invalid inputs and file errors handled gracefully.


## Challenges Faced

* **JSON-aware token masking:** Building token-level rules for valid JSON, function names, and number/string/boolean values was the main challenge.
* **Tokenizer handling:** Tokens containing the leading-space marker had to be converted correctly into normal spaces.
* **Generation robustness:** Missing or invalid input files could cause generation to loop indefinitely, so early validation was added.
* **Model & tooling limitations:** Improving function selection required better context, while `mypy` and `flake8` needed configuration to ignore the provided SDK and virtual environment.


## Testing Strategy

* **End-to-end testing:** Run the full CLI using the provided JSON files and verify the generated output.
* **Failure testing:** Test missing files, invalid JSON, missing output directories, and empty prompt lists to ensure graceful handling.
* **Logic testing:** Test constrained-decoding helpers with normal and edge cases for function names, numbers, strings, and booleans.


## Example Usage

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Input (`function_calling_tests.json`):

```json
[
  {"prompt": "What is the sum of 2 and 3?"}
]
```

Output (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```