import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument(
    "--functions_definition",
    default="data/input/functions_definition.json"
    )
parser.add_argument(
    "--input",
    default="data/input/function_calling_tests.json"
    )
parser.add_argument(
    "--output",
    default="data/output/function_calling_results.json"
    )
parser.add_argument(
    "--model",
    default="Qwen/Qwen3-0.6B",
    help="HuggingFace model identifier to use",
)

args = parser.parse_args()

os.environ["LLM_MODEL_NAME"] = args.model
from function_call import process_all_prompts

if __name__ == "__main__":
    process_all_prompts(args.output, args.functions_definition, args.input)
