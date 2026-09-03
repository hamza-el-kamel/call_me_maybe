from models import PromptTest, FunctionDefinition
from pydantic import ValidationError
import json


def functions_definition(
        path: str = "data/input/functions_definition.json"
        ) -> list[FunctionDefinition]:
    """
    Load and validate function definitions from a JSON file.
    """
    try:
        with open(path, "r") as file:
            data = json.load(file)
        ft_definition = [
            FunctionDefinition.model_validate(function)
            for function in data
            ]
        return ft_definition
    except FileNotFoundError:
        print("The file does not exist")
        return []

    except json.JSONDecodeError:
        print("The JSON is not valid")
        return []

    except ValidationError:
        print("The data is not valid")
        return []


def function_calling(
        path: str = "data/input/function_calling_tests.json"
        ) -> list[PromptTest]:
    """
    Load and validate prompt tests from a JSON file.
    """
    try:
        with open(path, "r") as file:
            data = json.load(file)
        test = [
            PromptTest.model_validate(test)
            for test in data
            ]
        return test
    except FileNotFoundError:
        print("The file does not exist")
        return []

    except json.JSONDecodeError:
        print("The JSON is not valid")
        return []

    except ValidationError:
        print("The data is not valid")
        return []
