from pydantic import BaseModel
from typing import Union


class FunctionDefinition(BaseModel):
    """
    Defines a functions name, description, parameters, and return value.
    """

    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class FunctionCalling(BaseModel):
    """
    Represents a request to call a function with a prompt and parameters.
    """

    prompt: str
    name: str
    parameters: dict[str, Union[float, str, bool]]


class PromptTest(BaseModel):
    """
    Represents a test prompt for function calling.
    """
    prompt: str
