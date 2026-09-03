import json


class LoadVocab():
    """
    Load a vocabulary file and convert between token IDs and token text
    """
    def __init__(self, path: str) -> None:
        """
        Load the vocabulary from the JSON file at the given path
        """
        with open(path, "r") as file:
            data: dict[str, int] = json.load(file)
            self.data = data
            self.swaped_data = {value: key for key, value in self.data.items()}

    def id_to_token(self, token_id: int) -> str:
        """
        Return the token text associated with a token ID.
        """
        return self.swaped_data[token_id]

    def token_to_id(self, token: str) -> int:
        """
        Return the token ID associated with a token.
        """
        return self.data[token]
