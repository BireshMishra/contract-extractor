from .providers import LLMClient, LLMResponse



def extract(client : LLMClient):
    client.complete()