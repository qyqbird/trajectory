from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

OpenAI_client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
