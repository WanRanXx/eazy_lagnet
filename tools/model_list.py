from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
print(client.models.list())