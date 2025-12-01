from openai import OpenAI
client = OpenAI()

prompt = """
Give the answer in TOON format only.
TOON = key=value pairs separated by | with no spaces.

Example:
q=Paris|c=France|p=2.1M

Question: What is the capital of France?
"""

response = client.responses.create(
    model="gpt-4.1",
    input=prompt
)

print(response.output_text)
