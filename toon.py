# --------------------------------------------------------------
# llm_openai.py  (2025 API Compatible)
# --------------------------------------------------------------

from openai import OpenAI
import json

client = OpenAI()

# --------------------------------------------------------------
# Tool implementation
# --------------------------------------------------------------

def get_weather(city: str, units: str = "metric"):

    fake_weather_db = {
        "New York": {"temp": 12, "condition": "Cloudy"},
        "San Francisco": {"temp": 17, "condition": "Sunny"},
        "London": {"temp": 9, "condition": "Rainy"}
    }

    base = fake_weather_db.get(city, {"temp": 20, "condition": "Unknown"})
    return {
        "city": city,
        "temp": base["temp"],
        "condition": base["condition"],
        "units": units
    }

# --------------------------------------------------------------
# LLM wrapper using new API (responses.create)
# --------------------------------------------------------------

class llm_openai:

    @staticmethod
    def complete(prompt: str):

        # Step 1: Ask model with tool definitions
        response = client.responses.create(
            model="gpt-4.1",
            input=prompt,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Returns weather info for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "units": {
                                    "type": "string",
                                    "enum": ["metric", "imperial"],
                                    "default": "metric"
                                }
                            },
                            "required": ["city"]
                        }
                    }
                }
            ]
        )

        # Extract first message
        msg = response.output[0]

        # ----------------------------------------------------------
        # CASE 1 — Model returned final text, no tool call
        # ----------------------------------------------------------
        if msg.type == "message" and not hasattr(msg, "tool_calls"):
            return msg.content[0].text

        # ----------------------------------------------------------
        # CASE 2 — Tool call required
        # ----------------------------------------------------------
        final_messages = [ {"role": "user", "content": prompt} ]

        for tool_call in msg.tool_calls:

            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # Execute backend tool
            if name == "get_weather":
                tool_result = get_weather(**args)
            else:
                tool_result = {"error": "Unknown tool"}

            final_messages.append({
                "role": "tool",
                "content": json.dumps(tool_result),
                "tool_call_id": tool_call.id
            })

        # Ask the model again with tool response
        final = client.responses.create(
            model="gpt-4.1",
            messages=final_messages
        )

        return final.output_text


# --------------------------------------------------------------
# Run Example
# --------------------------------------------------------------

if __name__ == "__main__":
    print(llm_openai.complete("What's the weather in New York?"))
