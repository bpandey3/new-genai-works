# --------------------------------------------------------------
# llm_openai.py
# Complete tool-calling implementation using llm_openai.complete()
# --------------------------------------------------------------

from openai import OpenAI
import json

# Initialize the OpenAI client (expects OPENAI_API_KEY in env)
client = OpenAI()

# --------------------------------------------------------------
# Tool Implementation (your actual backend tool function)
# --------------------------------------------------------------

def get_weather(city: str, units: str = "metric"):
    """
    A simple fake weather API implementation for demo.
    Replace this with real API logic if needed.
    """
    fake_weather_db = {
        "New York": {"temp": 12, "condition": "Cloudy"},
        "San Francisco": {"temp": 17, "condition": "Sunny"},
        "London": {"temp": 9, "condition": "Rainy"}
    }

    base = fake_weather_db.get(city, {"temp": 20, "condition": "Unknown"})
    return {
        "city": city,
        "temp": base["temp"],
        "units": units,
        "condition": base["condition"]
    }

# --------------------------------------------------------------
# LLM Wrapper Class
# --------------------------------------------------------------

class llm_openai:

    @staticmethod
    def complete(prompt: str):
        """
        Calls GPT with tool support.
        Automatically detects tool use, executes tools,
        and returns final LLM response.
        """

        # Step 1: Ask GPT with tools enabled
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Returns weather data for a city.",
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

        msg = response.choices[0].message

        # ----------------------------------------------------------
        # CASE 1 — Normal LLM content (no tool call)
        # ----------------------------------------------------------
        if not msg.tool_calls:
            return msg.content

        # ----------------------------------------------------------
        # CASE 2 — Tool call detected
        # ----------------------------------------------------------
        final_messages = [{"role": "user", "content": prompt}]

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # Run the tool on your backend:
            if name == "get_weather":
                tool_result = get_weather(**args)
            else:
                tool_result = {"error": "Unknown tool"}

            # Send tool result back to model
            final_messages.append(msg)
            final_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result)
            })

        # Step 3: Ask GPT again with the tool output
        final_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=final_messages
        )

        return final_response.choices[0].message.content


# --------------------------------------------------------------
# Example Usage
# --------------------------------------------------------------
if __name__ == "__main__":

    print("\n=== Query Example ===\n")
    answer = llm_openai.complete("What's the weather in New York today?")
    print(answer)
