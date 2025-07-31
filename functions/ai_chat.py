import os
from openai import OpenAI
from dotenv import load_dotenv


class AI:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def chatgpt(self, model="gpt-4o-mini", instructions="", input_text=""):
        """Interfaces with openai python function"""
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        if input_text:
            messages.append({"role": "user", "content": input_text})

        if not messages:
            return ""

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    gpt = AI()
    translated_text = gpt.chatgpt(
        instructions="Translate the following English text to Japanese:",
        input_text="Hello there"
    )
    print(translated_text)
