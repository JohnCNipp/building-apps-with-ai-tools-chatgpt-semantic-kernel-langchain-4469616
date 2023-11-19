import os
import openai
from openai import OpenAI
from OPEN_AI_KEY import OPENAI_API_KEY
# openai.api_key = OPENAI_API_KEY
client = OpenAI(
    api_key = OPENAI_API_KEY
)
# Challenge: Turning Away Rude Customers
# Build a GPT-4 python app that talks with a user.
# End the conversation if they're being rude
while True:
    user_input = input("Hi, how can I help you today? Hope you are well.\n")
    if user_input == 'exit' or user_input == 'quit':
        break
    response = client.chat.completions.create(
    messages = [
        {   "role": "system", "content": """You are a sentiment classification bot,
            who is chatting with customers. Classify whether the reponse is RUDE
            or OK. If they talk about you, or the product without being
            constructive, or are just complaining, mark them as RUDE. If not,
            mark OK.""",
            },
            {"role": "user", "content": user_input},
        ],
        model = 'gpt-3.5-turbo',
        temperature = .7,
        max_tokens = 150
    )
    customer_response = response.choices[0].message.content
    if customer_response == "RUDE":
        print("I'm sorry, I can't help rude customers")
    # perhaps make this more complex and have another bot ready here?
    else:
        print("Very nice")

# test case 1 'you're the worst human i've talked to' -> RUDE
# test case 2 'hey how's your day going'
# test case 3 'I like pizza. What do you like?'
# test case 4 'I bite my thumb at you!'
# test case 5 'I think this product doesnt work!' -> RUDE

# Verified to be successful
