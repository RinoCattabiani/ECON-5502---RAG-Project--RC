from groq import Groq
import os


# API key
api_key = os.getenv("groq_api_key")

# create a client 
client = Groq(api_key=api_key)

# create a model
model = "llama-3.1-8b-instant"

# store the running conversation here 
history = [
    {"role": "system", "content": "You are a friendly chatbot."}
]

print("Chatbot ready! Type 'exit' to stop.")

# keep chatting until user types "exit"
while True:
    user_text = input("\nYou: ")
    if user_text.lower() == "exit":
        print("Goodbye!")
        break

    # add the users question
    history.append({"role": "user", "content": user_text})

    # ask model to reply 
    response = client.chat.completions.create(
        model=model,
        messages=history
    )

    bot_text = response.choices[0].message.content 

    # add reply so follow-up questions have context 
    history.append({"role": "assistant", "content": bot_text})

    # show the reply
    print("\nBot:", bot_text)