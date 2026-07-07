import os
from byteplussdkarkruntime import Ark

# Make sure that you have stored the API Key in the environment variable ARK_API_KEY
# Initialize the Ark client to read your API Key from an environment variable
client = Ark(
    # This is the default path. You can configure it based on the service location
    base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
    # Get your Key authentication from the environment variable. This is the default mode and you can modify it as required
    api_key=os.environ.get("ARK_API_KEY"),
)

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
   # Specify the Ark Inference Endpoint ID you created, which has been changed to your Inference Endpoint ID for you.
    model="deepseek-v4-pro-260425",
    messages=[
        {"role": "system", "content": "You are an artificial intelligence assistant."},
        {"role": "user", "content": "What are the common cruciferous plants?"},
    ],
)
print(completion.choices[0].message.content)

# Streaming:
print("----- streaming request -----")
stream = client.chat.completions.create(
    model="deepseek-v4-pro-260425",
    messages=[
        {"role": "system", "content": "You are an artificial intelligence assistant."},
        {"role": "user", "content": "What are the common cruciferous plants?"},
    ],
    # Whether the response content is streamed back
    stream=True,
)
for chunk in stream:
    if not chunk.choices:
        continue
    print(chunk.choices[0].delta.content, end="")
print()