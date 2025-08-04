# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 2025-08-02-langchain
#     language: python
#     name: python3
# ---

# %%
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# %%
model = ChatOllama(
    model="llama3.2",
    temperature=0.1,
    max_tokens=1000,
    top_p=0.95,
    top_k=40,
)

# %%
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = model.invoke(messages)
ai_msg

# %%
messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

model.invoke(messages)

# %%
for token in model.stream(messages):
    print(token, end="|")

# %%
