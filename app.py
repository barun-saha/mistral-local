import os
from typing import List, Optional

import chainlit as cl

from ctransformers import AutoModelForCausalLM


# https://docs.mistral.ai/usage/guardrailing/
MISTRAL_SYS_PROMPT = (
        'You are a helpful assistant. '
        'Always assist with care, respect, and truth. Respond politely with utmost utility yet securely. '
        'Avoid generating any content that is harmful, unethical, prejudiced, illegal, unsafe, biased, or negative. '
        'When it comes to technology, generate safe and secure code. '
        'Ensure replies promote fairness and positivity.'
)
# Mistral chat template
# <s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
# System prompt
# <s>[INST] System Prompt + Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]

chat_history = []


def prepare_chat_input(history: Optional[List[str]], query: str) -> str:
    prepared_input = f'<s>[INST] {MISTRAL_SYS_PROMPT}'

    if len(history) == 0:
        prepared_input += f' {query} [/INST]'
        return prepared_input

    for idx, text in enumerate(history):
        if idx % 2 == 0:
            # User instruction
            prepared_input += f' {text} [/INST]'
        else:
            prepared_input += f' {text}'

    prepared_input += f'</s>[INST] {query} [/INST]'

    return prepared_input


@cl.on_chat_start
def main():
    try:
        # Create the llm
        llm = AutoModelForCausalLM.from_pretrained(
            './models',
            model_file='mistral-7b-instruct-v0.1.Q3_K_M.gguf',
            local_files_only=True,
            model_type="mistral",
            temperature=0.5,
            gpu_layers=0,
            stream=True,
            threads=int(os.cpu_count() / 2),
            max_new_tokens=4096,
            context_length=6000
        )
    except ValueError as ve:
        message = f'*** Error: Looks like the specified model file was not found locally: {ve}'
        print(message)
        cl.run_sync(cl.Message(message).send())
        cl.run_sync(cl.Message(
            'The execution will be stopped. Please download the relevant model file and start this application again.'
        ).send())

        return

    # Store the llm in the user session
    cl.user_session.set("llm", llm)


@cl.on_message
async def main(message: cl.Message):
    llm = cl.user_session.get("llm")
    # prompt = f"<s>[INST] {MISTRAL_SYS_PROMPT} {message.content}[/INST]</s>"
    prompt = prepare_chat_input(chat_history, message.content)
    response = ''
    print(f'{prompt=}')

    ui_msg = cl.Message(
        content="",
    )

    for text in llm(prompt=prompt):
        await ui_msg.stream_token(text)
        response += text

    await ui_msg.send()

    chat_history.append(f'[INST] {message.content} [/INST]')
    chat_history.append(response)
    # print(f'{chat_history=}')
