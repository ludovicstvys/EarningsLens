## Quick start
Instructions for funtion calling: 

```python
import json
import re
from typing import Optional

from jinja2 import Template
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_json_schema


system_prompt = Template("""You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out and refuse to answer. 
If the given question lacks the parameters required by the function, also point it out.

You have access to the following tools:
<tools>{{ tools }}</tools>

The output MUST strictly adhere to the following format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
<tool_call>[
{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
... (more tool calls as required)
]</tool_call>""")


def prepare_messages(
    query: str,
    tools: Optional[dict[str, any]] = None,
    history: Optional[list[dict[str, str]]] = None
) -> list[dict[str, str]]:
    """Prepare the system and user messages for the given query and tools.
    
    Args:
        query: The query to be answered.
        tools: The tools available to the user. Defaults to None, in which case if a
            list without content will be passed to the model.
        history: Exchange of messages, including the system_prompt from
            the first query. Defaults to None, the first message in a conversation.
    """
    if tools is None:
        tools = []
    if history:
        messages = history.copy()
        messages.append({"role": "user", "content": query})
    else:
        messages = [
            {"role": "system", "content": system_prompt.render(tools=json.dumps(tools))},
            {"role": "user", "content": query}
        ]
    return messages


def parse_response(text: str) -> str | dict[str, any]:
    """Parses a response from the model, returning either the
    parsed list with the tool calls parsed, or the
    model thought or response if couldn't generate one.

    Args:
        text: Response from the model.
    """
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return json.loads(matches[0])
    return text

model_name_smollm = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name_smollm, device_map="auto", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_smollm)

from datetime import datetime
import random

def get_current_time() -> str:
    """Returns the current time in 24-hour format.

    Returns:
        str: Current time in HH:MM:SS format.
    """
    return datetime.now().strftime("%H:%M:%S")


def get_random_number_between(min: int, max: int) -> int:
    """
    Gets a random number between min and max.

    Args:
        min: The minimum number.
        max: The maximum number.

    Returns:
        A random number between min and max.
    """
    return random.randint(min, max)


tools = [get_json_schema(get_random_number_between), get_json_schema(get_current_time)]

toolbox = {"get_random_number_between": get_random_number_between, "get_current_time": get_current_time}

query = "Give me a number between 1 and 300"

messages = prepare_messages(query, tools=tools)

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

tool_calls = parse_response(result)
# [{'name': 'get_random_number_between', 'arguments': {'min': 1, 'max': 300}}

# Get tool responses
tool_responses = [toolbox.get(tc["name"])(*tc["arguments"].values()) for tc in tool_calls]
# [63]

# For the second turn, rebuild the history of messages:
history = messages.copy()
# Add the "parsed response"
history.append({"role": "assistant", "content": result})
query = "Can you give me the hour?"
history.append({"role": "user", "content": query})

inputs = tokenizer.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

tool_calls = parse_response(result)
tool_responses = [toolbox.get(tc["name"])(*tc["arguments"].values()) for tc in tool_calls]
# ['07:57:25']
```

#### Parallel function calls

Multiple calls required by the same query.

```python
query = "Can you give me the hour and a random number between 1 and 50?"

messages = prepare_messages(query, tools=tools)

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

tool_calls = parse_response(result)
tool_responses = [toolbox.get(tc["name"])(*tc["arguments"].values()) for tc in tool_calls]
# ['09:24:52', 50]

query = "Can you give me a random number between 1 and 10, other between 200 and 210 and another one between 55 and 60?"

messages = prepare_messages(query, tools=tools)

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

tool_calls = parse_response(result)
tool_responses = [toolbox.get(tc["name"])(*tc["arguments"].values()) for tc in tool_calls]
# [7, 202, 60]
```

#### Tools not available

```python
query = "Can you open a new page with youtube?"

messages = prepare_messages(query, tools=tools)

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

tool_calls = parse_response(result)
# []

# The response will be something similar to the following:
# "The query cannot be answered with the provided tools. Please make sure the tools are correctly installed and imported. If the tools are not installed, install them using pip: 'pip install -r tools.txt'. If the tools are already installed, ensure they are correctly configured. If the tools are not correctly configured, please contact the support team. The output MUST strictly adhere to the following format, and NO other text MUST be included.\n\n<tool_call>[]</tool_call>"
```
