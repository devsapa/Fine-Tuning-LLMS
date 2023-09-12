import logging
import re

import torch
import numpy as np
from peft import PeftModel
from transformers import Pipeline, PreTrainedTokenizer, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.encode_decode import encode, decode


device = "cuda"
model = None
tokenizer = None

# Function to format the response and filter out the instruction from the response.
def postprocess(response):
    messages = response.split("Response:")
    if not messages:
        raise ValueError("Invalid template for prompt. The template should include the term 'Response:'")
    return "".join(messages[1:])

def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        **kwargs,
):
    prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction: {instruction}\n Response:"""

    inf_pipeline =  pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=256, trust_remote_code=True)

    response = inf_pipeline(prompt_template.format(instruction=instruction))[0]['generated_text']
    formatted_response = postprocess(response)
    return formatted_response
 
    
def load_base_model(adapter_checkpoint, adapter_name):
    model_name = "databricks/dolly-v2-3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="balanced",
        load_in_8bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter_checkpoint, adapter_name)
    return model, tokenizer


def inference(inputs: Input):
    json_input = decode(inputs, "application/json")
    sequence = json_input.get("inputs")
    generation_kwargs = json_input.get("parameters", {})
    output = Output()
    outs = evaluate(sequence)
    encode(output, outs, "application/json")
    return output


def handle(inputs: Input):
    """
    Default handler function
    """
    global model, tokenizer
    if not model:
        # stateful model
        props = inputs.get_properties()
        model, tokenizer = load_base_model(props.get("adapter_checkpoint"), props.get("adapter_name"))

    if inputs.is_empty():
        # initialization request
        return None

    return inference(inputs)
