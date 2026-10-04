"""Example usage of google/DiarizationLM-Gemma-4-E4B-v1."""

from diarizationlm import utils
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/DiarizationLM-Gemma-4-E4B-v1"

HYPOTHESIS = (
    "<speaker:1> Hello, how are you doing <speaker:2> today? I am doing well."
    " What about <speaker:1> you? I'm doing well, too. Thank you."
)

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, device_map="cuda")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
)

print("Tokenizing input...")
prompt = f"<|turn>user\n{HYPOTHESIS} --> <turn|>\n<|turn>model\n"
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

print("Generating completion...")
outputs = model.generate(
    **inputs,
    max_new_tokens=int(inputs.input_ids.shape[1] * 1.2),
    do_sample=False,
    use_cache=True,
)

print("Decoding completion...")
completion = tokenizer.batch_decode(
    outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
)[0]
completion = utils.truncate_suffix_and_tailing_text(completion, " [eod]")

print("Transferring completion to hypothesis text...")
transferred_completion = utils.transfer_llm_completion(completion, HYPOTHESIS)

print("========================================")
print("Hypothesis:", HYPOTHESIS)
print("========================================")
print("Completion:", completion)
print("========================================")
print("Transferred completion:", transferred_completion)
print("========================================")
