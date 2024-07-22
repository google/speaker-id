from diarizationlm import utils
from transformers import LlamaForCausalLM, AutoTokenizer

HYPOTHESIS = (
    "<speaker:1> Hello, how are you doing <speaker:2> today? I am doing well."
    " What about <speaker:1> you? I'm doing well, too. Thank you."
)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(
    "google/DiarizationLM-13b-Fisher-v1", device_map="cuda"
)
model = LlamaForCausalLM.from_pretrained(
    "google/DiarizationLM-13b-Fisher-v1", device_map="cuda"
)

print("Tokenizing input...")
inputs = tokenizer([HYPOTHESIS + " --> "], return_tensors="pt").to("cuda")

print("Generating completion...")
outputs = model.generate(
    **inputs, max_new_tokens=inputs.input_ids.shape[1] * 1.2, use_cache=False
)

print("Decoding completion...")
completion = tokenizer.batch_decode(
    outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
)[0]

print("Transferring completion to hypothesis text...")
transferred_completion = utils.transfer_llm_completion(completion, HYPOTHESIS)

print("========================================")
print("Hypothesis:", HYPOTHESIS)
print("========================================")
print("Completion:", completion)
print("========================================")
print("Transferred completion:", transferred_completion)
print("========================================")
