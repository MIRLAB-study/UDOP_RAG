from transformers import AutoProcessor, AutoModel
from datasets import load_dataset
import torch

processor = AutoProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)
model = AutoModel.from_pretrained("microsoft/udop-large")

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[:2]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]
print(type(example['image']))
# print(example['tokens'])
# print(example['bboxes'])
inputs = processor(image, words, boxes=boxes, return_tensors="pt",padding=True)

decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])
# print(inputs)
for key,vlaue in inputs.items():
    print(key, vlaue.shape)

# forward pass
encoder_outputs=model.encoder(**inputs)
last_hidden_state = encoder_outputs.last_hidden_state
print(last_hidden_state.shape)
for key,vlaue in encoder_outputs.items():
    print(key, vlaue.shape)
# outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
# last_hidden_states = outputs.last_hidden_state
# list(last_hidden_states.shape)
# print(list(last_hidden_state.shape))