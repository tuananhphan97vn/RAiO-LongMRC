from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers import squad_convert_examples_to_features , AutoTokenizer , AutoModelForQuestionAnswering
import torch 
processor =  SquadV1Processor()
file_name = '/data/nlp/hoangnv74/question-answering/data/viquad/train_positive.json'
# examples = processor.get_train_examples(None, filename=file_name)
tokenizer = AutoTokenizer.from_pretrained(
    "../roberta-base",
    do_lower_case=True,
    use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
)

model = AutoModelForQuestionAnswering.from_pretrained("../roberta-base")

text = r"""
Adoptees who publicly support Roe targeted by anti-abortion activists: ‘What if you were aborted?’
"""

questions = [
"Adoptees who publicly support Roe targeted by anti-abortion activists: ‘What if you were aborted?’"
]

# for question in questions:
#     inputs = tokenizer.encode(question, text, add_special_tokens=True, return_tensors="pt")
#     input_ids = inputs["input_ids"].tolist()[0]

#     print('input ', inputs)
#     print('input_dix shape ', inputs['input_ids'].shape)
#     text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

a = tokenizer.encode(questions)
print(a)
b = tokenizer.decode(a)
print(b)
