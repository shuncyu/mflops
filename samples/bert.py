from functools import partial

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from mflops import get_model_compute_info


def bert_input_constructor(input_shape, tokenizer):
    inp_seq = ""
    for _ in range(input_shape[1] - 2):  # there are two special tokens [CLS] and [SEP]
        inp_seq += tokenizer.pad_token  # let's use pad token to form a fake
    # sequence for subsequent flops calculation

    inputs = tokenizer([inp_seq] * input_shape[0], padding=True, truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * input_shape[0])
    # Batch size input_shape[0], sequence length input_shape[128]
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs


if __name__ == '__main__':
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    flops_count, params_count = get_model_compute_info(
            model, (2, 128),
            input_constructor=partial(bert_input_constructor, tokenizer=bert_tokenizer),
            print_per_layer_stat=False)

# Output:
# Computational complexity:       21.74 GMac
# Number of parameters:           109.48 M
