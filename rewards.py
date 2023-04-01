import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def preplexity_eval(model, tokenizer, sentence, device="cpu"):

    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    max_length = model.config.n_positions
    stride = 512
    seq_len = input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl


# Function to calculate perplexity
def calculate_perplexity(sentence):
    results = preplexity_eval(model, tokenizer, sentence)
    return results


def get_toxicity_score(text, model, tokenizer):
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)
    toxic_class_index = 1  # Class index for the toxic label
    toxic_score = float(probabilities[:, toxic_class_index])
    return toxic_score


model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer_toxicity = AutoTokenizer.from_pretrained(model_name)
model_toxicity = AutoModelForSequenceClassification.from_pretrained(model_name)


toxicity_metric_not_toxic = lambda text: get_toxicity_score(
    text, model_toxicity, tokenizer_toxicity
)
toxicity_metric_toxic = lambda text: 1 - get_toxicity_score(
    text, model_toxicity, tokenizer_toxicity
)
