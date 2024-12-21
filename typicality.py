import numpy as np
import torch
from tqdm import tqdm


def typical_set_sampling(logits, temperature, epsilon):
    probs = torch.softmax(logits / temperature, dim=-1).numpy()
    log_probs = np.log(probs)
    entropy = -np.sum(probs * log_probs)

    mask = np.abs(log_probs + entropy) < epsilon
    typical_probs = np.where(mask, probs, 0)
    typical_probs_sum = np.sum(typical_probs)

    # Check if there are any typical probabilities
    if typical_probs_sum > 0:
        typical_probs /= typical_probs_sum
        return np.random.choice(logits.shape[-1], p=typical_probs.ravel())
    else:
        # Fallback to the token with the highest probability
        return np.argmax(probs)


def generate_with_typical_set_sampling(
    model, tokenizer, input_text, temperature=1.0, max_length=50, num_beams=1, epsilon=0.1
):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        model.eval()

        # Initialize beams
        beams = [input_ids for _ in range(num_beams)]

        for _ in tqdm(range(max_length - len(input_ids[0]))):

            # Calculate the next token logits for all beams
            next_token_logits = [model(beam).logits[:, -1, :] for beam in beams]

            # Apply typical set sampling to select the next token for all beams
            next_tokens = [typical_set_sampling(logits, temperature, epsilon) for logits in next_token_logits]

            # Update the beams with the new tokens
            beams = [
                torch.cat([beam, torch.tensor([[next_token]], dtype=torch.long)], dim=-1)
                for beam, next_token in zip(beams, next_tokens)
            ]

    decoded_outputs = [tokenizer.decode(output[0], skip_special_tokens=True) for output in beams]
    return decoded_outputs
