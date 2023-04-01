import math
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class MCTSNode:
    def __init__(
        self,
        parent=None,
        action=None,
        tokenizer=None,
        model=None,
        reward_function=None,
        sentence_length=20,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    ):
        self.parent = parent
        self.tokenizer = tokenizer
        self.model = model
        self.reward = 0
        self.children = []
        self.visit_count = 0
        self.generated_text = parent.generated_text + [action] if parent else []
        self.reward_function = reward_function
        self.sentence_length = sentence_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    # Other methods remain the same

    def rollout(self, rollout_length=1):
        current_node = self
        for _ in range(rollout_length):
            # Generate the full sentence first
            while len(current_node.generated_text) < self.sentence_length:
                current_node = current_node.expand()

            # Perform the rollout on the full sentence
            text = self.tokenizer.decode(
                current_node.generated_text, skip_special_tokens=True
            )
            reward = self.reward_function(text)

            # Backpropagate rewards to the ancestors
            node_to_update = current_node
            while node_to_update is not None:
                node_to_update.reward += reward
                node_to_update.visit_count += 1
                node_to_update = node_to_update.parent

            current_node = current_node.expand()
            # print(tokenizer.decode(current_node.generated_text, skip_special_tokens=True))

    def expand(self):
        input_ids = torch.tensor([self.generated_text])
        logits = self.model(input_ids).logits[:, -1, :]
        logits = logits / self.temperature

        # Apply top-k and top-p filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_logits.softmax(dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[0, sorted_indices_to_remove[0]]
        logits[0, indices_to_remove] = float("-inf")
        top_k_indices = torch.topk(logits, self.top_k)[1]

        # Sample from top-k and top-p filtered logits
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()

        # Create a new child node and add it to the children list
        child_node = MCTSNode(
            parent=self,
            action=action,
            tokenizer=self.tokenizer,
            model=self.model,
            reward_function=self.reward_function,
            sentence_length=self.sentence_length,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        self.children.append(child_node)
        return child_node

    def best_child_max(self, c=1.0):
        # Should I keep this?
        return max(
            self.children,
            key=lambda node: node.reward / (node.visit_count + 1e-5)
            + c * math.sqrt(math.log(self.visit_count + 1) / (node.visit_count + 1e-5)),
        )

    def best_child(self, c=1.0):
        def ucb(node):
            avg_reward = node.reward / (node.visit_count + 1e-5)
            exploration_term = c * math.sqrt(
                math.log(self.visit_count + 1) / (node.visit_count + 1e-5)
            )
            return avg_reward + exploration_term

        best_node = max(self.children, key=ucb)
        # print(best_node.reward / (best_node.visit_count + 1e-5), tokenizer.decode(best_node.generated_text, skip_special_tokens=True))
        return best_node


def mcts_sentence_generator(
    model,
    tokenizer,
    prompt,
    reward_function,
    num_tokens=20,
    iterations=10,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
):
    model.eval()

    initial_text = tokenizer.encode(prompt, return_tensors="pt").tolist()[0]
    root = MCTSNode(
        tokenizer=tokenizer,
        model=model,
        reward_function=reward_function,
        sentence_length=num_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    root.generated_text = initial_text

    for _ in tqdm(range(num_tokens)):
        for _ in range(iterations):
            current_node = root
            current_node.rollout()
            current_node = current_node.best_child()
        # print(tokenizer.decode(current_node.generated_text))
        # print('---------------------------')
        root = current_node

    generated_token_ids = root.generated_text
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return generated_text
