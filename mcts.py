import math

import numpy as np
import torch
from colorama import Back, Fore, Style, init
from tqdm import tqdm


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
        top_k=50,
        top_p=0.9,
    ):
        """Initialize MCTS node.

        Args:
            parent: Parent node
            action: Token ID of the action that led to this node
            tokenizer: Tokenizer for text conversion
            model: Language model
            reward_function: Function to evaluate text quality
            sentence_length: Maximum sentence length
            temperature: Sampling temperature
            top_k: Number of top tokens to keep for sampling
            top_p: Cumulative probability threshold for nucleus sampling
        """
        self.parent = parent
        self.tokenizer = tokenizer
        self.model = model
        self.total_reward = 0
        self.children = []
        self.visit_count = 0
        self.generated_text = parent.generated_text + [action] if parent else []
        self.reward_function = reward_function
        self.sentence_length = sentence_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self._is_terminal = len(self.generated_text) >= sentence_length
        self.value = 0

    def rollout(self):
        """Perform a single MCTS rollout."""
        if self._is_terminal:
            text = self.tokenizer.decode(self.generated_text, skip_special_tokens=True)
            reward = self.reward_function(text)
            self._backpropagate(reward)
            return

        # Expand and continue rollout
        next_node = self.expand()
        next_node.rollout()

    def _backpropagate(self, reward):
        """Backpropagate rewards through the tree."""
        node = self
        while node is not None:
            node.total_reward += reward
            node.visit_count += 1
            node.value = node.total_reward / (node.visit_count + 1e-8)
            node = node.parent

    def expand(self):
        """Expand the current node by adding a child."""
        with torch.no_grad():
            input_ids = torch.tensor([self.generated_text]).to(self.model.device)
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :] / self.temperature

            # Apply top-k filtering
            if self.top_k > 0:
                values, _ = torch.topk(logits, self.top_k)
                min_value = values[:, -1].unsqueeze(-1).expand_as(logits)
                logits = torch.where(logits < min_value, torch.full_like(logits, float("-inf")), logits)

            # Apply nucleus (top-p) filtering
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = float("-inf")

            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()

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

    def best_child(self, c=1.414):
        """Select best child using UCB1 formula.

        Args:
            c: Exploration constant (default: sqrt(2))
        """
        if not self.children:
            return None

        def ucb(node):
            exploitation = node.total_reward / (node.visit_count + 1e-8)
            exploration = c * math.sqrt(math.log(self.visit_count + 1) / (node.visit_count + 1e-8))
            return exploitation + exploration

        return max(self.children, key=ucb)


def get_color_for_value(value, min_val, max_val):
    """Return color based on normalized value."""
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)

    if normalized < 0.33:
        return Fore.BLUE
    elif normalized < 0.66:
        return Fore.YELLOW
    else:
        return Fore.RED


def mcts_sentence_generator(
    model,
    tokenizer,
    prompt,
    reward_function,
    num_tokens=20,
    iterations=10,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Generate text using MCTS with the specified parameters.

    Args:
        model: Language model
        tokenizer: Tokenizer for text conversion
        prompt: Initial text prompt
        reward_function: Function to evaluate text quality
        num_tokens: Maximum number of tokens to generate
        iterations: Number of MCTS iterations per token
        temperature: Sampling temperature
        top_k: Number of top tokens to keep for sampling
        top_p: Cumulative probability threshold for nucleus sampling
        device: Device to run the model on
    """
    model.eval()
    model.to(device)

    initial_text = tokenizer.encode(prompt, return_tensors="pt").tolist()[0]
    root = MCTSNode(
        tokenizer=tokenizer,
        model=model,
        reward_function=reward_function,
        sentence_length=len(initial_text) + num_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    root.generated_text = initial_text

    with torch.no_grad():
        for _ in tqdm(range(num_tokens), desc="Generating tokens"):
            # Run MCTS iterations
            for _ in range(iterations):
                root.rollout()

            # Select best next node
            best_child = root.best_child()
            if best_child is None:
                break
            root = best_child

    # Collect all nodes and their values by traversing back from root
    generated_tokens = []
    current = root
    while current is not None:
        if current.generated_text:
            generated_tokens.insert(0, (current.generated_text[-1], current.value))
        current = current.parent

    print(generated_tokens)

    # Get value range for normalization
    values = [v for _, v in generated_tokens[len(initial_text) :]]
    if values:
        min_val, max_val = min(values), max(values)
    else:
        min_val, max_val = 0, 1

    # Generate colored text
    colored_text = ""

    # Decode the full text with no colors
    decoded_full = tokenizer.decode([t for t, _ in generated_tokens], skip_special_tokens=True)
    print(prompt + " " + decoded_full)

    # Process tokens in sequence
    current_text = ""
    for i, (token, value) in enumerate(generated_tokens):
        # Decode up to current token
        new_text = tokenizer.decode([t for t, _ in generated_tokens[: i + 1]], skip_special_tokens=True)
        # Extract just the new part
        token_text = new_text[len(current_text) :]
        current_text = new_text

        # Color the token if it's generated (not from prompt)
        if i >= len(initial_text):
            color = get_color_for_value(value, min_val, max_val)
            colored_text += color + token_text + Style.RESET_ALL
        else:
            colored_text += token_text

    return prompt + " " + colored_text
