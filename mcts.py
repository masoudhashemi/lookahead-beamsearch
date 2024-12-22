import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama so that ANSI color codes work across platforms
init(autoreset=True)


def get_color_for_value(value, min_val, max_val):
    """
    Return a color based on the node value, normalized between min_val and max_val.

    Args:
        value (float): Value of the node/token.
        min_val (float): Minimum value among all tokens.
        max_val (float): Maximum value among all tokens.

    Returns:
        str: A color from colorama.Fore.
    """
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


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
    """
    Filter a distribution of logits using top-k and top-p (nucleus) filtering.

    Args:
        logits (torch.Tensor): Logits distribution (size [vocab_size]).
        top_k (int): Keep only top k tokens with highest logits.
        top_p (float): Keep the smallest set of tokens whose cumulative probability
                       exceeds top_p.

    Returns:
        torch.Tensor: Filtered logits with some values set to -inf.
    """
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_value = values[-1]
        logits = torch.where(
            logits < min_value,
            torch.tensor(-math.inf, device=logits.device),
            logits,
        )

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -math.inf

    return logits


class MCTSNode:
    """
    A node for the Monte Carlo Tree Search (MCTS) in text generation.

    Each node represents a partial sequence of tokens, storing:
      - The action (token) that led to it
      - References to its parent and any children
      - Total reward accumulated and visit counts
      - Model/tokenizer references
      - Other hyperparameters relevant to text generation
    """

    def __init__(
        self,
        parent=None,
        action=None,
        tokenizer=None,
        model=None,
        reward_function=None,
        max_length=20,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        num_rollouts=5,
    ):
        """
        Initialize an MCTSNode.

        Args:
            parent (MCTSNode): The parent node of this node. None if this is the root.
            action (int): The token ID (action) that led to this node from its parent.
            tokenizer: A tokenizer for converting between IDs and text.
            model: A language model (e.g., GPT-2).
            reward_function (callable): A function that takes decoded text and returns a reward float.
            max_length (int): Maximum sequence length for the generated text.
            temperature (float): Sampling temperature for next-token distribution.
            top_k (int): Top-k sampling parameter.
            top_p (float): Top-p (nucleus) sampling parameter.
            num_rollouts (int): Number of rollouts to perform from each node during expansion.
        """
        self.parent = parent
        self.action = action
        self.tokenizer = tokenizer
        self.model = model
        self.reward_function = reward_function
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.num_rollouts = num_rollouts

        # Statistics for MCTS
        self.total_reward = 0.0
        self.visits = 0
        self.children = []
        self.value = 0.0

        # Build up the generated text: inherit from parent or start fresh
        if parent is not None and action is not None:
            self.generated_text = parent.generated_text + [action]
        else:
            self.generated_text = [] if action is None else [action]

        # Check if this node represents a sequence at or beyond max_length
        self._is_terminal = len(self.generated_text) >= self.max_length

    def select_child(self, c_param=1.414):
        """
        Select a child node based on the UCT (Upper Confidence Bound for Trees) formula:
            UCT(child) = Q(child) + c_param * sqrt((log(N)) / n)

        where:
            Q(child) = child.total_reward / child.visits  (exploitation)
            N = sum of visits to all children of the parent
            n = child.visits

        Args:
            c_param (float): Exploration constant.

        Returns:
            MCTSNode: The child node with the highest UCT value.
        """
        # Total visits among siblings
        total_visits = sum(child.visits for child in self.children)

        def uct_value(child):
            if child.visits == 0:
                return math.inf
            exploitation = child.total_reward / child.visits
            exploration = c_param * math.sqrt(
                math.log(total_visits + 1) / (child.visits + 1e-8)
            )
            return exploitation + exploration

        return max(self.children, key=uct_value)

    def rollout(self, temperature, top_k, top_p, look_ahead=3):
        """
        Perform a rollout from this node. We:

          1) Sample 'num_rollouts' possible next tokens from the filtered logits.
          2) For each sampled token, perform a greedy rollout for 'look_ahead' steps.
             We sum up the reward at each step and then average it.

        Args:
            temperature (float): Temperature for sampling.
            top_k (int): Top-k cutoff.
            top_p (float): Top-p cutoff (nucleus sampling).
            look_ahead (int): Number of greedy expansion steps for each sampled token.

        Returns:
            dict: Mapping from token_id -> { "reward": float, "visits": int }
        """
        model = self.model
        tokenizer = self.tokenizer
        reward_function = self.reward_function

        # Ensure the input is placed on the same device as the model
        device = next(model.parameters()).device
        input_ids = torch.tensor([self.generated_text], device=device)

        with torch.no_grad():
            # Get logits for the last token
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :] / temperature

            # Filter logits using top-k and top-p
            filtered_logits = top_k_top_p_filtering(logits[0], top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)

            # Sample num_rollouts tokens from the distribution
            sampled_tokens = torch.multinomial(probs, num_samples=self.num_rollouts)

            # Store the stats for each sampled token
            token_stats = {
                token.item(): {"reward": 0.0, "visits": 0} for token in sampled_tokens
            }

            # Evaluate each sampled token via greedy expansion
            for token in sampled_tokens:
                rollout_reward = 0.0
                # Start with the prompt plus the newly sampled token
                current_text = torch.cat([input_ids[0], token.unsqueeze(0)], dim=0)

                # Greedy expansion for 'look_ahead' steps
                for _ in range(look_ahead):
                    if current_text.shape[0] >= self.max_length:
                        break

                    outputs = model(current_text.unsqueeze(0))
                    next_token_id = outputs.logits[0, -1, :].argmax(dim=-1)
                    current_text = torch.cat([current_text, next_token_id.unsqueeze(0)], dim=0)

                    # Compute reward for the newly formed text
                    decoded_text = tokenizer.decode(current_text, skip_special_tokens=True)
                    step_reward = reward_function(decoded_text)
                    rollout_reward += step_reward

                # Average reward across the look_ahead steps
                avg_reward = rollout_reward / max(1, look_ahead)
                t_id = token.item()
                token_stats[t_id]["reward"] += avg_reward
                token_stats[t_id]["visits"] += 1

            return token_stats

    def expand(self):
        """
        Expand the current node by generating new children based on rollouts.

        For each sampled token during rollout, create a child node and immediately
        backpropagate the obtained reward.
        """
        # Perform the rollout
        token_stats = self.rollout(
            self.temperature, self.top_k, self.top_p, look_ahead=3
        )

        # Create child nodes for each token and backpropagate the average reward
        for token_id, stats in token_stats.items():
            child_node = MCTSNode(
                parent=self,
                action=token_id,
                tokenizer=self.tokenizer,
                model=self.model,
                reward_function=self.reward_function,
                max_length=self.max_length,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                num_rollouts=self.num_rollouts,
            )
            self.children.append(child_node)

            # Update this child with the average reward from rollout
            avg_reward_for_token = stats["reward"]
            child_node._backpropagate(avg_reward_for_token)

    def _backpropagate(self, reward):
        """
        Backpropagate the reward through the tree up to the root node.

        Args:
            reward (float): Reward to propagate.
        """
        node = self
        while node is not None:
            node.total_reward += reward
            node.visits += 1
            node.value = node.total_reward / (node.visits + 1e-8)
            node = node.parent


def mcts_sentence_generator(
    model,
    tokenizer,
    prompt,
    reward_function,
    num_tokens=20,
    iterations=10,
    num_rollouts=5,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    device=None,
):
    """
    Generate text using Monte Carlo Tree Search (MCTS) with a language model.
    Additionally, color-code each newly generated token based on its MCTS node value.

    Args:
        model: A language model (e.g., GPT-2).
        tokenizer: The tokenizer compatible with the model.
        prompt (str): The initial text prompt.
        reward_function (callable): A function that accepts a decoded text and returns a float reward.
        num_tokens (int): Maximum number of new tokens to generate.
        iterations (int): MCTS iterations per token step.
        num_rollouts (int): Number of rollouts each node does during expansion.
        temperature (float): Sampling temperature for next-token distribution.
        top_k (int): Top-k parameter for next-token filtering.
        top_p (float): Top-p parameter for nucleus filtering.
        device (str or torch.device): The torch device to run on (e.g., "cpu" or "cuda").
    
    Returns:
        str: A color-coded string representing the final generated text, where each
             token is colored according to its value.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Encode the initial prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    initial_text_ids = input_ids[0].tolist()

    # Create the MCTS root node
    root = MCTSNode(
        tokenizer=tokenizer,
        model=model,
        reward_function=reward_function,
        max_length=len(initial_text_ids) + num_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_rollouts=num_rollouts,
    )
    # Initialize the root with the prompt's token IDs
    root.generated_text = initial_text_ids

    # Generate tokens using MCTS
    for _ in tqdm(range(num_tokens), desc="Generating tokens"):
        # For each token, we run several MCTS iterations
        for _ in range(iterations):
            node = root

            # 1. Selection: move down the tree until we reach a leaf or terminal node
            while node.children and not node._is_terminal:
                node = node.select_child()

            # 2. Expansion + Simulation (Rollout)
            if not node._is_terminal:
                node.expand()

        # After MCTS iterations, choose the best child from the root to proceed
        if root.children:
            root = root.select_child()
        else:
            # No children means we can't proceed further
            break

    # Reconstruct the token/value path from the final node up to the root
    token_value_pairs = []
    node = root
    while node.parent is not None:
        token_value_pairs.append((node.action, node.value))
        node = node.parent
    token_value_pairs.reverse()  # Because we collected them from leaf -> root

    # Compute min & max values among the generated tokens
    if token_value_pairs:
        all_values = [pair[1] for pair in token_value_pairs]
        min_val, max_val = min(all_values), max(all_values)
    else:
        # If no tokens were generated, just return the prompt
        return prompt

    # Build the color-coded text
    colored_text = prompt  # start with the prompt in plain text
    for (token_id, val) in token_value_pairs:
        color = get_color_for_value(val, min_val, max_val)
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        # Append the color-coded token to the final string (add space for clarity)
        colored_text += " " + color + token_str + Style.RESET_ALL

    return colored_text
