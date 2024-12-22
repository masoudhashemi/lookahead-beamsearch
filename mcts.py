import math
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama so that ANSI color codes work across platforms
init(autoreset=True)

# Create a dedicated logger for MCTS
logger = logging.getLogger("MCTSLogger")
logger.setLevel(logging.INFO)

def get_color_for_value(value, min_val, max_val):
    """
    Return a color based on the node value, normalized between min_val and max_val.
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
    """
    # -- Top-k filtering
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_value = values[-1]
        logits = torch.where(
            logits < min_value,
            torch.tensor(-math.inf, device=logits.device),
            logits,
        )

    # -- Top-p (nucleus) filtering
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
    A node for Monte Carlo Tree Search in text generation.
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
        look_ahead=3,
        verbose=False,
    ):
        """
        Initialize an MCTSNode.
        
        If `verbose=True`, the logger level for this node is set to INFO; 
        otherwise, it's set to WARNING.
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
        self.look_ahead = look_ahead

        # MCTS statistics
        self.total_reward = 0.0
        self.visits = 0
        self.children = []
        self.value = 0.0

        # Build up the generated text (inherit from parent or start fresh)
        if parent is not None and action is not None:
            self.generated_text = parent.generated_text + [action]
        else:
            self.generated_text = [] if action is None else [action]

        # Terminal if we reached max_length
        self._is_terminal = len(self.generated_text) >= self.max_length

        # We'll store a local logger level if needed
        if verbose:
            self._logger_level = logging.INFO
        else:
            self._logger_level = logging.WARNING

    def select_child(self, c_param=1.414):
        """
        Select a child using the UCB (Upper Confidence Bound) formula:
          UCT(child) = Q(child) + c_param * sqrt((log(N)) / (n + 1e-8))
        where:
          Q(child) = child.total_reward / child.visits
          N = sum of visits to all children of the parent
          n = child.visits
        """
        total_visits = sum(child.visits for child in self.children)

        def uct_value(child):
            if child.visits == 0:
                return float('inf')
            exploitation = child.total_reward / child.visits
            exploration = c_param * math.sqrt(
                math.log(total_visits + 1) / (child.visits + 1e-8)
            )
            return exploitation + exploration

        selected_child = max(self.children, key=uct_value)

        logger.log(
            self._logger_level,
            f"[SelectChild] Chose child token={selected_child.action}, "
            f"UCT value={uct_value(selected_child):.3f}, visits={selected_child.visits}, "
            f"current node text='{self.tokenizer.decode(self.generated_text, skip_special_tokens=True)}'"
        )

        return selected_child

    def rollout(self):
        """
        Perform a short rollout from this node by:
          1) Sampling 'num_rollouts' possible tokens from the distribution.
          2) For each sampled token, greedily expand 'look_ahead' steps.
          3) Compute reward for the final text.

        Returns a dict: 
            token_id -> { "reward": float, "visits": int }
        """
        model = self.model
        tokenizer = self.tokenizer
        reward_function = self.reward_function

        device = next(model.parameters()).device
        input_ids = torch.tensor([self.generated_text], device=device)

        with torch.no_grad():
            # Get next-token logits
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :] / self.temperature
            filtered_logits = top_k_top_p_filtering(logits[0], top_k=self.top_k, top_p=self.top_p)
            probs = F.softmax(filtered_logits, dim=-1)

            # Sample multiple tokens
            sampled_tokens = torch.multinomial(probs, num_samples=self.num_rollouts)

            token_stats = {token.item(): {"reward": 0.0, "visits": 0} for token in sampled_tokens}

            for token in sampled_tokens:
                rollout_reward = 0.0
                current_text = torch.cat([input_ids[0], token.unsqueeze(0)], dim=0)

                # Greedy expand for look_ahead steps
                for _ in range(self.look_ahead):
                    if current_text.shape[0] >= self.max_length:
                        break
                    outputs = model(current_text.unsqueeze(0))
                    next_token_id = outputs.logits[0, -1, :].argmax(dim=-1)
                    current_text = torch.cat([current_text, next_token_id.unsqueeze(0)], dim=0)

                decoded_text = tokenizer.decode(current_text, skip_special_tokens=True)
                rollout_reward = reward_function(decoded_text)

                t_id = token.item()
                token_stats[t_id]["reward"] += rollout_reward
                token_stats[t_id]["visits"] += 1

                # Log each rollout if needed
                logger.log(
                    self._logger_level,
                    f"[Rollout] Sampled token='{tokenizer.decode([t_id], skip_special_tokens=True)}' "
                    f"(ID={t_id}), Reward={rollout_reward:.3f}, "
                    f"Partial seq='{decoded_text[:60]}{'...' if len(decoded_text)>60 else ''}'"
                )

        return token_stats

    def expand(self):
        """
        Expand this node by sampling multiple tokens (rollout) and creating child nodes.
        Then backpropagate the average reward for each newly created child.
        """
        # If already terminal, no expansion
        if self._is_terminal:
            return

        token_stats = self.rollout()  # {token_id: {"reward": x, "visits": y}, ...}

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
                look_ahead=self.look_ahead,
                verbose=(self._logger_level == logging.INFO),
            )
            self.children.append(child_node)

            avg_reward = stats["reward"] / max(stats["visits"], 1)
            child_node._backpropagate(avg_reward)

            # Log child creation if needed
            logger.log(
                self._logger_level,
                f"[Expand] Created child token='{self.tokenizer.decode([token_id], skip_special_tokens=True)}' "
                f"(ID={token_id}), Avg reward={avg_reward:.3f}, child value={child_node.value:.3f}"
            )

    def _backpropagate(self, reward):
        """
        Backpropagate a reward up the chain to the root.
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
    look_ahead=5,
    device=None,
    verbose=False,
):
    """
    Generate text using MCTS with rollouts and color the tokens by their final MCTS value.
    Uses Python's logging library for logging messages.
    """
    # Configure global logging level based on 'verbose'
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Encode the initial prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    initial_text_ids = input_ids[0].tolist()

    # Root node for MCTS
    root = MCTSNode(
        tokenizer=tokenizer,
        model=model,
        reward_function=reward_function,
        max_length=len(initial_text_ids) + num_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_rollouts=num_rollouts,
        look_ahead=look_ahead,
        verbose=verbose
    )
    # Initialize the root with the prompt tokens
    root.generated_text = initial_text_ids

    for step_idx in tqdm(range(num_tokens), desc="Generating tokens"):
        logger.log(
            root._logger_level,
            f"[MCTS] Step {step_idx+1}/{num_tokens} - current text: "
            f"'{tokenizer.decode(root.generated_text, skip_special_tokens=True)}'"
        )

        # Run MCTS iterations from the current root
        for i in range(iterations):
            node = root

            # 1. Selection: traverse down to a leaf
            while node.children and not node._is_terminal:
                node = node.select_child()
            
            # 2. Expansion + Simulation
            if not node._is_terminal:
                node.expand()

            logger.log(
                root._logger_level,
                f"[MCTS Iteration] Expanded leaf node, text='{tokenizer.decode(node.generated_text, skip_special_tokens=True)}', "
                f"children={len(node.children)}, node value={node.value:.3f}"
            )

        # After 'iterations' expansions, pick the child with the highest value (greedy)
        if root.children:
            best_child = max(root.children, key=lambda c: c.value)
            logger.log(
                root._logger_level,
                f"[MCTS] Best child chosen token ID={best_child.action}, value={best_child.value:.3f}, "
                f"New partial text='{tokenizer.decode(best_child.generated_text, skip_special_tokens=True)}'"
            )
            root = best_child
        else:
            logger.log(root._logger_level, "[MCTS] No children from the root; stopping generation.")
            break

        if root._is_terminal:
            logger.log(root._logger_level, "[MCTS] Reached max_length, stopping generation.")
            break

    # Collect the generated tokens from the final node
    token_value_pairs = []
    node = root
    while node.parent is not None:
        token_value_pairs.append((node.action, node.value))
        node = node.parent
    token_value_pairs.reverse()

    # If no tokens were actually generated, just return the prompt
    if not token_value_pairs:
        return prompt

    # Determine min/max value for color scaling
    all_values = [pair[1] for pair in token_value_pairs]
    min_val, max_val = min(all_values), max(all_values)

    # Build color-coded final text
    colored_text = prompt
    for (token_id, val) in token_value_pairs:
        color = get_color_for_value(val, min_val, max_val)
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        colored_text += " " + color + token_str + Style.RESET_ALL

    logger.log(root._logger_level, f"[MCTS] Final generated text: '{colored_text}'")

    return colored_text
