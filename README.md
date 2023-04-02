# Look ahead Beam Search for HuggingFace Transformers
This code uses a Monte Carlo Tree Search algorithm to find the best next token in a sequence. For each token it generates few rollouts and computes the reward and the token with the highest reward is chosen.

As a baseline a beam search code with typical sampling is also provided.

To run the code with a typical beam search run:
``` bash
python3 main.py
```

To run the code with a look ahead beam search run:
``` bash
python main.py --num-generations 5 --generator mcts
```
