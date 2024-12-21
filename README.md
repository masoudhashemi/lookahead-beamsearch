# Look ahead Beam Search for HuggingFace Transformers

This code uses a Monte Carlo Tree Search algorithm to find the best next token in a sequence. For each token it generates few rollouts and computes the reward and the token with the highest reward is chosen.

As a baseline a beam search code with typical sampling is also provided.

To run the code with a typical beam search run:

``` bash
python3 main.py
```

To run the code with a look ahead beam search run, with no toxicity reward function:

``` bash
python main.py --num-generations 5 --generator mcts --reward_function no-toxicity
```

To run the code with a look ahead beam search run, with toxicity reward function:

``` bash
python main.py --num-generations 5 --generator mcts --reward_function toxicity
```

## Updates

Critical tokens matter [paper](https://arxiv.org/pdf/2411.19943) explores how individual tokens within an LLM's reasoning process can significantly impact the accuracy of its final output. The authors identify "critical tokens" that, when generated, often lead to incorrect reasoning trajectories. To address this, they propose a new method called cDPO that automatically identifies these critical tokens by comparing the generation likelihoods of models fine-tuned on positive and negative reasoning examples.  By using this contrastive estimation approach, cDPO provides token-level rewards during the learning process, effectively guiding the LLM towards more accurate reasoning paths. Experiments on benchmark datasets show that cDPO improves the reasoning capabilities of various LLMs, including Llama-3 and deepseek-math.

We can use the MCTS code in this repo to generate outputs with a high reward and for each token a value is assigne 
