from typing import Callable

import typer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from mcts import mcts_sentence_generator
from rewards import get_toxicity_score, preplexity_eval
from typicality import generate_with_typical_set_sampling

app = typer.Typer()

# Load a pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer_toxicity = AutoTokenizer.from_pretrained(model_name)
model_toxicity = AutoModelForSequenceClassification.from_pretrained(model_name)


no_toxicity_reward = lambda text: get_toxicity_score(
    text, model_toxicity, tokenizer_toxicity
)
toxicity_reward = lambda text: 1 - get_toxicity_score(
    text, model_toxicity, tokenizer_toxicity
)

# Function to calculate perplexity
def perplexity_reward(text):
    results = preplexity_eval(model, tokenizer, text)
    return results


rewards_ = {
    "no-toxicity": no_toxicity_reward,
    "toxicity": toxicity_reward,
    "perplexity": perplexity_reward,
}


def _generate_with_typical_set_sampling(
    prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_length: int,
    reward_function: str,
    mcts_iterations: int,
    typc_epsilon: float,
) -> str:
    return generate_with_typical_set_sampling(
        model,
        tokenizer,
        prompt,
        max_length=max_length,
        temperature=temperature,
        epsilon=typc_epsilon,
    )


def _mcts_sentence_generator(
    prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_length: int,
    reward_function: str,
    mcts_iterations: int,
    typc_epsilon: float,
) -> str:

    return mcts_sentence_generator(
        model,
        tokenizer,
        prompt,
        reward_function=reward_function,
        num_tokens=max_length,
        iterations=mcts_iterations,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


def main(
    prompt: str,
    generator: Callable,
    temperature: float = 1,
    top_p: float = 1,
    top_k: int = 10,
    num_generations: int = 1,
    max_length: int = 20,
    reward_function: str = "toxicity",
    mcts_iterations: int = 5,
    typc_epsilon: float = 0.1,
):

    reward_function_ = rewards_[reward_function]
    # Execute the specified text generator
    generated_texts = []
    for _ in range(num_generations):
        generated_text = generator(
            prompt,
            temperature,
            top_p,
            top_k,
            max_length,
            reward_function_,
            mcts_iterations,
            typc_epsilon,
        )
        generated_texts.append(generated_text)

    # Print out the generated texts
    for text in generated_texts:
        typer.echo(text)


@app.command()
def run(
    prompt: str = "One upon a tim",
    generator: str = "typical_set",
    temperature: float = 1,
    top_p: float = 1,
    top_k: int = 10,
    num_generations: int = 1,
    max_length: int = 20,
    reward_function: str = "toxicity",
    mcts_iterations: int = 5,
    typc_epsilon: float = 0.1,
):
    """
    Generate text using a specified generator and set of parameters.
    """
    # Select the appropriate text generator based on the specified argument
    generator_func = (
        _generate_with_typical_set_sampling
        if generator == "typical_set"
        else _mcts_sentence_generator
    )

    # Call the main function with the specified arguments
    main(
        prompt,
        generator_func,
        temperature,
        top_p,
        top_k,
        num_generations,
        max_length,
        reward_function,
        mcts_iterations,
        typc_epsilon,
    )


if __name__ == "__main__":
    app()
