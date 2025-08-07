import argparse
import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def preprocess_and_save(n_samples, output_path, random_seed=42):
    """
    Loads TriviaQA from the Hugging Face Hub, subsamples it,
    reformats it, and saves it to a CSV file.
    """
    print("Loading 'trivia_qa' dataset from Hugging Face Hub...")
    # Using the 'rc.nocontext' configuration which is suitable for question answering
    try:
        dataset = load_dataset("trivia_qa", "rc.nocontext", trust_remote_code=True)['validation']
    except Exception as e:
        print(f"Failed to load dataset. It might be temporarily unavailable or require a newer 'datasets' version.")
        print(f"Error: {e}")
        return

    print(f"Shuffling and selecting {n_samples} samples with random seed {random_seed}...")
    # Shuffle for randomness and select the desired number of samples
    subsampled_dataset = dataset.shuffle(seed=random_seed).select(range(n_samples))

    print("Reformatting data to '(raw_question, correct_answer)' format...")
    questions = []
    correct_answers = []

    # The 'answer' field is a dictionary containing 'value', 'aliases', etc.
    # We will use 'aliases' as it provides a list of valid answer variations.
    for item in tqdm(subsampled_dataset, desc="Processing samples"):
        questions.append(item['question'])
        # The aliases are the most comprehensive list of correct answers
        correct_answers.append(str(item['answer']['aliases']))

    # Create the final DataFrame in the desired format
    df = pd.DataFrame({
        'raw_question': questions,
        'correct_answer': correct_answers
    })

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to the specified CSV file path
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nSuccessfully saved {n_samples} samples to '{output_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Download, subsample, and reformat the TriviaQA dataset."
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=20000,
        help="The number of random samples to select from the dataset."
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='datasets/triviaqa_20k.csv',
        help="The path to save the final processed CSV file."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for shuffling to ensure reproducibility."
    )
    args = parser.parse_args()

    preprocess_and_save(args.n_samples, args.output_file, args.seed)
