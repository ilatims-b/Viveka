import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import set_seed

# Assume these utils are in a separate file or defined elsewhere in your project
from compute_correctness import compute_correctness
from probing_utils import (LIST_OF_DATASETS, LIST_OF_MODELS,
                           MODEL_FRIENDLY_NAMES, generate,
                           load_model_and_validate_gpu, tokenize)


def parse_args():
    parser = argparse.ArgumentParser(description="A script for generating model answers and outputting to csv")
    parser.add_argument("--model",
                        choices=LIST_OF_MODELS,
                        required=True)
    parser.add_argument("--dataset",
                        choices=LIST_OF_DATASETS)
    parser.add_argument("--verbose", action='store_true', help='print more information')
    parser.add_argument("--n_samples", type=int, help='number of examples to use', default=None)
    parser.add_argument("--train_size", type=int,help='train size of datasets',default=2000)

    return parser.parse_args()


# ==============================================================================
# DATA LOADING FUNCTIONS (Unchanged from your original script)
# ==============================================================================
def load_data_movies(test=False):
    file_name = 'movie_qa'
    if test:
        file_path = f'../data/{file_name}_test.csv'
    else: # train
        file_path = f'../data/{file_name}_train.csv'
    if not os.path.exists(file_path):
        data = pd.read_csv(f"../data/{file_name}.csv")
        train, test = train_test_split(data, train_size=10000, random_state=42)
        train.to_csv(f"../data/{file_name}_train.csv", index=False)
        test.to_csv(f"../data/{file_name}_test.csv", index=False)
    data = pd.read_csv(file_path)
    return data['Question'], data['Answer']

def load_data_nli(split, data_file_names):
    data_folder = '../data'
    file_path = f"{data_folder}/{data_file_names[split]}.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = pd.read_csv(file_path)
    return data['Question'], data['Answer'], data['Origin']

def load_data_snli(split):
    return load_data_nli(split, {'train': 'snli_train', 'test': 'snli_validation'})

def load_data_mnli(split):
    return load_data_nli(split, {'train': 'mnli_train', 'test': 'mnli_validation'})

def load_data_nq(split, with_context=False):
    file_name = 'nq_wc'
    file_path = f'../data/{file_name}_dataset_{split}.csv'
    if not os.path.exists(file_path):
        all_data = pd.read_csv(f"../data/{file_name}_dataset.csv")
        train, test = train_test_split(all_data, train_size=10000, random_state=42)
        train.to_csv(f"../data/{file_name}_dataset_train.csv", index=False)
        test.to_csv(f"../data/{file_name}_dataset_test.csv", index=False)
    data = pd.read_csv(file_path)
    context_data = data['Context'] if with_context else None
    return data['Question'], data['Answer'], context_data

def load_data_winogrande(split):
    file_path = f"../data/winogrande_{split}.csv"
    if not os.path.exists(file_path):
        all_data = pd.read_csv(f"../data/winogrande.csv")
        train, test = train_test_split(all_data, train_size=10000, random_state=42)
        train.to_csv(f"../data/winogrande_train.csv", index=False)
        test.to_csv(f"../data/winogrande_test.csv", index=False)
    data = pd.read_csv(file_path)
    return data['Question'], data['Answer'], data['Wrong_Answer']

def load_data_triviaqa(test=False, legacy=False):
    args = parse_args() # To get train_size
    if test:
        file_path = '../data/triviaqa-unfiltered/unfiltered-web-dev.json'
    else:
        file_path = '../data/triviaqa-unfiltered/unfiltered-web-train.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)['Data']
    data, _ = train_test_split(data, train_size=args.train_size, random_state=42)
    return [ex['Question'] for ex in data], [ex['Answer']['Aliases'] for ex in data]

def load_data_math(test=False):
    file_name = "AnswerableMath_test.csv" if test else "AnswerableMath.csv"
    data = pd.read_csv(f"../data/{file_name}")
    answers = data['answer'].map(lambda x: eval(x)[0])
    return data['question'], answers

def load_data_imdb(split):
    dataset = load_dataset("imdb")
    indices = np.arange(len(dataset[split]))
    np.random.shuffle(indices)
    subset = dataset[split].select(indices[:10000])
    return subset['text'], subset['label']

def load_winobias(dev_or_test):
    data = pd.read_csv(f'../data/winobias_{dev_or_test}.csv')
    return (data['sentence'], data['q'], data['q_instruct']), data['answer'], data['incorrect_answer'], data['stereotype'], data['type']

def load_hotpotqa(split, with_context):
    dataset = load_dataset("hotpot_qa", 'distractor')
    # Use a fixed seed for reproducibility of the random subset
    np.random.seed(42)
    subset_indices = np.random.choice(len(dataset[split]), 10000, replace=False)
    
    questions = [dataset[split][int(x)]['question'] for x in subset_indices]
    labels = [dataset[split][int(x)]['answer'] for x in subset_indices]

    if with_context:
        prompts = []
        for idx in subset_indices:
            context_text = ""
            for sentences in dataset[split][int(idx)]['context']['sentences']:
                context_text += " ".join(sentences) + "\n"
            prompts.append(context_text.strip() + "\n" + dataset[split][int(idx)]['question'])
        questions = prompts
        
    return questions, labels

def load_data(dataset_name):
    """Master data loading function."""
    # This dispatcher is simplified for clarity, assuming original script's logic
    max_new_tokens = 100
    context, origin, stereotype, type_, wrong_labels = None, None, None, None, None
    
    if dataset_name == 'triviaqa':
        all_questions, labels = load_data_triviaqa(False)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'math':
        all_questions, labels = load_data_math(test=False)
        preprocess_fn = math_preprocess
        max_new_tokens = 200
    elif dataset_name == 'winogrande':
        all_questions, labels, wrong_labels = load_data_winogrande('train')
        preprocess_fn = winogrande_preprocess
    elif dataset_name == 'natural_questions_with_context':
        all_questions, labels, context = load_data_nq('train', with_context=True)
        preprocess_fn = nq_preprocess
    # Add other datasets from your script here...
    else:
        raise TypeError(f"Data type '{dataset_name}' is not supported in this example")
        
    return all_questions, context, labels, max_new_tokens, origin, preprocess_fn, stereotype, type_, wrong_labels


# ==============================================================================
# PREPROCESSING FUNCTIONS (Unchanged from your original script)
# ==============================================================================
def triviqa_preprocess(model_name, all_questions, labels):
    prompts = []
    if 'instruct' in model_name.lower():
        return all_questions
    else:
        for q in all_questions:
            prompts.append(f'''Q: {q}\nA:''')
        return prompts

def math_preprocess(model_name, all_questions, labels):
    if 'instruct' in model_name.lower():
        return [q + " Answer shortly." for q in all_questions]
    else:
        return [f"Q: {q}\nA:" for q in all_questions]

def winogrande_preprocess(model_name, all_questions, labels):
    if 'instruct' in model_name.lower():
        return all_questions
    new_questions = []
    q1, label1 = None, None
    for q, label in zip(all_questions, labels):
        q_base = q.split("Who does the blank refer to in the sentence?")[0].split("What does the blank refer to in the sentence?")[0]
        if q1 is None:
            q1, label1 = q_base, label
        q_with_ex = f"{q1} The blank refers to: {label1}\n{q_base} The blank refers to:"
        new_questions.append(q_with_ex)
    return new_questions
    
def nq_preprocess(model_name, all_questions, labels, with_context, context):
    prompts = []
    if with_context:
        for q, c in zip(all_questions, context):
            prompt_text = f"Context: {c}\n\nQ: {q}\nA:" if 'instruct' not in model_name.lower() else f"{c}\n{q}"
            prompts.append(prompt_text)
    else:
        prompts = triviqa_preprocess(model_name, all_questions, labels)
    return prompts


# ==============================================================================
# CORE GENERATION AND ACTIVATION EXTRACTION LOGIC
# ==============================================================================

def generate_model_answers(data, model, tokenizer, device, model_name, do_sample=False,
                           temperature=1.0, top_p=1.0, max_new_tokens=100, stop_token_id=None, verbose=False):
    """
    Generates answers and returns textual answers and the full token IDs for prompts + answers.
    """
    all_textual_answers = []
    all_input_output_ids = []
    
    for prompt in tqdm(data, desc="Generating Answers"):
        model_input = tokenize(prompt, tokenizer, model_name).to(device)
        with torch.no_grad():
            # Note: We are NOT using output_scores or output_hidden_states here to keep generation fast.
            # We will get the activations in a separate step later.
            model_output = generate(model_input, model, model_name, do_sample,
                                    output_scores=False, # Set to False for speed
                                    max_new_tokens=max_new_tokens,
                                    top_p=top_p, temperature=temperature,
                                    stop_token_id=stop_token_id, tokenizer=tokenizer)

        # Decode only the generated part for the textual answer
        answer = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):], skip_special_tokens=True)
        all_textual_answers.append(answer)
        
        # Save the full sequence of token IDs (prompt + answer)
        all_input_output_ids.append(model_output['sequences'][0].cpu())

    return all_textual_answers, all_input_output_ids


def get_final_residual_stream(model, all_input_output_ids, device):
    """
    NEW FUNCTION
    Performs a single forward pass for each completed (prompt + answer) sequence
    to get the final, full residual stream activations.
    """
    all_activations = []
    model.eval() # Ensure model is in evaluation mode

    for full_ids in tqdm(all_input_output_ids, desc="Extracting Activations"):
        # Move token IDs to the correct device
        input_ids = full_ids.to(device).unsqueeze(0) # Add batch dimension
        
        with torch.no_grad():
            # Perform a forward pass to get hidden states
            outputs = model(input_ids, output_hidden_states=True)
        
        # The 'hidden_states' is a tuple of (num_layers + 1) tensors.
        # Index 0 is the embedding output, 1 to N are the layer outputs.
        # We stack the outputs from all transformer layers.
        # Shape of each layer's output: (batch_size, sequence_length, hidden_dim)
        residual_stream = torch.stack([
            s.squeeze(0).cpu().to(torch.float16) for s in outputs.hidden_states[1:]
        ])
        # Final shape for one prompt: (num_layers, sequence_length, hidden_dim)
        all_activations.append(residual_stream)
        
    return all_activations


# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================

def init_wandb(args):
    cfg = vars(args)
    wandb.init(
        project="generate_answers_with_activations",
        config=cfg
    )

def main():
    args = parse_args()
    init_wandb(args)
    set_seed(42)
    dataset_size = args.n_samples

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_validate_gpu(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stop_token_id = None
    if 'instruct' not in args.model.lower():
        stop_token_id = tokenizer.encode('\n', add_special_tokens=False)[-1]

    print(f"Loading dataset: {args.dataset}...")
    all_questions, context, labels, max_new_tokens, origin, preprocess_fn, stereotype, type_, wrong_labels = load_data(args.dataset)

    if not os.path.exists('../output'):
        os.makedirs('../output')

    # Define output file paths
    model_name_safe = MODEL_FRIENDLY_NAMES[args.model]
    file_path_answers = f"../output/{model_name_safe}-answers-{args.dataset}.csv"
    file_path_activations = f"../output/{model_name_safe}-residual_activations-{args.dataset}.pt"

    if dataset_size:
        # Safely slice all relevant data lists
        all_questions = all_questions[:dataset_size]
        labels = labels[:dataset_size]
        if origin is not None: origin = origin[:dataset_size]
        if wrong_labels is not None: wrong_labels = wrong_labels[:dataset_size]
        if context is not None: context = context[:dataset_size]
        if stereotype is not None: stereotype = stereotype[:dataset_size]
        if type_ is not None: type_ = type_[:dataset_size]


    print("Preprocessing prompts...")
    output_csv = {'raw_question': all_questions}
    
    if 'natural_questions' in args.dataset:
        with_context = 'with_context' in args.dataset
        prompts = preprocess_fn(args.model, all_questions, labels, with_context, context)
    else:
        prompts = preprocess_fn(args.model, all_questions, labels)
    
    # --- Step 1: Generate answers and get the full token IDs ---
    model_answers, input_output_ids = generate_model_answers(
        prompts, model, tokenizer, device, args.model,
        max_new_tokens=max_new_tokens, stop_token_id=stop_token_id
    )

    # --- Step 2: Get final residual stream activations ---
    all_residual_activations = get_final_residual_stream(model, input_output_ids, device)

    print("Computing correctness...")
    res = compute_correctness(prompts, args.dataset, args.model, labels, model, model_answers, tokenizer, wrong_labels)
    correctness = res['correctness']
    acc = np.mean(correctness)
    wandb.summary[f'accuracy'] = acc
    print(f"Accuracy: {acc:.4f}")

    # --- Step 3: Save all outputs ---
    output_csv['question_prompt'] = prompts
    output_csv['model_answer'] = model_answers
    output_csv['correct_answer'] = labels
    output_csv['automatic_correctness'] = correctness
    # (Add other optional fields to CSV as in your original script)
    
    print(f"Saving answers and metadata to {file_path_answers}...")
    pd.DataFrame.from_dict(output_csv).to_csv(file_path_answers, index=False)

    print(f"Saving residual stream activations to {file_path_activations}...")
    # This saves the list of tensors. Each element corresponds to a prompt.
    torch.save(all_residual_activations, file_path_activations)
    
    print("Script finished successfully!")

if __name__ == "__main__":
    main()