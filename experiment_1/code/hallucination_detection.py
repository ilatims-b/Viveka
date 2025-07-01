import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# You will need these utility functions from your project
from probing_utils import extract_internal_reps_specific_layer_and_token, load_model_and_validate_gpu

import argparse

# --- Step 1: Parse Arguments ---
parser = argparse.ArgumentParser(description="Run probing experiment for Gemma-2B")

parser.add_argument('--pkl_path', type=str, required=True,
                    help='Relative path to the .pkl file under ../checkpoints/')
parser.add_argument('--probe_token', type=str, required=True,
                    help='Token to probe in the input sequence')
parser.add_argument('--probe_at', type=str, required=True,
                    help='Component to probe at (e.g., attention_output, mlp, etc.)')
parser.add_argument('--probe_layer', type=int, default=24,
                    help='Layer to probe at (default: 24)')
parser.add_argument('--model_name', type=str, default='google/gemma-2-2b-it',
                    help='Name or path of the Hugging Face model (default: google/gemma-2-2b-it)')

args = parser.parse_args()

# --- Step 2: Assign Parameters ---
MODEL_NAME = args.model_name
PKL_FILE_PATH = f"../checkpoints/{args.pkl_path}"
PROBE_LAYER = args.probe_layer
PROBE_TOKEN = args.probe_token
PROBE_AT = args.probe_at

print("Loading the base language model (e.g., Gemma)...")
model, tokenizer = load_model_and_validate_gpu(MODEL_NAME)

print(f"Loading the probe from {PKL_FILE_PATH}...")
with open(PKL_FILE_PATH, 'rb') as f:
    probe_clf = pickle.load(f)

import torch
import numpy as np

# --- Step 2: Prepare Your New TriviaQA-style Questions ---
print("Extracting hidden representations for TriviaQA-style data...")

questions_list = [
    "What is the capital of France?",
    "Who painted the Mona Lisa?",
    "What is the smallest planet in our solar system?",
    "Which element has the chemical symbol 'O'?",
    "What year did World War II end?",
    "Who discovered penicillin?",
    "What is the tallest mountain in the world?",
    "In which country is the Great Pyramid of Giza located?",
    "Who wrote 'Pride and Prejudice'?",
    "What is the square root of 144?"
]

answers_list = [
    "The capital of France is Paris.",
    "The Mona Lisa was painted by Leonardo da Vinci.",
    "The smallest planet in our solar system is Mercury.",
    "The chemical symbol 'O' stands for Oxygen.",
    "World War II ended in 1945.",
    "Penicillin was discovered by Alexander Fleming.",
    "The tallest mountain in the world is Mount Everest.",
    "The Great Pyramid of Giza is located in Egypt.",
    "Pride and Prejudice was written by Jane Austen.",
    "The square root of 144 is 12."
]

# --- Step 3: Extract Internal Representations ---
input_output_ids = []

for question, answer in zip(questions_list, answers_list):
    q_ids = tokenizer([question], return_tensors='pt').input_ids[0]
    a_ids = tokenizer([answer], return_tensors='pt').input_ids[0]
    combined_ids = torch.cat((q_ids, a_ids[1:]), dim=0)
    input_output_ids.append(combined_ids)

# Dummy labels (for compatibility)
dummy_labels = [1] * len(questions_list)

# Extract hidden representations
hidden_vectors = extract_internal_reps_specific_layer_and_token(
    model,
    tokenizer,
    questions_list,
    input_output_ids,
    PROBE_AT,
    MODEL_NAME,
    PROBE_LAYER,
    PROBE_TOKEN,
    answers_list,
    dummy_labels
)

X_new = np.array(hidden_vectors)

# --- Step 4: Make Predictions ---
print("\n--- Inference Results ---")
predicted_classes = probe_clf.predict(X_new)
probabilities = probe_clf.predict_proba(X_new)

for i in range(len(questions_list)):
    print(f"\nQ: {questions_list[i]}")
    print(f"A: {answers_list[i]}")
    print(f"Predicted Class: {predicted_classes[i]}")
    print(f"Confidence in correctness: {probabilities[i][1]:.2%}")
    if predicted_classes[i] == 1:
        print("Conclusion: The probe believes the answer is likely NOT a hallucination.")
    else:
        print("Conclusion: The probe believes the answer is likely a hallucination.")
