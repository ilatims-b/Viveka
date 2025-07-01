import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# You will need these utility functions from your project
from probing_utils import extract_internal_reps_specific_layer_and_token, load_model_and_validate_gpu

# --- Step 1: Define Parameters and Load Models ---

MODEL_NAME = 'google/gemma-2-2b-it'
pkl_path = input('enter pkl path:')
PKL_FILE_PATH = f"../checkpoints/{pkl_path}"
PROBE_LAYER = 24
probe_token = input('enter probe token:')
PROBE_TOKEN = f"{probe_token}"

# --- CORRECTED PARAMETER ---
# This value MUST be one of the keys from the LAYERS_TO_TRACE dictionary you provided.
# 'attention_output' is a common and likely choice. If you trained with a different
# --probe_at argument (e.g., 'mlp'), change this value to match.
probe_at = input('where should we probe at:')
PROBE_AT = f'{probe_at}'
# --- END OF CORRECTION ---


print("Loading the base language model (e.g., Gemma)...")
model, tokenizer = load_model_and_validate_gpu(MODEL_NAME)

print(f"Loading the probe from {PKL_FILE_PATH}...")
with open(PKL_FILE_PATH, 'rb') as f:
    probe_clf = pickle.load(f)

# --- Step 2: Prepare Your New Data (Unchanged) ---
new_question = "What is 5 + 2?"
new_model_answer = "The answer is 7."

# --- Step 3: Extract the Internal Representation ---
print("Extracting hidden representation for the new data point...")

questions_list = [new_question]
answers_list = [new_model_answer]

question_tokens_tensor = tokenizer(questions_list, return_tensors='pt').input_ids
answer_tokens_tensor = tokenizer(answers_list, return_tensors='pt').input_ids

question_ids_1d = question_tokens_tensor[0]
answer_ids_1d = answer_tokens_tensor[0]
combined_ids = torch.cat((question_ids_1d, answer_ids_1d[1:]), dim=0)
input_output_ids = [combined_ids]

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
    [1]
)

X_new = np.array(hidden_vectors)

# --- Step 4: Make a Prediction (Unchanged) ---
print("Making a prediction with the loaded probe...")

probabilities = probe_clf.predict_proba(X_new)
predicted_class = probe_clf.predict(X_new)
prob_of_correctness = probabilities[0][1]

print("\n--- Inference Results ---")
print(f"Question: '{new_question}'")
print(f"Model Answer: '{new_model_answer}'")
print(f"Predicted Class: {predicted_class[0]}")
print(f"Probe's confidence in correctness: {prob_of_correctness:.2%}")

if predicted_class[0] == 1:
    print("Conclusion: The probe believes the answer is likely NOT a hallucination.")
else:
    print("Conclusion: The probe believes the answer is likely a hallucination.")