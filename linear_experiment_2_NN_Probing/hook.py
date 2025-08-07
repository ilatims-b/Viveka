from utils import (create_prompts, generate_model_answers)
import json
import os
from tqdm import tqdm
import torch as t
from thefuzz import fuzz


# The Hook class is a utility to capture intermediate activations from a model layer.
class Hook:
    """A simple hook class to store the output of a layer."""
    def __init__(self):
        self.out = None
    def __call__(self, module, module_inputs, module_outputs):
        # The output of a model layer can be a tuple. We are interested in the first element.
        self.out = module_outputs[0] if isinstance(module_outputs, tuple) else module_outputs

def generate_and_label_answers(
    statements,
    correct_answers,
    tokenizer,
    model,
    device,
    num_generations=30,
    output_dir="probes_data"
):
    """
    STAGE 1: Generate answers for all statements, label them using fuzzy matching,
    and save the results to a single comprehensive JSON cache file. This function
    should be run once for the entire dataset.
    """
    model_name = model.name_or_path.replace("/", "_") if hasattr(model, 'name_or_path') else 'unknown'
    generations_dir = os.path.join(output_dir, "generations")
    os.makedirs(generations_dir, exist_ok=True)
    generations_cache_path = os.path.join(generations_dir, f"{model_name}_generations.json")

    # Load existing cache to resume progress, or create a new one.
    if os.path.exists(generations_cache_path):
        with open(generations_cache_path, 'r', encoding='utf-8') as f:
            generations_cache = json.load(f)
    else:
        generations_cache = {}

    # Process each statement in the dataset.
    for stmt, correct_ans in tqdm(
        zip(statements, correct_answers),
        desc="Stage 1: Generating and Labeling Answers",
        total=len(statements)
    ):
        # If the statement is already in our cache, we can skip it.
        if stmt in generations_cache:
            continue

        # Create the appropriate prompt format for the model.
        prompt = create_prompts([stmt], model_name)[0]

        # Generate `num_generations` different answers for the same prompt.
        generated_texts = []
        for _ in range(num_generations):
            answer_raw, _ = generate_model_answers(
                [prompt], model, tokenizer, device, model_name, max_new_tokens=64
            )
            generated_texts.append(answer_raw[0].strip())

        # --- Fuzzy String Matching for Labeling ---
        # First, safely parse the ground truth answer string into a list.
        try:
            correct_answers_list = eval(correct_ans)
            if not isinstance(correct_answers_list, list):
                correct_answers_list = [str(correct_answers_list)]
        except (SyntaxError, NameError):
            correct_answers_list = [str(correct_ans)]

        # Compare each generated answer to the list of correct answers.
        ground_truth_labels = []
        for text in generated_texts:
            # A generated answer is considered correct if it has a high fuzzy match score
            # with any of the possible correct answers.
            is_match = any(fuzz.partial_ratio(str(ans).lower(), text.lower()) > 90 for ans in correct_answers_list)
            ground_truth_labels.append(1 if is_match else 0)

        # Store the generated texts and their labels in the cache.
        generations_cache[stmt] = {
            "prompt": prompt,
            "generated_answers": generated_texts,
            "ground_truth_labels": ground_truth_labels
        }

    # After processing all statements, save the complete cache to disk.
    with open(generations_cache_path, 'w', encoding='utf-8') as f:
        json.dump(generations_cache, f, indent=2, ensure_ascii=False)
    print(f"\nGeneration and labeling complete. Cache saved to '{generations_cache_path}'.")


def get_truth_probe_activations(
    statements,
    tokenizer,
    model,
    layers,
    layer_indices,
    device,
    output_dir="probes_data"
):
    """
    STAGE 2: Load the generated answers from the cache, create the 'True'/'False'
    prompts, run them through the model, and save the captured activations.
    """
    model_name = model.name_or_path.replace("/", "_") if hasattr(model, 'name_or_path') else 'unknown'

    # --- Setup directories and load the cache from Stage 1 ---
    generations_dir = os.path.join(output_dir, "generations")
    activations_dir = os.path.join(output_dir, "activations", model_name)
    os.makedirs(activations_dir, exist_ok=True)
    generations_cache_path = os.path.join(generations_dir, f"{model_name}_generations.json")

    if not os.path.exists(generations_cache_path):
        raise FileNotFoundError(f"Generations cache not found at '{generations_cache_path}'. Please run the 'generate' stage first.")
    with open(generations_cache_path, 'r', encoding='utf-8') as f:
        generations_cache = json.load(f)

    # Handle the case where all layers are requested.
    if -1 in layer_indices:
        layer_indices = list(range(len(layers)))

    # --- Hook setup for capturing activations ---
    residual_hooks = {l: Hook() for l in layer_indices}
    handles = [layers[l].register_forward_hook(residual_hooks[l]) for l in layer_indices]

    # Process each statement using the data from the cache.
    for stmt_idx, stmt in enumerate(tqdm(
        statements,
        desc="Stage 2: Extracting Activations",
        total=len(statements)
    )):
        if stmt not in generations_cache:
            print(f"Warning: Statement '{stmt[:50]}...' not found in cache. Skipping.")
            continue

        # To allow resuming, check if output files for this statement already exist.
        output_exists = all(os.path.exists(os.path.join(activations_dir, f"layer_{l_idx}_stmt_{stmt_idx}_.pt")) for l_idx in layer_indices)
        if output_exists:
            continue

        generation_data = generations_cache[stmt]

        # --- Create the probing prompts and their corresponding final labels ---
        appended_prompts = []
        final_labels = []
        base_prompt = generation_data["prompt"]
        for answer_text, ground_truth in zip(generation_data["generated_answers"], generation_data["ground_truth_labels"]):
            # For each generated answer, create two prompts.
            prompt_true = f"{base_prompt} {answer_text} True"
            prompt_false = f"{base_prompt} {answer_text} False"
            appended_prompts.extend([prompt_true, prompt_false])

            # The final label for the '... True' prompt is the original ground_truth.
            # The final label for the '... False' prompt is the opposite.
            final_labels.extend([ground_truth, 1 - ground_truth])

        # --- Batch process and capture activations ---
        # The batch size here is num_generations * 2 (e.g., 60).
        inputs = tokenizer(appended_prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        with t.no_grad():
            model(**inputs) # Forward pass to trigger the hooks.

        # For each specified layer, extract the captured activations.
        for l_idx in layer_indices:
            all_layer_acts = residual_hooks[l_idx].out

            # We need the activation of the very last token in each sequence.
            sequence_lengths = inputs['attention_mask'].sum(dim=1)
            last_token_indices = sequence_lengths - 1
            last_token_activations = all_layer_acts[t.arange(len(all_layer_acts)), last_token_indices]

            # Save the activations and labels for this statement and layer.
            save_path = os.path.join(activations_dir, f"layer_{l_idx}_stmt_{stmt_idx}_.pt")
            data_to_save = {
                'activations': last_token_activations.cpu(),
                'labels': t.tensor(final_labels, dtype=t.int).cpu()
            }
            t.save(data_to_save, save_path)

    # Clean up the hooks to free memory.
    for h in handles:
        h.remove()
    print(f"\nActivation extraction complete. Probes saved in '{activations_dir}'.")
