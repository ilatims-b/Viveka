from utils import (create_prompts, generate_model_answers, check_correctness)
import json
import os
from tqdm import tqdm
import torch as t
from thefuzz import fuzz

# The Hook class is unchanged and necessary for this task.
class Hook:
    def __init__(self): 
        self.out = None
    def __call__(self, module, module_inputs, module_outputs):
        self.out = module_outputs[0] if isinstance(module_outputs, tuple) else module_outputs

def probe_truth_representation(
    statements, 
    correct_answers, 
    tokenizer, 
    model, 
    layers, 
    layer_indices, 
    device, 
    num_generations=30, 
    output_dir="probes_data"
):
    """
    Generates responses, appends 'True'/'False', and extracts last-token activations.
    
    For each statement, this function:
    1. Generates `num_generations` answers (or loads from cache).
    2. Labels these answers using fuzzy string matching and saves to a JSON cache.
    3. For each answer, creates two new prompts: one ending in 'True', one in 'False'.
    4. Runs a batch of (num_generations * 2) prompts through the model.
    5. Captures the last-token activation for each prompt at specified layers.
    6. Saves the activations and a corresponding label tensor to a .pt file for each layer.
    """
    model_name = model.name_or_path.replace("/", "_") if hasattr(model, 'name_or_path') else 'unknown'
    
    # --- Handle 'all layers' case ---
    if layer_indices == [-1]:
        layer_indices = list(range(len(layers)))

    # --- Setup output directories ---
    generations_dir = os.path.join(output_dir, "generations")
    activations_dir = os.path.join(output_dir, "activations", model_name)
    os.makedirs(generations_dir, exist_ok=True)
    os.makedirs(activations_dir, exist_ok=True)
    
    generations_cache_path = os.path.join(generations_dir, f"{model_name}_generations.json")

    # --- Caching Logic for initial generations ---
    if os.path.exists(generations_cache_path):
        with open(generations_cache_path, 'r', encoding='utf-8') as f:
            generations_cache = json.load(f)
    else:
        generations_cache = {}

    # --- Hook setup is required for activation capture ---
    residual_hooks = {l: Hook() for l in layer_indices}
    handles = [layers[l].register_forward_hook(residual_hooks[l]) for l in layer_indices]
    
    # Main loop over each statement
    for stmt_idx, (stmt, correct_ans) in enumerate(tqdm(
        zip(statements, correct_answers), 
        desc="Probing Truth Representation", 
        total=len(statements)
    )):
        
        # --- Part 1: Get initial 30 generations and ground_truth labels (from cache or new) ---
        if stmt in generations_cache:
            generation_data = generations_cache[stmt]
        else:
            prompt = create_prompts([stmt], model_name)[0]
            
            generated_texts = []
            for _ in range(num_generations):
                answer_raw, _ = generate_model_answers(
                    [prompt], model, tokenizer, device, model_name, max_new_tokens=64
                )
                generated_texts.append(answer_raw[0].strip())
            
            # --- MODIFICATION: This is the explicit labeling step you requested ---
            # Get the ground truth label for each generated answer using fuzzy string matching.
            
            # First, ensure correct_ans is a list of strings
            try:
                correct_answers_list = eval(correct_ans)
                if not isinstance(correct_answers_list, list):
                    correct_answers_list = [str(correct_answers_list)]
            except (SyntaxError, NameError):
                correct_answers_list = [str(correct_ans)]

            # Now, label using fuzzy matching
            ground_truth_labels = []
            for text in generated_texts:
                # Use a fuzzy partial string match. If any correct answer has a high
                # similarity score with the generated text, we label it as correct (1).
                # A threshold of 90 is used for high confidence.
                is_match = any(fuzz.partial_ratio(str(ans).lower(), text.lower()) > 90 for ans in correct_answers_list)
                ground_truth_labels.append(1 if is_match else 0)

            generation_data = {
                "prompt": prompt,
                "generated_answers": generated_texts,
                "ground_truth_labels": ground_truth_labels
            }
            generations_cache[stmt] = generation_data
        
        # --- Part 2: Create stimuli and final labels for probing ---
        appended_prompts = []
        final_labels = []
        
        base_prompt = generation_data["prompt"]
        for answer_text, ground_truth in zip(generation_data["generated_answers"], generation_data["ground_truth_labels"]):
            prompt_true = f"{base_prompt} {answer_text} True"
            prompt_false = f"{base_prompt} {answer_text} False"
            
            appended_prompts.extend([prompt_true, prompt_false])
            
            # Apply the labeling logic: (ground_truth, NOT(ground_truth))
            final_labels.extend([ground_truth, 1 - ground_truth])

        # --- Part 3: Batch process and capture activations ---
        inputs = tokenizer(appended_prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with t.no_grad():
            model(**inputs)

        # For each layer, extract the activations and save them with labels
        for l_idx in layer_indices:
            all_layer_acts = residual_hooks[l_idx].out
            
            sequence_lengths = inputs['attention_mask'].sum(dim=1)
            last_token_indices = sequence_lengths - 1
            
            last_token_activations = all_layer_acts[t.arange(len(all_layer_acts)), last_token_indices]
            
            # --- Part 4: Save the data for this layer and statement ---
            save_path = os.path.join(activations_dir, f"layer_{l_idx}_stmt_{stmt_idx}.pt")
            
            data_to_save = {
                'activations': last_token_activations.cpu(),
                'labels': t.tensor(final_labels, dtype=t.int).cpu()
            }
            t.save(data_to_save, save_path)

    # --- Final Step: Clean up hooks and save the generation cache ---
    for h in handles: 
        h.remove()
        
    with open(generations_cache_path, 'w', encoding='utf-8') as f:
        json.dump(generations_cache, f, indent=2, ensure_ascii=False)

    print(f"Processing complete. Generations cached at '{generations_cache_path}'.")
    print(f"Activation probes saved in '{activations_dir}'.")