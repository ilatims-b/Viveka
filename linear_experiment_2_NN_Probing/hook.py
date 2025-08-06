import json
import os
from tqdm import tqdm
import torch as t

# All necessary imports from utils.py are retained
from utils import (create_prompts, generate_model_answers, check_correctness)

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
    2. For each answer, creates two new prompts: one ending in 'True', one in 'False'.
    3. Runs a batch of (num_generations * 2) prompts through the model.
    4. Captures the last-token activation for each prompt at specified layers.
    5. Saves the activations and a corresponding label tensor to a .pt file for each layer(batch of 60).
    """
    model_name = model.name_or_path.replace("/", "_") if hasattr(model, 'name_or_path') else 'unknown'
    
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
        
        # --- Part 1: Get initial 30 generations (from cache or new) ---
        if stmt in generations_cache:
            generation_data = generations_cache[stmt]
        else:
            # Create the initial prompt for the statement
            prompt = create_prompts([stmt], model_name)[0]
            
            # Generate 30 answers for the prompt
            generated_texts = []
            for _ in range(num_generations):
                answer_raw, _ = generate_model_answers(
                    [prompt], model, tokenizer, device, model_name, max_new_tokens=64
                )
                generated_texts.append(answer_raw[0].strip())
            
            # Label correctness for each generated answer
            ground_truth_labels = [check_correctness(text, correct_ans) for text in generated_texts]
            
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
            # Create the two variations for each generated answer
            prompt_true = f"{base_prompt} {answer_text} True"
            prompt_false = f"{base_prompt} {answer_text} False"
            
            appended_prompts.extend([prompt_true, prompt_false])
            
            # Apply the labeling logic: (ground_truth, NOT(ground_truth))
            final_labels.extend([ground_truth, 1 - ground_truth])

        # --- Part 3: Batch process and capture activations ---
        # Tokenize the entire batch of 60 prompts
        inputs = tokenizer(appended_prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        # Run a single forward pass for the entire batch to trigger hooks
        with t.no_grad():
            model(**inputs)

        # The sequence length of the padded batch
        seq_len = inputs['input_ids'].shape[1]

        # For each layer, extract the activations and save them with labels
        for l_idx in layer_indices:
            # The hook now contains the batched output: [60, seq_len, hidden_size]
            all_layer_acts = residual_hooks[l_idx].out
            
            # Extract the activation for the last non-padded token of each item in the batch
            # We use attention_mask to find the length of each sequence
            sequence_lengths = inputs['attention_mask'].sum(dim=1)
            last_token_indices = sequence_lengths - 1
            
            # Use advanced indexing to get all last-token activations at once
            last_token_activations = all_layer_acts[t.arange(len(all_layer_acts)), last_token_indices]
            
            # --- Part 4: Save the data for this layer and statement ---
            save_path = os.path.join(activations_dir, f"layer_{l_idx}_stmt_{stmt_idx}.pt")
            
            # Save activations and labels together in a dictionary
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