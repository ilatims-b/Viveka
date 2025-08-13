from utils import (create_prompts, generate_model_answers)
import json
import os
from tqdm import tqdm
import torch as t
from thefuzz import fuzz


class Hook:
    """A simple hook class to store the output of a layer."""
    def __init__(self):
        self.out = None
    def __call__(self, module, module_inputs, module_outputs):
        self.out = module_outputs[0] if isinstance(module_outputs, tuple) else module_outputs

def generate_and_label_answers(
    statements,
    correct_answers,
    tokenizer,
    model,
    device,
    num_generations=32,
    output_dir="current_run"
):
    """ 
    STAGE 1: Generate answers for a slice of statements in batch,
    label them, and save/update the results in a central JSON cache.
    """
    model_name = model.name_or_path.replace("/", "_") if hasattr(model, 'name_or_path') else 'unknown'
    generations_dir = os.path.join(output_dir, "generations")
    os.makedirs(generations_dir, exist_ok=True)
    generations_cache_path = os.path.join(generations_dir, f"{model_name}_generations.json")

    if os.path.exists(generations_cache_path):
        with open(generations_cache_path, 'r', encoding='utf-8') as f:
            generations_cache = json.load(f)
    else:
        generations_cache = {}

    # Filter out already-generated statements
    batch_statements = []
    batch_correct_answers = []
    for stmt, correct_ans in zip(statements, correct_answers):
        if stmt not in generations_cache:
            batch_statements.append(stmt)
            batch_correct_answers.append(correct_ans)

    if not batch_statements:
        return

    # Create prompts for all statements in this batch
    prompts = create_prompts(batch_statements, model_name)

    # Generate all answers in parallel
    all_generated, _ = generate_model_answers(
        prompts, model, tokenizer, device, model_name,
        max_new_tokens=64,
        num_return_sequences=num_generations,
        do_sample=True
    )

    # all_generated will be length = len(batch_statements) * num_generations
    # We regroup them by statement
    for i, stmt in enumerate(batch_statements):
        stmt_generations = all_generated[i * num_generations:(i + 1) * num_generations]
        generated_texts = [g.strip() for g in stmt_generations]

        # Parse correct answers list
        try:
            correct_answers_list = eval(batch_correct_answers[i])
            if not isinstance(correct_answers_list, list):
                correct_answers_list = [str(correct_answers_list)]
        except (SyntaxError, NameError):
            correct_answers_list = [str(batch_correct_answers[i])]

        # Label generations
        ground_truth_labels = []
        for text in generated_texts:
            is_match = any(fuzz.partial_ratio(str(ans).lower(), text.lower()) > 90
                           for ans in correct_answers_list)
            ground_truth_labels.append(1 if is_match else 0)

        generations_cache[stmt] = {
            "prompt": prompts[i],
            "generated_answers": generated_texts,
            "ground_truth_labels": ground_truth_labels
        }

    with open(generations_cache_path, 'w', encoding='utf-8') as f:
        json.dump(generations_cache, f, indent=2, ensure_ascii=False)

    print(f"\nGeneration and labeling complete for this slice. Cache updated at '{generations_cache_path}'.")



def get_truth_probe_activations(
    statements,
    tokenizer,
    model,
    layers,
    layer_indices,
    device,
    output_dir="probes_data",
    start_index=0
):
    """
    STAGE 2: Load generated answers from the cache for a slice of statements,
    and save the captured activations using the correct global index.
    """
    model_name = model.name_or_path.replace("/", "_") if hasattr(model, 'name_or_path') else 'unknown'

    generations_dir = os.path.join(output_dir, "generations")
    activations_dir = os.path.join(output_dir, "activations", model_name)
    os.makedirs(activations_dir, exist_ok=True)
    generations_cache_path = os.path.join(generations_dir, f"{model_name}_generations.json")

    if not os.path.exists(generations_cache_path):
        raise FileNotFoundError(f"Generations cache not found at '{generations_cache_path}'. Please run the 'generate' stage first.")
    with open(generations_cache_path, 'r', encoding='utf-8') as f:
        generations_cache = json.load(f)

    if -1 in layer_indices:
        layer_indices = list(range(len(layers)))

    residual_hooks = {l: Hook() for l in layer_indices}
    handles = [layers[l].register_forward_hook(residual_hooks[l]) for l in layer_indices]

    # This loop already had a tqdm wrapper
    for local_idx, stmt in enumerate(tqdm(
        statements,
        desc="Stage 2: Extracting Activations",
        total=len(statements)
    )):
        global_stmt_idx = start_index + local_idx
        
        if stmt not in generations_cache:
            print(f"Warning: Statement (Index {global_stmt_idx}) '{stmt[:50]}...' not found in cache. Skipping.")
            continue

        output_exists = all(os.path.exists(os.path.join(activations_dir, f"layer_{l_idx}_stmt_{global_stmt_idx}.pt")) for l_idx in layer_indices)
        if output_exists:
            continue

        generation_data = generations_cache[stmt]
        appended_prompts = []
        final_labels = []
        base_prompt = generation_data["prompt"]
        for answer_text, ground_truth in zip(generation_data["generated_answers"], generation_data["ground_truth_labels"]):
            prompt_true = f"{base_prompt} {answer_text} True"
            prompt_false = f"{base_prompt} {answer_text} False"
            appended_prompts.extend([prompt_true, prompt_false])
            final_labels.extend([ground_truth, 1 - ground_truth])

        inputs = tokenizer(appended_prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        with t.no_grad():
            model(**inputs)

        for l_idx in layer_indices:
            all_layer_acts = residual_hooks[l_idx].out
            sequence_lengths = inputs['attention_mask'].sum(dim=1)
            last_token_indices = sequence_lengths - 1
            last_token_activations = all_layer_acts[t.arange(len(all_layer_acts)), last_token_indices]

            save_path = os.path.join(activations_dir, f"layer_{l_idx}_stmt_{global_stmt_idx}.pt")
            data_to_save = {
                'activations': last_token_activations.cpu(),
                'labels': t.tensor(final_labels, dtype=t.int).cpu()
            }
            t.save(data_to_save, save_path)

    for h in handles:
        h.remove()
    print(f"\nActivation extraction complete for this slice. Probes saved in '{activations_dir}'.")