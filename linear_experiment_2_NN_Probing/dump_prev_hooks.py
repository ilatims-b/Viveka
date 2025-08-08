#hook.py safe vault
'''from utils import encode, generate, create_prompts, generate_model_answers, check_correctness, find_exact_answer_simple, extract_answer_direct, is_vague_or_non_answer, extract_answer_with_llm, _cleanup_extracted_answer, load_model, load_statements, StopOnTokens, find_answer_token_indices_by_string_matching, tokenize, try_llm_extraction
import argparse
from tqdm import tqdm
import os
import glob
from thefuzz import process, fuzz
from transformers import (AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer,
                          LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList)
import torch as t
import torch

class Hook:
    def __init__(self): 
        self.out = None
    def __call__(self, module, module_inputs, module_outputs):
        self.out = module_outputs[0] if isinstance(module_outputs, tuple) else module_outputs

def get_resid_acts(statements, correct_answers, tokenizer, model, layers, layer_indices, device, num_generations=30, enable_llm_extraction=False):
    """Fixed version with proper attention mask handling"""
    model_name = model.name_or_path if hasattr(model, 'name_or_path') else 'unknown'
   
    # Set pad token if not set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            model.resize_token_embeddings(len(tokenizer))
   
    # Hook into residual stream
    residual_hooks = {}
    handles = []
    for l in layer_indices:
        hook = Hook()
        handles.append(layers[l].register_forward_hook(hook))
        residual_hooks[l] = hook
    
    prompts = create_prompts(statements, model_name)
   
    stop_tokens = ['\n', '<end_of_turn>', '<eos>']
    stop_ids = set([tokenizer.encode(st, add_special_tokens=False)[-1] for st in stop_tokens])
    stopping_criteria = StoppingCriteriaList([StopOnTokens(list(stop_ids))])
   
    # Simplified activation storage - one entry per layer
    acts = {l: [] for l in layer_indices}
    batch_correctness, batch_model_answers, batch_exact_answers = [], [], []
    
    for stmt_idx, (stmt, prompt, correct_ans) in enumerate(tqdm(zip(statements, prompts, correct_answers), 
                                                                desc="Processing statements", 
                                                                total=len(statements), leave=False)):
        
        # Generate multiple answers for the same prompt
        stmt_model_answers = []
        stmt_correctness = []
        stmt_exact_answers = []
        
        for gen_idx in range(num_generations):
            # Generate single answer with fixed function
            model_answers_raw, generated_ids_list = generate_model_answers(
                [prompt], model, tokenizer, device, model_name, max_new_tokens=64,
                stopping_criteria=stopping_criteria
            )
            
            model_answer_text = model_answers_raw[0].strip()
            generated_ids = generated_ids_list[0]
            
            stmt_model_answers.append(model_answer_text)
            is_correct = check_correctness(model_answer_text, correct_ans)
            stmt_correctness.append(is_correct)
            
            exact_answer_str = find_exact_answer_simple(model_answer_text, correct_ans)
            
            # Use LLM extraction if needed
            if not exact_answer_str and enable_llm_extraction:
                exact_answer_str = extract_answer_with_llm(stmt, model_answer_text, model, tokenizer)
            
            stmt_exact_answers.append(exact_answer_str)
            
            # print(f"Debug: stmt_idx={stmt_idx}, gen_idx={gen_idx}, exact_answer_str='{exact_answer_str}'")

            # Extract activations if we have an exact answer OR if it's "NO ANSWER"
            if exact_answer_str or exact_answer_str == "NO ANSWER":
                # Tokenize the original prompt with attention mask
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
                
                # Create full sequence for activation extraction
                full_sequence = t.cat([inputs['input_ids'], generated_ids.unsqueeze(0).to(device)], dim=1)
                
                # Handle "NO ANSWER" case vs regular answer case
                if exact_answer_str == "NO ANSWER":
                    # Use the last token (-1) for "NO ANSWER" cases
                    adjusted_indices = t.tensor([full_sequence.shape[1] - 1], device=device)
                else:
                    # Regular case: find answer tokens
                    answer_token_indices = find_answer_token_indices_by_string_matching(
                        tokenizer, generated_ids, inputs['input_ids'].squeeze(), exact_answer_str
                    )
                    # if exact_answer_str != "NO ANSWER":
                    #     print(f"  answer_token_indices: {answer_token_indices}")
                    #     print(f"  inputs shape: {inputs['input_ids'].shape}")
                    #     print(f"  generated_ids shape: {generated_ids.shape}")
                    
                    if answer_token_indices is not None:
                        # Adjust indices for full sequence - convert to tensor and add offset
                        if isinstance(answer_token_indices, list):
                            adjusted_indices = [idx + inputs['input_ids'].shape[1] for idx in answer_token_indices]
                            adjusted_indices = t.tensor(adjusted_indices, device=device)
                        else:
                            adjusted_indices = answer_token_indices + inputs['input_ids'].shape[1]
                    else:
                        adjusted_indices = None
                
                # Extract activations if we have valid indices
                if adjusted_indices is not None:
                    with t.no_grad():
                        # Run model with proper attention mask
                        attention_mask = t.ones_like(full_sequence)
                        model(full_sequence, attention_mask=attention_mask)
                    
                    # Extract residual stream activations
                    for l in layer_indices:
                        try:
                            residual_out = residual_hooks[l].out[0][adjusted_indices].mean(dim=0).detach()
                            acts[l].append(residual_out)
                        except IndexError:
                            print(f"IndexError on layer {l} for statement {stmt_idx}, generation {gen_idx}")
                            
                        # if adjusted_indices is not None:
                        #     print(f"  ✓ Extracted activation for layer {l}")
                        # else:
                        #     print(f"  ✗ Failed to extract activation - adjusted_indices is None")

                            break
        
        # Store lists of answers for this statement
        batch_model_answers.append(stmt_model_answers)
        batch_correctness.append(stmt_correctness)
        batch_exact_answers.append(stmt_exact_answers)
    
    # Stack activations
    for k in list(acts.keys()):
        if acts[k]: 
            acts[k] = t.stack(acts[k]).cpu().float()
        else: 
            del acts[k]
    
    # Clean up hooks
    for h in handles: 
        h.remove()
       
    return acts, batch_correctness, batch_model_answers, batch_exact_answers

    '''

'''from utils import encode, generate, create_prompts, generate_model_answers, check_correctness, find_exact_answer_simple, extract_answer_direct, is_vague_or_non_answer, extract_answer_with_llm, _cleanup_extracted_answer, load_model, load_statements, StopOnTokens, find_answer_token_indices_by_string_matching, tokenize, try_llm_extraction
import argparse
from tqdm import tqdm
import os
import glob
from thefuzz import process, fuzz
from transformers import (AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer,
                          LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList)
import torch as t
import torch

class Hook:
    def __init__(self): self.out = None
    def __call__(self, module, module_inputs, module_outputs):
        self.out = module_outputs[0] if isinstance(module_outputs, tuple) else module_outputs

def find_answer_tokens_robust(tokenizer, generated_ids, exact_answer_str):
    """
    More robust token matching that handles multi-word answers better
    """
    if not exact_answer_str:
        return None
    
    # Convert generated_ids to text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Try to find the answer in the generated text
    answer_lower = exact_answer_str.lower().strip()
    generated_lower = generated_text.lower()
    
    # Find the answer in the text
    start_pos = generated_lower.find(answer_lower)
    if start_pos == -1:
        # Try without extra whitespace/punctuation
        import re
        # Remove extra spaces and punctuation for matching
        clean_answer = re.sub(r'[^\w\s]', '', answer_lower)
        clean_generated = re.sub(r'[^\w\s]', '', generated_lower)
        start_pos = clean_generated.find(clean_answer)
        if start_pos == -1:
            return None
    
    # If we found the text, try to map it back to token positions
    # Tokenize the generated text and find which tokens correspond to the answer
    tokens = tokenizer.tokenize(generated_text)
    
    # Find approximate token range by character positions
    # This is a heuristic approach
    answer_tokens = tokenizer.tokenize(exact_answer_str)
    
    # Try to find a sequence of tokens that matches the answer tokens
    for i in range(len(tokens) - len(answer_tokens) + 1):
        token_slice = tokens[i:i + len(answer_tokens)]
        # Check if this slice could represent our answer
        slice_text = tokenizer.convert_tokens_to_string(token_slice).lower().strip()
        if answer_lower in slice_text or slice_text in answer_lower:
            return list(range(i, i + len(answer_tokens)))
    
    # Fallback: return the last few tokens (often where the answer appears)
    if len(tokens) >= 3:
        return list(range(len(tokens) - 3, len(tokens)))
    else:
        return list(range(len(tokens)))


def get_resid_acts(statements, correct_answers, tokenizer, model, layers, layer_indices, device, num_generations=30, enable_llm_extraction=False):
    """Fixed version with robust token matching"""
    model_name = model.name_or_path if hasattr(model, 'name_or_path') else 'unknown'
   
    # Set pad token if not set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            model.resize_token_embeddings(len(tokenizer))
   
    # Hook into residual stream
    residual_hooks = {}
    handles = []
    for l in layer_indices:
        hook = Hook()
        handles.append(layers[l].register_forward_hook(hook))
        residual_hooks[l] = hook
    
    prompts = create_prompts(statements, model_name)
   
    stop_tokens = ['\n', '<end_of_turn>', '<eos>']
    stop_ids = set([tokenizer.encode(st, add_special_tokens=False)[-1] for st in stop_tokens])
    stopping_criteria = StoppingCriteriaList([StopOnTokens(list(stop_ids))])
   
    # Activation storage
    acts = {l: [] for l in layer_indices}
    batch_correctness, batch_model_answers, batch_exact_answers = [], [], []
    
    # DEBUG TRACKING
    total_generations = 0
    successful_extractions = 0
    failed_extractions = []
    
    for stmt_idx, (stmt, prompt, correct_ans) in enumerate(tqdm(zip(statements, prompts, correct_answers), 
                                                                desc="Processing statements", 
                                                                total=len(statements), leave=False)):
        
        stmt_model_answers = []
        stmt_correctness = []
        stmt_exact_answers = []
        
        for gen_idx in range(num_generations):
            total_generations += 1
            
            # Generate single answer
            model_answers_raw, generated_ids_list = generate_model_answers(
                [prompt], model, tokenizer, device, model_name, max_new_tokens=64,
                stopping_criteria=stopping_criteria
            )
            
            model_answer_text = model_answers_raw[0].strip()
            generated_ids = generated_ids_list[0]
            
            stmt_model_answers.append(model_answer_text)
            is_correct = check_correctness(model_answer_text, correct_ans)
            stmt_correctness.append(is_correct)
            
            exact_answer_str = find_exact_answer_simple(model_answer_text, correct_ans)
            
            # Use LLM extraction if needed
            if not exact_answer_str and enable_llm_extraction:
                exact_answer_str = extract_answer_with_llm(stmt, model_answer_text, model, tokenizer)
            
            stmt_exact_answers.append(exact_answer_str)
            
            # Extract activations if we have an exact answer OR if it's "NO ANSWER"
            if exact_answer_str or exact_answer_str == "NO ANSWER":
                try:
                    # Tokenize the original prompt with attention mask
                    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
                    
                    # Create full sequence for activation extraction
                    full_sequence = t.cat([inputs['input_ids'], generated_ids.unsqueeze(0).to(device)], dim=1)
                    
                    # Handle "NO ANSWER" case vs regular answer case
                    if exact_answer_str == "NO ANSWER":
                        adjusted_indices = t.tensor([full_sequence.shape[1] - 1], device=device)
                    else:
                        # Use the robust token matching
                        answer_token_indices = find_answer_tokens_robust(tokenizer, generated_ids, exact_answer_str)
                        
                        if answer_token_indices is not None:
                            # Adjust indices for full sequence
                            adjusted_indices = [idx + inputs['input_ids'].shape[1] for idx in answer_token_indices]
                            adjusted_indices = t.tensor(adjusted_indices, device=device)
                        else:
                            # Fallback: use the last few tokens
                            fallback_indices = max(1, min(3, len(generated_ids)))  # Use last 1-3 tokens
                            adjusted_indices = t.tensor(
                                list(range(full_sequence.shape[1] - fallback_indices, full_sequence.shape[1])), 
                                device=device
                            )
                            print(f"📍 Using fallback indices for stmt {stmt_idx}, gen {gen_idx}: {exact_answer_str[:30]}...")
                    
                    # Extract activations - we should always have valid indices now
                    with t.no_grad():
                        attention_mask = t.ones_like(full_sequence)
                        model(full_sequence, attention_mask=attention_mask)
                    
                    # Extract residual stream activations
                    layer_success = True
                    temp_activations = {}
                    
                    for l in layer_indices:
                        try:
                            residual_out = residual_hooks[l].out[0][adjusted_indices].mean(dim=0).detach()
                            temp_activations[l] = residual_out
                        except (IndexError, TypeError, RuntimeError) as e:
                            print(f"❌ Layer {l} extraction failed for stmt {stmt_idx}, gen {gen_idx}: {e}")
                            layer_success = False
                            break
                    
                    # Only add activations if ALL layers succeeded
                    if layer_success:
                        for l in layer_indices:
                            acts[l].append(temp_activations[l])
                        successful_extractions += 1
                    else:
                        failed_extractions.append({
                            'stmt_idx': stmt_idx,
                            'gen_idx': gen_idx,
                            'reason': 'layer_extraction_failed',
                            'exact_answer': exact_answer_str
                        })
                        
                except Exception as e:
                    print(f"❌ General extraction error for stmt {stmt_idx}, gen {gen_idx}: {e}")
                    failed_extractions.append({
                        'stmt_idx': stmt_idx,
                        'gen_idx': gen_idx,
                        'reason': f'exception: {str(e)}',
                        'exact_answer': exact_answer_str
                    })
            else:
                # No exact answer found
                failed_extractions.append({
                    'stmt_idx': stmt_idx,
                    'gen_idx': gen_idx,
                    'reason': 'no_exact_answer',
                    'exact_answer': exact_answer_str
                })
        
        # Store lists of answers for this statement
        batch_model_answers.append(stmt_model_answers)
        batch_correctness.append(stmt_correctness)
        batch_exact_answers.append(stmt_exact_answers)
    
    # DEBUG OUTPUT
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"   Total generations: {total_generations}")
    print(f"   Successful extractions: {successful_extractions}")
    print(f"   Failed extractions: {len(failed_extractions)}")
    print(f"   Success rate: {successful_extractions/total_generations*100:.1f}%")
    
    # Stack activations
    for k in list(acts.keys()):
        if acts[k]: 
            acts[k] = t.stack(acts[k]).cpu().float()
            print(f"   Layer {k}: {acts[k].shape[0]} activations")
        else: 
            del acts[k]
    
    # Clean up hooks
    for h in handles: 
        h.remove()
       
    return acts, batch_correctness, batch_model_answers, batch_exact_answers'''

"""
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
    """"""
    Generates responses, appends 'True'/'False', and extracts last-token activations.
    
    For each statement, this function:
    1. Generates `num_generations` answers (or loads from cache).
    2. For each answer, creates two new prompts: one ending in 'True', one in 'False'.
    3. Runs a batch of (num_generations * 2) prompts through the model.
    4. Captures the last-token activation for each prompt at specified layers.
    5. Saves the activations and a corresponding label tensor to a .pt file for each layer(batch of 60).
    """
   """ model_name = model.name_or_path.replace("/", "_") if hasattr(model, 'name_or_path') else 'unknown'
    
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

    """
"""
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
""" Generates responses, appends 'True'/'False', and extracts last-token activations.
    
    For each statement, this function:
    1. Generates `num_generations` answers (or loads from cache).
    2. Labels these answers using fuzzy string matching and saves to a JSON cache.
    3. For each answer, creates two new prompts: one ending in 'True', one in 'False'.
    4. Runs a batch of (num_generations * 2) prompts through the model.
    5. Captures the last-token activation for each prompt at specified layers.
    6. Saves the activations and a corresponding label tensor to a .pt file for each layer.
    """    
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
"""

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
