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