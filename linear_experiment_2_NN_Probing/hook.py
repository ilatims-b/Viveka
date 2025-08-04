from utils import encode, generate, create_prompts, generate_model_answers, check_correctness, find_exact_answer_simple, extract_answer_direct, is_vague_or_non_answer, extract_answer_with_llm, _cleanup_extracted_answer, load_model, load_statements, StopOnTokens, find_answer_token_indices_by_string_matching, tokenize, try_llm_extraction
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

def get_resid_acts(statements, correct_answers, tokenizer, model, layers, layer_indices, device, num_generations=30, enable_llm_extraction=False):
    """Version with comprehensive tracking to debug the mismatch"""
    model_name = model.name_or_path if hasattr(model, 'name_or_path') else 'unknown'
   
    # Set pad token if not set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            model.resize_token_embeddings(len(tokenizer))
   
    # Hook into residual stream - back to single hook setup for stability
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
   
    # Activation storage and tracking
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
            
            # DEBUG: Track what happens with each generation
            extraction_attempted = False
            extraction_successful = False
            
            # Extract activations if we have an exact answer OR if it's "NO ANSWER"
            if exact_answer_str or exact_answer_str == "NO ANSWER":
                extraction_attempted = True
                
                try:
                    # Tokenize the original prompt with attention mask
                    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
                    
                    # Create full sequence for activation extraction
                    full_sequence = t.cat([inputs['input_ids'], generated_ids.unsqueeze(0).to(device)], dim=1)
                    
                    # Handle "NO ANSWER" case vs regular answer case
                    if exact_answer_str == "NO ANSWER":
                        adjusted_indices = t.tensor([full_sequence.shape[1] - 1], device=device)
                    else:
                        answer_token_indices = find_answer_token_indices_by_string_matching(
                            tokenizer, generated_ids, inputs['input_ids'].squeeze(), exact_answer_str
                        )
                        
                        if answer_token_indices is not None:
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
                            extraction_successful = True
                            successful_extractions += 1
                        else:
                            failed_extractions.append({
                                'stmt_idx': stmt_idx,
                                'gen_idx': gen_idx,
                                'reason': 'layer_extraction_failed',
                                'exact_answer': exact_answer_str
                            })
                    else:
                        failed_extractions.append({
                            'stmt_idx': stmt_idx,
                            'gen_idx': gen_idx,
                            'reason': 'no_valid_indices',
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
                    'exact_answer': exact_answer_str,
                    'model_answer_preview': model_answer_text[:50] + "..." if len(model_answer_text) > 50 else model_answer_text
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
    
    if failed_extractions:
        print(f"\n❌ FAILURE BREAKDOWN:")
        from collections import Counter
        failure_reasons = Counter(f['reason'] for f in failed_extractions)
        for reason, count in failure_reasons.items():
            print(f"   {reason}: {count}")
        
        print(f"\n🔍 FIRST FEW FAILURES:")
        for i, failure in enumerate(failed_extractions[:5]):
            print(f"   {i+1}. Stmt {failure['stmt_idx']}, Gen {failure['gen_idx']}: {failure['reason']}")
            if 'model_answer_preview' in failure:
                print(f"      Model answer: {failure['model_answer_preview']}")
            print(f"      Exact answer: {failure['exact_answer']}")
    
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
       
    return acts, batch_correctness, batch_model_answers, batch_exact_answers