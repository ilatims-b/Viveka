from utils import encode, generate, create_prompts, generate_model_answers, check_correctness, find_exact_answer_simple, extract_answer_direct, is_vague_or_non_answer, extract_answer_with_llm, _cleanup_extracted_answer, load_model, load_statements


class Hook:
    def __init__(self): self.out = None
    def __call__(self, module, module_inputs, module_outputs):
        self.out = module_outputs[0] if isinstance(module_outputs, tuple) else module_outputs

def get_resid_acts(statements, correct_answers, tokenizer, model, layers, layer_indices, device, num_generations=30, enable_llm_extraction=False):
    """Modified function to handle multiple generations per statement"""
    mo  del_name = model.name_or_path if hasattr(model, 'name_or_path') else 'unknown'
   
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
            
            # Extract activations if we have an exact answer
            if exact_answer_str:
                input_ids = tokenize(prompt, tokenizer, model_name)
                answer_token_indices = find_answer_token_indices_by_string_matching(
                    tokenizer, generated_ids, input_ids, exact_answer_str
                )
                
                if answer_token_indices is not None:
                    with t.no_grad():
                        model(generated_ids.unsqueeze(0).to(device))
                    
                    # Extract residual stream activations
                    for l in layer_indices:
                        try:
                            residual_out = residual_hooks[l].out[0][answer_token_indices].mean(dim=0).detach()
                            acts[l].append(residual_out)
                        except IndexError:
                            print(f"IndexError on layer {l} for statement {stmt_idx}, generation {gen_idx}")
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