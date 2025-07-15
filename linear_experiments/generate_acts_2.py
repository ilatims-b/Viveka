import torch
import torch as t
# MODIFIED: Import StoppingCriteria for robust stopping
from transformers import (AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer,
                          LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList)
import argparse
import pandas as pd
from tqdm import tqdm
import os
import glob
from thefuzz import process, fuzz

# --- NEW: Custom StoppingCriteria Class ---
class StopOnTokens(StoppingCriteria):
    """
    A custom StoppingCriteria that stops generation when any of the specified
    stop token IDs are generated.
    """
    def __init__(self, stop_ids: list):
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if the last generated token is in the stop list
        if input_ids[0][-1] in self.stop_ids:
            return True
        return False

# --- User-provided helper functions ---
def generate(model_input, model, model_name, do_sample=False, output_scores=False, temperature=1.0, top_k=50, top_p=1.0,
             max_new_tokens=100, stop_token_id=None, tokenizer=None, output_hidden_states=False, additional_kwargs=None,
             stopping_criteria=None):
    if stop_token_id is not None: eos_token_id = stop_token_id
    else: eos_token_id = None
    
    model_output = model.generate(model_input,
                                  max_new_tokens=max_new_tokens, output_hidden_states=output_hidden_states,
                                  output_scores=output_scores,
                                  return_dict_in_generate=True, do_sample=do_sample,
                                  temperature=temperature, top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
                                  stopping_criteria=stopping_criteria,
                                  **(additional_kwargs or {}))
    return model_output

def tokenize(prompt, tokenizer, model_name, tokenizer_args=None):
    if 'instruct' in model_name.lower() or 'it' in model_name.lower():
        messages = [{"role": "user", "content": prompt}]
        model_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('cuda')
    else:
        model_input = tokenizer(prompt, return_tensors='pt', **(tokenizer_args or {})).to('cuda')
        if "input_ids" in model_input:
            model_input = model_input["input_ids"]
    return model_input

def create_prompts(statements, model_name):
    """Applies the correct prompt format based on the model type."""
    mn_lower = model_name.lower()
    
    if 'instruct' in mn_lower or 'it' in mn_lower:
        return statements
    elif 'gemma' in mn_lower:
        return [f"<start_of_turn>user\nQ: {s}<end_of_turn>\n<start_of_turn>model\nA:" for s in statements]
    else:
        return [f"Q:<start_of_turn>user\n{s}<end_of_turn>\n<start_of_turn>model\nA:" for s in statements]


def generate_model_answers(data, model, tokenizer, device, model_name, do_sample=False,
                           temperature=1.0, top_p=1.0, max_new_tokens=100, stop_token_id=None, verbose=False,
                           stopping_criteria=None):
    all_textual_answers = []
    all_input_output_ids = []
    
    for prompt in data:
        model_input = tokenize(prompt, tokenizer, model_name)
        with torch.no_grad():
            model_output = generate(model_input, model, model_name, do_sample=do_sample,
                                    max_new_tokens=max_new_tokens,
                                    top_p=top_p, temperature=temperature,
                                    stop_token_id=stop_token_id, tokenizer=tokenizer,
                                    stopping_criteria=stopping_criteria)
        
        answer = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):], skip_special_tokens=True)
        all_textual_answers.append(answer)
        all_input_output_ids.append(model_output['sequences'][0].cpu())
    return all_textual_answers, all_input_output_ids

# --- Other Helper Functions ---

def _cleanup_extracted_answer(decoded_output):
    """Clean up the extracted answer more aggressively"""
    # Remove common artifacts
    answer = decoded_output.strip()
    
    # Remove stop tokens and artifacts
    tokens_to_remove = [
        ".</s>", "</s>", ".<|eot_id|>", "<|eot_id|>", ".<eos>", "<eos>", 
        "<end_of_turn>", "\n", "Answer:", "A:", "The answer is", "The answer:"
    ]
    
    for token in tokens_to_remove:
        answer = answer.replace(token, "")
    
    # Take only the first meaningful part
    answer = answer.strip()
    
    # Remove parenthetical information
    if "(" in answer:
        answer = answer.split("(")[0].strip()
    
    # Remove trailing punctuation
    answer = answer.rstrip(".,!?;:")
    
    # If still too long, take first few words
    words = answer.split()
    if len(words) > 5:
        answer = " ".join(words[:3])
    
    return answer if answer else None

def check_correctness(model_answer, correct_answers_list):
    if isinstance(correct_answers_list, str):
        try: labels_ = eval(correct_answers_list)
        except: labels_ = [correct_answers_list]
    else: labels_ = correct_answers_list
    if not isinstance(labels_, list): labels_ = [str(labels_)]
    if not isinstance(model_answer, str): return 0
    for ans in labels_:
        if str(ans).lower() in model_answer.lower(): return 1
    return 0

def find_exact_answer_simple(model_answer: str, correct_answer):
    if not isinstance(model_answer, str): return None
    try:
        if isinstance(correct_answer, str):
            correct_answer_eval = eval(correct_answer)
            if isinstance(correct_answer_eval, list): correct_answer = correct_answer_eval
    except (SyntaxError, NameError): pass
    found_ans, found_ans_index = "", -1
    if isinstance(correct_answer, list):
        current_best_index = len(model_answer)
        for ans in correct_answer:
            ans_str = str(ans)
            ans_index = model_answer.lower().find(ans_str.lower())
            if ans_index != -1 and ans_index < current_best_index:
                found_ans, current_best_index = ans_str, ans_index
        found_ans_index = current_best_index if found_ans else -1
    else:
        ans_str = str(correct_answer)
        found_ans_index = model_answer.lower().find(ans_str.lower())
        found_ans = ans_str
    if found_ans_index != -1:
        return model_answer[found_ans_index : found_ans_index + len(found_ans)]
    return None

def extract_answer_with_llm(question, model_answer, model, tokenizer, max_retries=2):
    """
    Extract answer using LLM with improved prompting and multiple attempts
    """
    model_name = model.name_or_path if hasattr(model, 'name_or_path') else 'unknown'
    mn_lower = model_name.lower()
    
    # Try different prompt strategies
    prompts_to_try = [
        # Simple direct prompt
        f"Q: {question}\nA: {model_answer}\n\nWhat is the exact answer? (respond with just the answer):",
        
        # Few-shot prompt
        f"""Extract the answer:

Q: What is the capital of France?
A: The capital of France is Paris, which is located in the northern part of the country.
Answer: Paris

Q: Who wrote Romeo and Juliet?
A: Romeo and Juliet was written by William Shakespeare in the 1590s.
Answer: William Shakespeare

Q: {question}
A: {model_answer}
Answer:""",
        
        # Direct instruction
        f"Question: {question}\nResponse: {model_answer}\n\nExtract only the key answer (1-3 words):"
    ]
    
    for attempt, base_prompt in enumerate(prompts_to_try):
        try:
            # Format according to model type
            if 'instruct' in mn_lower or 'it' in mn_lower:
                # For instruct models, use chat template
                messages = [{"role": "user", "content": base_prompt}]
                inputs = tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    return_tensors="pt"
                ).to(model.device)
                inputs = {"input_ids": inputs}
            else:
                # For base models, add appropriate formatting
                if 'gemma' in mn_lower:
                    formatted_prompt = f"<start_of_turn>user\n{base_prompt}<end_of_turn>\n<start_of_turn>model\n"
                else:
                    formatted_prompt = base_prompt
                
                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            # Create stopping criteria
            stop_tokens = ['<end_of_turn>', '<eos>', '</s>', '<|eot_id|>', '\n\n']
            stop_ids = []
            for st in stop_tokens:
                try:
                    encoded = tokenizer.encode(st, add_special_tokens=False)
                    if encoded:
                        stop_ids.append(encoded[-1])
                except:
                    continue
            
            stopping_criteria = StoppingCriteriaList([StopOnTokens(list(set(stop_ids)))])
            
            # Generate with different parameters for each attempt
            temp = 0.1 if attempt == 0 else 0.3
            max_tokens = 20 if attempt == 0 else 30
            
            with t.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    do_sample=temp > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                    repetition_penalty=1.1
                )
            
            # Decode only new tokens
            new_tokens = outputs[0, inputs['input_ids'].shape[1]:]
            decoded_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up the answer
            cleaned_answer = _cleanup_extracted_answer(decoded_answer)
            
            # Validate the answer
            if cleaned_answer and len(cleaned_answer) > 0:
                # Check if it's not just the instruction repeated
                if not any(phrase in cleaned_answer.lower() for phrase in [
                    "extract", "answer from", "question", "response", "ignore factual"
                ]):
                    return cleaned_answer
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            continue
    
    return None

def find_answer_token_indices_by_string_matching(tokenizer, full_generated_ids, prompt_ids, exact_answer_str):
    try: full_decoded_text = tokenizer.decode(full_generated_ids, skip_special_tokens=False)
    except: return None
    match = process.extractOne(exact_answer_str, [full_decoded_text], scorer=fuzz.partial_ratio, score_cutoff=90)
    if not match: return None
    best_match_str = match[0]
    start_char = full_decoded_text.find(best_match_str)
    if start_char == -1: return None
    end_char = start_char + len(best_match_str)
    encoding = tokenizer(full_decoded_text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoding['offset_mapping']
    token_indices = []
    for i, (token_start, token_end) in enumerate(offset_mapping):
        if token_start < end_char and start_char < token_end:
            if i < len(full_generated_ids):
                token_indices.append(i)
    return token_indices if token_indices else None

class Hook:
    def __init__(self): self.out = None
    def __call__(self, module, module_inputs, module_outputs):
        self.out = module_outputs[0] if isinstance(module_outputs, tuple) else module_outputs

def load_model(model_repo_id: str, device: str):
    print(f"Loading from Hugging Face Hub: {model_repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_repo_id,
        device_map='auto',
        torch_dtype=t.bfloat16 if t.cuda.is_available() and t.cuda.is_bf16_supported() else t.float16
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    layers = getattr(getattr(model, 'model', None), 'layers', None)
    if layers is None:
        raise AttributeError(f"Could not find layers for model {model_repo_id}. Please check the model architecture.")
    return tokenizer, model, layers

# FIXED: Added encoding='utf-8' for robust file reading.
def load_statements(dataset_name):
    path = f"datasets/{dataset_name}.csv"
    df = pd.read_csv(path, encoding='utf-8')
    question_col = 'statement' if 'statement' in df.columns else 'raw_question'
    label_col = 'label' if 'label' in df.columns else 'correct_answer'
    if question_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Dataset {dataset_name}.csv must have a question and an answer column.")
    return df, df[question_col].tolist(), df[label_col].tolist()

def get_acts(statements, correct_answers, tokenizer, model, layers, layer_indices, device, enable_llm_extraction=False):
    model_name = model.name_or_path if hasattr(model, 'name_or_path') else 'unknown'
    
    attn_hooks, mlp_hooks = {}, {}
    handles = []
    for l in layer_indices: 
        hook_a, hook_m = Hook(), Hook()
        handles.extend([
            layers[l].self_attn.register_forward_hook(hook_a),
            layers[l].mlp.register_forward_hook(hook_m)
        ])
        attn_hooks[l], mlp_hooks[l] = hook_a, hook_m

    prompts = create_prompts(statements, model_name)
    
    stop_tokens = ['\n', '<end_of_turn>', '<eos>']
    stop_ids = set([tokenizer.encode(st, add_special_tokens=False)[-1] for st in stop_tokens])
    stopping_criteria = StoppingCriteriaList([StopOnTokens(list(stop_ids))])
    
    all_model_answers_raw, all_generated_ids = generate_model_answers(
        prompts, model, tokenizer, device, model_name, max_new_tokens=64,
        stopping_criteria=stopping_criteria
    )

    acts = {2*l: [] for l in layer_indices}
    acts.update({2*l + 1: [] for l in layer_indices})
    batch_correctness, batch_model_answers, batch_exact_answers = [], [], []

    iterator = zip(statements, prompts, correct_answers, all_model_answers_raw, all_generated_ids)
    for stmt, prompt, correct_ans, model_answer_text_raw, generated_ids in tqdm(iterator, desc="Processing batch", total=len(statements), leave=False):
        
        model_answer_text = model_answer_text_raw.strip()
        
        batch_model_answers.append(model_answer_text)
        is_correct = check_correctness(model_answer_text, correct_ans)
        batch_correctness.append(is_correct)
        
        exact_answer_str = find_exact_answer_simple(model_answer_text, correct_ans)
        
        # Use the improved LLM extraction
        if not exact_answer_str and enable_llm_extraction:
            print(f"Trying LLM extraction for: {stmt[:50]}...")
            exact_answer_str = extract_answer_with_llm(stmt, model_answer_text, model, tokenizer)
            if exact_answer_str:
                print(f"LLM extracted: '{exact_answer_str}'")
        
        batch_exact_answers.append(exact_answer_str)
        
        if not exact_answer_str:
            continue

        input_ids = tokenize(prompt, tokenizer, model_name)
        answer_token_indices = find_answer_token_indices_by_string_matching(tokenizer, generated_ids, input_ids, exact_answer_str)
        if answer_token_indices is None:
            continue
        
        with t.no_grad():
            model(generated_ids.unsqueeze(0).to(device))

        for l in layer_indices:
            try:
                a = attn_hooks[l].out[0][answer_token_indices].mean(dim=0).detach()
                m = mlp_hooks[l].out[0][answer_token_indices].mean(dim=0).detach()
                acts[2*l].append(a)
                acts[2*l+1].append(m)
            except IndexError:
                print(f"IndexError on layer {l}. Skipping activation extraction for this item.")
                break

    for k in list(acts.keys()):
        if acts[k]: acts[k] = t.stack(acts[k]).cpu().float()
        else: del acts[k]

    for h in handles: h.remove()
        
    return acts, batch_correctness, batch_model_answers, batch_exact_answers

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract activations and check correctness from Hugging Face models.")
    parser.add_argument('--model_repo_id', type=str, required=True, help='The exact Hugging Face Hub repository ID (e.g., google/gemma-2b-it)')
    parser.add_argument('--layers', nargs='+', required=True, type=int, help='Layer indices to extract from (-1 for all)')
    parser.add_argument('--datasets', nargs='+', required=True, help='Dataset names (without .csv)')
    parser.add_argument('--output_dir', default='acts_output', help='Root directory for saving all outputs')
    parser.add_argument('--device', default='cuda' if t.cuda.is_available() else 'cpu', help='Device to run on (cpu or cuda)')
    parser.add_argument('--enable_llm_extraction', action='store_true', help='Enable LLM-based answer extraction as a fallback.')
    parser.add_argument('--early_stop', action='store_true', help='Process only two batches and save a subsampled CSV for quick checks.')
    args = parser.parse_args()

    ds = args.datasets
    if ds == ['all']:
        ds = [os.path.relpath(fp, 'datasets').replace('.csv', '') for fp in glob.glob('datasets/**/*.csv', recursive=True)]

    t.set_grad_enabled(False)
    tokenizer, model, layer_modules = load_model(args.model_repo_id, args.device)
    
    li = args.layers
    if -1 in li: li = list(range(len(layer_modules)))

    for dataset in ds:
        print(f"\n--- Processing dataset: {dataset} ---")
        try:
            df, stmts, correct_answers = load_statements(dataset)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping dataset '{dataset}': {e}")
            continue
            
        safe_repo_name = args.model_repo_id.replace("/", "__")
        save_base = os.path.join(args.output_dir, safe_repo_name, dataset)
        os.makedirs(save_base, exist_ok=True)
        
        all_correctness_results, all_model_answers, all_exact_answers = [], [], []
        
        batch_size = 25
        batch_count = 0 
        for start in tqdm(range(0, len(stmts), batch_size), desc=f"Overall progress for {dataset}"):
            batch_stmts = stmts[start:start + batch_size]
            batch_correct_ans = correct_answers[start:start + batch_size]
            
            acts, batch_correctness, batch_model_ans, batch_exact_ans = get_acts(
                batch_stmts, batch_correct_ans, tokenizer, model, layer_modules, li, args.device, 
                enable_llm_extraction=args.enable_llm_extraction
            )
            
            all_correctness_results.extend(batch_correctness)
            all_model_answers.extend(batch_model_ans)
            all_exact_answers.extend(batch_exact_ans)

            if acts:
                for pseudo, tensor in acts.items():
                    filename = os.path.join(save_base, f"layer_{pseudo}_{start}.pt")
                    t.save(tensor, filename)
            
            batch_count += 1
            if args.early_stop and batch_count >= 2:
                print(f"\nEarly stopping after {batch_count} batches.")
                break
        
        num_results = len(all_model_answers)
        if args.early_stop:
            if num_results > 0:
                df_sub = df.iloc[:num_results].copy()
                df_sub['model_answer'] = all_model_answers
                df_sub['automatic_correctness'] = all_correctness_results
                df_sub['exact_answer'] = all_exact_answers
                
                output_csv_path = os.path.join(save_base, f"{dataset}_SUBSAMPLED_with_results.csv")
                # FIXED: Added encoding='utf-8' for robust file writing.
                df_sub.to_csv(output_csv_path, index=False, encoding='utf-8')
                print(f"✅ Early stop: Saved subsampled results to: {output_csv_path}")
            else:
                print("⚠️ Warning: No results generated during early stop run. No CSV saved.")
        elif num_results == len(df):
            df['model_answer'] = all_model_answers
            df['automatic_correctness'] = all_correctness_results
            df['exact_answer'] = all_exact_answers
            
            output_csv_path = os.path.join(save_base, f"{dataset}_with_results.csv")
            # FIXED: Added encoding='utf-8' for robust file writing.
            df.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"✅ Saved full dataset with results to: {output_csv_path}")
        else:
            print(f"⚠️ Warning: Mismatch between results ({num_results}) and dataset rows ({len(df)}). CSV not saved.")