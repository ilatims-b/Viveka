import torch
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import argparse
import pandas as pd
from tqdm import tqdm
import os
import glob
from thefuzz import process, fuzz

# --- NEW: User-provided helper functions ---

def generate(model_input, model, model_name, do_sample=False, output_scores=False, temperature=1.0, top_k=50, top_p=1.0,
             max_new_tokens=100, stop_token_id=None, tokenizer=None, output_hidden_states=False, additional_kwargs=None):
    """
    Generates token sequences from a model input, with extensive customization options.
    """
    if stop_token_id is not None:
        eos_token_id = stop_token_id
    else:
        eos_token_id = None
    
    # Corrected a syntax error in the original provided line: removed one closing parenthesis.
    model_output = model.generate(model_input,
                                  max_new_tokens=max_new_tokens, output_hidden_states=output_hidden_states,
                                  output_scores=output_scores,
                                  return_dict_in_generate=True, do_sample=do_sample,
                                  temperature=temperature, top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
                                  **(additional_kwargs or {}))

    return model_output

def tokenize(prompt, tokenizer, model_name, tokenizer_args=None):
    """
    Tokenizes a prompt, applying a chat template if the model is an "instruct" version.
    """
    if 'instruct' in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        # The model_input is moved to cuda within this function.
        model_input = tokenizer.apply_chat_template(messages, return_tensors="pt", **(tokenizer_args or {})).to('cuda')
    else:  # non-instruct model
        model_input = tokenizer(prompt, return_tensors='pt', **(tokenizer_args or {}))
        if "input_ids" in model_input:
            model_input = model_input["input_ids"].to('cuda')
    return model_input

# --- Main Answer Generation Function (uses new helpers) ---

def generate_model_answers(data, model, tokenizer, device, model_name, do_sample=False,
                           temperature=1.0, top_p=1.0, max_new_tokens=100, stop_token_id=None, verbose=False):
    """Generates answers for a list of prompts."""
    all_textual_answers = []
    all_input_output_ids = []
    
    for prompt in data:
        # The new tokenize function handles device placement. The 'device' arg is unused but kept for compatibility.
        model_input = tokenize(prompt, tokenizer, model_name)
        with torch.no_grad():
            # The new generate function returns a dict-like object directly.
            model_output = generate(model_input, model, model_name, do_sample=do_sample,
                                    max_new_tokens=max_new_tokens,
                                    top_p=top_p, temperature=temperature,
                                    stop_token_id=stop_token_id, tokenizer=tokenizer)
        
        # Accessing 'sequences' from the ModelOutput object works like a dictionary.
        answer = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):], skip_special_tokens=True)
        all_textual_answers.append(answer)
        all_input_output_ids.append(model_output['sequences'][0].cpu())

    return all_textual_answers, all_input_output_ids

# --- Other Helper Functions (Unchanged) ---

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

def _create_extraction_prompt(raw_question, model_answer):
    return f"""
        Extract the exact answer from the long answer. If the long answer doesn't answer the question, return “NO ANSWER.” Ignore factual correctness; extract what appears most relevant.

        Examples:
        Q: Which musical featured the song The Street Where You Live?
        A: The song "The Street Where You Live" is from the Lerner and Loewe musical "My Fair Lady."
        Exact answer: My Fair Lady

        Q: Who wrote Philosophiæ Naturalis Principia Mathematica?
        A: Albert Einstein
        Exact answer: Albert Einstein

        Now extract for this:
        Q: {raw_question}
        A: {model_answer}
        Exact answer:
    """

def _cleanup_batched_answer(decoded_output, model_name):
    if "Exact answer:" in decoded_output: answer_part = decoded_output.split("Exact answer:")[-1]
    else: answer_part = decoded_output
    tokens_to_remove = [".</s>", "</s>", ".<|eot_id|>", "<|eot_id|>", ".<eos>", "<eos>", "<end_of_turn>"]
    for token in tokens_to_remove: answer_part = answer_part.replace(token, "")
    for line in answer_part.strip().split('\n'):
        cleaned_line = line.strip()
        if cleaned_line: return cleaned_line.split("(")[0].strip().strip(".")
    return ""

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

def extract_answer_with_llm(question, model_answer, model, tokenizer):
    prompt = _create_extraction_prompt(question, model_answer)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with t.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    model_name = model.name_or_path if hasattr(model, 'name_or_path') else 'gemma'
    exact_answer = _cleanup_batched_answer(decoded_output, model_name)
    if exact_answer and exact_answer.upper() != "NO ANSWER": return exact_answer
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

def load_statements(dataset_name):
    path = f"datasets/{dataset_name}.csv"
    df = pd.read_csv(path)
    question_col = 'statement' if 'statement' in df.columns else 'raw_question'
    label_col = 'label' if 'label' in df.columns else 'correct_answer'
    if question_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Dataset {dataset_name}.csv must have a question and an answer column.")
    return df, df[question_col].tolist(), df[label_col].tolist()

# --- MODIFIED get_acts Function ---

def get_acts(statements, correct_answers, tokenizer, model, layers, layer_indices, device, enable_llm_extraction=False):
    """
    Attaches hooks, gets activations, checks correctness, and returns all results.
    """
    # 1. Set up hooks
    attn_hooks, mlp_hooks = {}, {}
    handles = []
    for l in layer_indices: 
        hook_a, hook_m = Hook(), Hook()
        handles.extend([
            layers[l].self_attn.register_forward_hook(hook_a),
            layers[l].mlp.register_forward_hook(hook_m)
        ])
        attn_hooks[l], mlp_hooks[l] = hook_a, hook_m

    # 2. Generate all model answers for the batch upfront
    model_name = model.name_or_path if hasattr(model, 'name_or_path') else 'unknown'
    all_model_answers_raw, all_generated_ids = generate_model_answers(
        statements, model, tokenizer, device, model_name, max_new_tokens=64
    )

    # 3. Process the generated answers to get activations
    acts = {2*l: [] for l in layer_indices}
    acts.update({2*l + 1: [] for l in layer_indices})
    batch_correctness, batch_model_answers = [], []

    iterator = zip(statements, correct_answers, all_model_answers_raw, all_generated_ids)
    for stmt, correct_ans, model_answer_text_raw, generated_ids in tqdm(iterator, desc="Processing batch", total=len(statements), leave=False):
        
        model_answer_text = model_answer_text_raw.strip()
        if model_answer_text.lower().startswith(stmt.lower()):
            model_answer_text = model_answer_text[len(stmt):].lstrip(":. ").strip()
        
        batch_model_answers.append(model_answer_text)
        is_correct = check_correctness(model_answer_text, correct_ans)
        batch_correctness.append(is_correct)
        
        exact_answer_str = find_exact_answer_simple(model_answer_text, correct_ans)
        if not exact_answer_str and enable_llm_extraction:
            exact_answer_str = extract_answer_with_llm(stmt, model_answer_text, model, tokenizer)
        
        if not exact_answer_str:
            continue

        # MODIFIED: Call new tokenize function, passing model_name and removing .to(device)
        input_ids = tokenize(stmt, tokenizer, model_name)
        answer_token_indices = find_answer_token_indices_by_string_matching(tokenizer, generated_ids, input_ids, exact_answer_str)
        if answer_token_indices is None:
            continue
        
        # This second forward pass is CRUCIAL to trigger the hooks and get activations
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

    # 4. Clean up and return results
    for k in list(acts.keys()):
        if acts[k]:
            acts[k] = t.stack(acts[k]).cpu().float()
        else:
            del acts[k]

    for h in handles:
        h.remove()
        
    return acts, batch_correctness, batch_model_answers


# --- Main Execution Block (Unchanged) ---

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
    if -1 in li:
        li = list(range(len(layer_modules)))

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
        
        all_correctness_results, all_model_answers = [], []
        
        batch_size = 25
        batch_count = 0 
        for start in tqdm(range(0, len(stmts), batch_size), desc=f"Overall progress for {dataset}"):
            batch_stmts = stmts[start:start + batch_size]
            batch_correct_ans = correct_answers[start:start + batch_size]
            
            acts, batch_correctness, batch_model_ans = get_acts(
                batch_stmts, batch_correct_ans, tokenizer, model, layer_modules, li, args.device, 
                enable_llm_extraction=args.enable_llm_extraction
            )
            
            all_correctness_results.extend(batch_correctness)
            all_model_answers.extend(batch_model_ans)

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
                
                output_csv_path = os.path.join(save_base, f"{dataset}_SUBSAMPLED_with_results.csv")
                df_sub.to_csv(output_csv_path, index=False)
                print(f"✅ Early stop: Saved subsampled results to: {output_csv_path}")
            else:
                print("⚠️ Warning: No results generated during early stop run. No CSV saved.")
        elif num_results == len(df):
            df['model_answer'] = all_model_answers
            df['automatic_correctness'] = all_correctness_results
            
            output_csv_path = os.path.join(save_base, f"{dataset}_with_results.csv")
            df.to_csv(output_csv_path, index=False)
            print(f"✅ Saved full dataset with results to: {output_csv_path}")
        else:
            print(f"⚠️ Warning: Mismatch between results ({num_results}) and dataset rows ({len(df)}). CSV not saved.")