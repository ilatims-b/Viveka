import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import argparse
import pandas as pd
from tqdm import tqdm
import os
import glob
from thefuzz import process, fuzz

# --- Helper Functions ---

def check_correctness(model_answer, correct_answers_list):
    """Checks if any of the possible correct answers are in the model's answer."""
    # Ensure the list of correct answers is actually a list
    if isinstance(correct_answers_list, str):
        try:
            labels_ = eval(correct_answers_list)
        except:
            labels_ = [correct_answers_list] # Fallback for simple strings
    else:
        labels_ = correct_answers_list

    if not isinstance(labels_, list):
        labels_ = [str(labels_)]

    if not isinstance(model_answer, str):
        return 0 # Cannot be correct if there's no answer

    for ans in labels_:
        if str(ans).lower() in model_answer.lower():
            return 1 # Correct
    return 0 # Incorrect

def _create_extraction_prompt(raw_question, model_answer):
    """Creates the standard f-string prompt for the LLM extractor."""
    return f"""
        Extract the exact answer from the long answer. If the long answer doesn't answer the question, return “NO ANSWER.” Ignore factual correctness; extract what appears most relevant.

        Examples:
        Q: Which musical featured the song The Street Where You Live?
        A: The song "The Street Where You Live" is from the Lerner and Loewe musical "My Fair Lady."
        Exact answer: My Fair Lady

        Q: Which Swedish actress won the Best Supporting Actress Oscar for Murder on the Orient Express?
        A: No Swedish actress has won an Oscar for Best Supporting Actress for that film.
        Exact answer: NO ANSWER

        Q: Who wrote Philosophiæ Naturalis Principia Mathematica?
        A: Albert Einstein
        Exact answer: Albert Einstein

        Now extract for this:
        Q: {raw_question}
        A: {model_answer}
        Exact answer:
    """

def _cleanup_batched_answer(decoded_output, model_name):
    """Applies model-specific cleanup logic to the raw generated text."""
    if "Exact answer:" in decoded_output:
        answer_part = decoded_output.split("Exact answer:")[-1]
    else:
        answer_part = decoded_output

    tokens_to_remove = [".</s>", "</s>", ".<|eot_id|>", "<|eot_id|>", ".<eos>", "<eos>", "<end_of_turn>"]
    for token in tokens_to_remove:
        answer_part = answer_part.replace(token, "")

    for line in answer_part.strip().split('\n'):
        cleaned_line = line.strip()
        if cleaned_line:
            return cleaned_line.split("(")[0].strip().strip(".")
    return ""

def find_exact_answer_simple(model_answer: str, correct_answer):
    """Strategy 1: Fast-path logic using simple string searching."""
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
    """Strategy 2: Use the provided LLM to extract the most relevant answer."""
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
    """Finds answer tokens by performing a fuzzy string match on the fully decoded text."""
    try:
        full_decoded_text = tokenizer.decode(full_generated_ids, skip_special_tokens=False)
    except:
        return None

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
    def __init__(self):
        self.out = None
    def __call__(self, module, module_inputs, module_outputs):
        self.out = module_outputs[0] if isinstance(module_outputs, tuple) else module_outputs

def load_model(model_family: str, model_size: str, model_type: str, device: str):
    """Loads the primary model and tokenizer directly from the Hugging Face Hub."""
    if model_family.lower() == 'gemma2':
        organization = 'google'
        size_str = model_size.lower()
        type_str = '-it' if model_type.lower() in ['chat', 'instruct'] else ''
        model_name = f"gemma-{size_str}{type_str}"
    elif model_family.lower() == 'llama3':
        organization = 'meta-llama'
        type_str = "-instruct" if model_type.lower() in ['chat', 'instruct'] else ""
        model_name = f"Meta-Llama-3-{model_size}{type_str}"
    else:
        raise ValueError(f"Model family '{model_family}' is not configured for Hub loading.")

    model_path = f"{organization}/{model_name}"
    print(f"Loading from Hugging Face Hub: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=t.bfloat16 if t.cuda.is_available() and t.cuda.is_bf16_supported() else t.float16
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    layers = getattr(getattr(model, 'model', None), 'layers', None)
    if layers is None:
        raise AttributeError(f"Could not find layers for model {model_path}. Please check the model architecture.")
        
    return tokenizer, model, layers

def load_statements(dataset_name):
    """Loads questions and correct answers from a CSV file."""
    path = f"datasets/{dataset_name}.csv"
    df = pd.read_csv(path)
    question_col = 'statement' if 'statement' in df.columns else 'raw_question'
    label_col = 'label' if 'label' in df.columns else 'correct_answer'
    if question_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Dataset {dataset_name}.csv must have a question and an answer column.")
    return df, df[question_col].tolist(), df[label_col].tolist()

# --- Main Activation & Correctness Logic ---

def get_acts(statements, correct_answers, tokenizer, model, layers, layer_indices, device, enable_llm_extraction=False):
    """
    Attaches hooks, gets activations, checks correctness, and returns all results.
    """
    attn_hooks, mlp_hooks = {}, {}
    handles = []
    for l in layer_indices: 
        hook_a, hook_m = Hook(), Hook()
        handles.extend([
            layers[l].self_attn.register_forward_hook(hook_a),
            layers[l].mlp.register_forward_hook(hook_m)
        ])
        attn_hooks[l], mlp_hooks[l] = hook_a, hook_m

    acts = {2*l: [] for l in layer_indices}
    acts.update({2*l + 1: [] for l in layer_indices})
    
    batch_correctness, batch_model_answers = [], []

    desc = "Processing batch"
    for stmt, correct_ans in tqdm(zip(statements, correct_answers), desc=desc, total=len(statements), leave=False):
        input_ids = tokenizer.encode(stmt, return_tensors='pt', add_special_tokens=True).to(model.device)

        with t.no_grad():
            generated_ids = model.generate(input_ids, max_new_tokens=64, pad_token_id=tokenizer.pad_token_id)[0]
        
        model_answer_text = tokenizer.decode(generated_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        is_correct = check_correctness(model_answer_text, correct_ans)
        batch_model_answers.append(model_answer_text)
        batch_correctness.append(is_correct)
        
        exact_answer_str = find_exact_answer_simple(model_answer_text, correct_ans)
        if not exact_answer_str and enable_llm_extraction:
            exact_answer_str = extract_answer_with_llm(stmt, model_answer_text, model, tokenizer)
        
        if not exact_answer_str:
            continue

        answer_token_indices = find_answer_token_indices_by_string_matching(tokenizer, generated_ids, input_ids, exact_answer_str)
        if answer_token_indices is None:
            continue

        with t.no_grad():
            model(generated_ids.unsqueeze(0))

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
        if acts[k]:
            acts[k] = t.stack(acts[k]).cpu().float()
        else:
            del acts[k]

    for h in handles:
        h.remove()
        
    return acts, batch_correctness, batch_model_answers

# --- Main Execution Block ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract activations and check correctness from Hugging Face models.")
    parser.add_argument('--model_family', default='Gemma2', help='Primary model family (e.g., Gemma2, Llama3)')
    parser.add_argument('--model_size', default='2B', help='Primary model size (e.g., 2B, 8B)')
    parser.add_argument('--model_type', default='chat', help='Primary model type (base, chat, or instruct)')
    parser.add_argument('--layers', nargs='+', required=True, type=int, help='Layer indices to extract from (-1 for all)')
    parser.add_argument('--datasets', nargs='+', required=True, help='Dataset names (without .csv)')
    parser.add_argument('--output_dir', default='acts_output', help='Root directory for saving all outputs')
    parser.add_argument('--device', default='cuda' if t.cuda.is_available() else 'cpu', help='Device to run on (cpu or cuda)')
    parser.add_argument('--enable_llm_extraction', action='store_true', help='Enable LLM-based answer extraction as a fallback.')
    args = parser.parse_args()

    ds = args.datasets
    if ds == ['all']:
        ds = [os.path.relpath(fp, 'datasets').replace('.csv', '') for fp in glob.glob('datasets/**/*.csv', recursive=True)]

    t.set_grad_enabled(False)

    tokenizer, model, layer_modules = load_model(args.model_family, args.model_size, args.model_type, args.device)
    
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
            
        save_base = os.path.join(args.output_dir, args.model_family, args.model_size, args.model_type, dataset)
        os.makedirs(save_base, exist_ok=True)
        
        all_correctness_results, all_model_answers = [], []
        
        batch_size = 25
        for start in tqdm(range(0, len(stmts), batch_size), desc=f"Overall progress for {dataset}"):
            batch_stmts = stmts[start:start + batch_size]
            batch_correct_ans = correct_answers[start:start + batch_size]
            
            acts, batch_correctness, batch_model_ans = get_acts(
                batch_stmts, batch_correct_ans, tokenizer, model, layer_modules, li, args.device, 
                enable_llm_extraction=args.enable_llm_extraction
            )
            
            all_correctness_results.extend(batch_correctness)
            all_model_answers.extend(batch_model_ans)

            if not acts:
                # print(f"No valid activations collected for batch starting at index {start}.")
                pass
            else:
                for pseudo, tensor in acts.items():
                    filename = os.path.join(save_base, f"layer_{pseudo}_{start}.pt")
                    t.save(tensor, filename)
        
        if len(all_correctness_results) == len(df):
            df['model_answer'] = all_model_answers
            df['automatic_correctness'] = all_correctness_results
            
            output_csv_path = os.path.join(save_base, f"{dataset}_with_results.csv")
            df.to_csv(output_csv_path, index=False)
            print(f"✅ Saved dataset with results to: {output_csv_path}")
        else:
            print(f"⚠️ Warning: Mismatch between results ({len(all_correctness_results)}) and dataset rows ({len(df)}). CSV not saved.")