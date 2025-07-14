import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import argparse
import pandas as pd
from tqdm import tqdm
import os
import configparser
import glob
from thefuzz import process, fuzz

# Load configuration
config = configparser.ConfigParser()
# A dummy config.ini might be needed to run this script directly
if not os.path.exists('config.ini'):
    with open('config.ini', 'w') as f:
        f.write('[Llama3]\nweights_directory=./\n8B_base_subdir=.\n')
config.read('config.ini')


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
    """
    Applies model-specific cleanup logic to the raw generated text.
    **FIXED**: Made this function more robust to handle various special tokens
    and cases where the 'Exact answer:' prompt is not followed.
    """
    if "Exact answer:" in decoded_output:
        answer_part = decoded_output.split("Exact answer:")[-1]
    else:
        answer_part = decoded_output

    # Generic cleanup for all known special/EOS tokens, including <end_of_turn>
    tokens_to_remove = [
        ".</s>", "</s>",
        ".<|eot_id|>", "<|eot_id|>",
        ".<eos>", "<eos>",
        "<end_of_turn>"
    ]
    for token in tokens_to_remove:
        answer_part = answer_part.replace(token, "")

    # Further cleanup: take the first non-empty line and clean it
    for line in answer_part.strip().split('\n'):
        cleaned_line = line.strip()
        if cleaned_line:
            return cleaned_line.split("(")[0].strip().strip(".")
            
    return "" # Return empty if no answer is found

def find_exact_answer_simple(model_answer: str, correct_answer):
    """Strategy 1: Fast-path logic using simple string searching."""
    if not isinstance(model_answer, str):
        return None
    try:
        if isinstance(correct_answer, str):
            correct_answer_eval = eval(correct_answer)
            if isinstance(correct_answer_eval, list):
                correct_answer = correct_answer_eval
    except (SyntaxError, NameError):
        pass

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

    if exact_answer and exact_answer.upper() != "NO ANSWER":
        return exact_answer
    return None

def find_answer_token_indices_by_string_matching(tokenizer, full_generated_ids, prompt_ids, exact_answer_str):
    """
    Finds answer tokens by performing a fuzzy string match on the decoded text of the
    *generated portion only* and then maps character positions back to token IDs.
    
    **REWRITTEN**: This function is now much more robust.
    1.  Isolates the generated text to avoid prompt contamination.
    2.  Uses fuzzy partial matching to handle markdown and other formatting.
    3.  Ensures offset mapping is generated without adding extra special tokens.
    4.  Correctly calculates absolute token indices relative to the full sequence.
    """
    # 1. Isolate the token IDs for the generated answer part
    answer_ids = full_generated_ids[prompt_ids.shape[1]:]
    if len(answer_ids) == 0: return None
    
    # 2. Decode *only the answer part* to get the text we will search in.
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=False)

    # 3. Fuzzy String Search using partial_ratio for better substring matching
    match = process.extractOne(
        exact_answer_str,
        [answer_text],
        scorer=fuzz.partial_ratio,
        score_cutoff=90
    )

    if not match: return None
    best_match_in_answer_text = match[0]

    # 4. Find the character start/end of the match within the decoded answer_text
    start_char = answer_text.find(best_match_in_answer_text)
    if start_char == -1: return None
    end_char = start_char + len(best_match_in_answer_text)

    # 5. Get character-to-token offset mapping for the answer_text.
    # CRITICAL: `add_special_tokens=False` prevents altering character offsets.
    encoding = tokenizer(
        answer_text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    offset_mapping = encoding['offset_mapping']
    
    # 6. Find all tokens from the answer that overlap with the character span of the match.
    relative_token_indices = []
    for i, (token_start, token_end) in enumerate(offset_mapping):
        if token_start < end_char and start_char < token_end:
            relative_token_indices.append(i)
            
    if not relative_token_indices: return None

    # 7. Convert relative token indices (within the answer) to absolute indices
    # (within the full generated sequence) by adding the prompt's token length.
    prompt_length_in_tokens = prompt_ids.shape[1]
    absolute_token_indices = [i + prompt_length_in_tokens for i in relative_token_indices]
    
    return absolute_token_indices

class Hook:
    def __init__(self):
        self.out = None
    def __call__(self, module, module_inputs, module_outputs):
        self.out = module_outputs[0] if isinstance(module_outputs, tuple) else module_outputs

def load_model(model_family: str, model_size: str, model_type: str, device: str):
    """Loads the primary model whose activations will be probed."""
    model_path = os.path.join(
        config[model_family]['weights_directory'],
        config[model_family][f'{model_size}_{model_type}_subdir']
    )
    if model_family == 'Llama2':
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path)
        tokenizer.bos_token = '<s>'
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Simplified dtype handling for broader compatibility
    model = model.to(device=device)
    return tokenizer, model, model.model.layers

def load_statements(dataset_name):
    """Loads questions and correct answers from a CSV file."""
    path = f"datasets/{dataset_name}.csv"
    df = pd.read_csv(path)
    question_col = 'statement' if 'statement' in df.columns else 'raw_question'
    label_col = 'label' if 'label' in df.columns else 'correct_answer'
    if question_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Dataset {dataset_name}.csv must have a question and a answer column.")
    return df[question_col].tolist(), df[label_col].tolist()


def get_acts(statements, correct_answers, tokenizer, model, layers, layer_indices, device, enable_llm_extraction=False):
    """
    Attaches hooks and gets activations for exact answers using a robust,
    string-matching-based extraction process with built-in debugging.
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

    desc = "Finding answers and extracting activations"
    for stmt, correct_ans in tqdm(zip(statements, correct_answers), desc=desc, total=len(statements)):
        input_ids = tokenizer.encode(stmt, return_tensors='pt', add_special_tokens=True).to(device)

        with t.no_grad():
            generated_ids = model.generate(input_ids, max_new_tokens=64, pad_token_id=tokenizer.pad_token_id)[0]
        
        model_answer_text = tokenizer.decode(generated_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        exact_answer_str = find_exact_answer_simple(model_answer_text, correct_ans)

        if not exact_answer_str and enable_llm_extraction:
            exact_answer_str = extract_answer_with_llm(stmt, model_answer_text, model, tokenizer)
        
        if not exact_answer_str:
            print(f"\n--- DEBUG: FAILED [Step 1: Could not find answer string] ---\n - Question: '{stmt[:80]}...'\n - Model's Answer: '{model_answer_text}'")
            continue

        # **MODIFIED CALL**: Pass the full sequence, the prompt, and the clean answer string.
        answer_token_indices = find_answer_token_indices_by_string_matching(
            tokenizer,
            generated_ids,      # The full generated sequence
            input_ids,          # The prompt sequence
            exact_answer_str
        )

        if answer_token_indices is None:
            print(f"\n--- DEBUG: FAILED [Step 2: Could not map string to tokens] ---\n - Question: '{stmt[:80]}...'\n - Model's Answer: '{model_answer_text}'\n - String We Found: '{exact_answer_str}' (but failed to map)")
            continue

        with t.no_grad():
            model(generated_ids.unsqueeze(0))

        for l in layer_indices:
            a = attn_hooks[l].out[0][answer_token_indices].mean(dim=0).detach()
            m = mlp_hooks[l].out[0][answer_token_indices].mean(dim=0).detach()
            acts[2*l].append(a)
            acts[2*l+1].append(m)

    for k in list(acts.keys()):
        if acts[k]:
            acts[k] = t.stack(acts[k]).cpu().float()
        else:
            del acts[k]

    for h in handles:
        h.remove()
        
    return acts

# Main execution block remains the same

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract activations for exact answers using a single model.")
    parser.add_argument('--model_family', default='Llama3', help='Primary model family')
    parser.add_argument('--model_size', default='8B', help='Primary model size')
    parser.add_argument('--model_type', default='base', help='Primary model type (base or chat)')
    parser.add_argument('--layers', nargs='+', required=True, type=int, help='Layer indices to extract from (-1 for all)')
    parser.add_argument('--datasets', nargs='+', required=True, help='Dataset names (without .csv)')
    parser.add_argument('--output_dir', default='acts', help='Root directory for saving activations')
    parser.add_argument('--device', default='cpu', help='Device to run on (cpu or cuda)')
    parser.add_argument('--enable_llm_extraction', action='store_true', help='Enable using the primary model for LLM-based answer extraction as a fallback to string matching.')
    args = parser.parse_args()

    ds = args.datasets
    if ds == ['all']:
        ds = [os.path.relpath(fp, 'datasets').replace('.csv', '') for fp in glob.glob('datasets/**/*.csv', recursive=True)]

    t.set_grad_enabled(False)

    print(f"Loading primary model: {args.model_family} {args.model_size}...")
    tokenizer, model, layer_modules = load_model(args.model_family, args.model_size, args.model_type, args.device)
    
    if args.enable_llm_extraction:
        print("LLM-based extraction is ENABLED (will use the primary model).")
    else:
        print("LLM-based extraction is DISABLED. Using simple string matching only.")

    li = args.layers
    if -1 in li:
        li = list(range(len(layer_modules)))

    for dataset in ds:
        try:
            stmts, correct_answers = load_statements(dataset)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping dataset '{dataset}': {e}")
            continue
            
        save_base = os.path.join(args.output_dir, args.model_family, args.model_size, args.model_type, dataset)
        os.makedirs(save_base, exist_ok=True)
        
        batch_size = 25
        for start in range(0, len(stmts), batch_size):
            batch_stmts = stmts[start:start + batch_size]
            batch_correct_ans = correct_answers[start:start + batch_size]
            
            acts = get_acts(batch_stmts, batch_correct_ans, tokenizer, model, layer_modules, li, args.device, enable_llm_extraction=args.enable_llm_extraction)
            
            if not acts:
                print(f"No valid activations collected for batch in {dataset} (starts at index {start}).")
                continue

            for pseudo, tensor in acts.items():
                filename = os.path.join(save_base, f"layer_{pseudo}_{start}.pt")
                t.save(tensor, filename)
