import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import argparse
import pandas as pd
from tqdm import tqdm
import os
import configparser
import glob

# Load configuration
config = configparser.ConfigParser()
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
    """Applies model-specific cleanup logic to the raw generated text."""
    # This cleanup logic is taken directly from your provided script.
    answer_part = decoded_output.split("Exact answer:")[-1]
    model_name_lower = model_name.lower()
    if 'mistral' in model_name_lower:
        return answer_part.replace(".</s>", "").replace("</s>", "").split('\n')[0].split("(")[0].strip().strip(".")
    elif 'llama' in model_name_lower:
        return answer_part.replace(".<|eot_id|>", "").replace("<|eot_id|>", "").split('\n')[-1].split("(")[0].strip().strip(".")
    elif 'gemma' in model_name_lower:
        return answer_part.replace(".<eos>", "").replace("<eos>", "").split('\n')[-1].split("(")[0].strip().strip(".")
    else:
        print(f"Model {model_name} is not explicitly supported for cleanup. Using generic split.")
        return answer_part.split('\n')[0].strip()

# --- Helper functions for finding the answer string ---

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

def extract_answer_with_llm(question, model_answer, extraction_model, extraction_tokenizer):
    """Strategy 2: Use a secondary LLM to extract the most relevant answer."""
    prompt = _create_extraction_prompt(question, model_answer)
    inputs = extraction_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(extraction_model.device)
    
    with t.no_grad():
        outputs = extraction_model.generate(**inputs, max_new_tokens=30, pad_token_id=extraction_tokenizer.eos_token_id)
    
    decoded_output = extraction_tokenizer.decode(outputs[0], skip_special_tokens=False)
    exact_answer = _cleanup_batched_answer(decoded_output, extraction_model.name_or_path)

    # Validate that the extracted answer is a non-empty substring of the original answer
    if exact_answer and exact_answer != "NO ANSWER" and exact_answer.lower() in model_answer.lower():
        return exact_answer
    return None

def find_answer_token_indices(full_sequence_ids, answer_ids):
    """Finds the subsequence of answer token IDs within the full sequence of token IDs."""
    full_list = full_sequence_ids.cpu().numpy().tolist()
    answer_list = answer_ids.cpu().numpy().tolist()
    for i in range(len(full_list) - len(answer_list) + 1):
        if full_list[i:i + len(answer_list)] == answer_list:
            return list(range(i, i + len(answer_list)))
    return None

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

    if model_family.lower() == 'gemma2' and t.cuda.is_available() and t.cuda.is_bf16_supported():
        model = model.to(device=device, dtype=t.bfloat16)
    else:
        model = model.to(device=device, dtype=t.float16 if device.startswith('cuda') else t.float32)

    layers = model.model.layers if hasattr(model, 'model') and hasattr(model.model, 'layers') else model.transformer.layers
    return tokenizer, model, layers

def load_statements(dataset_name):
    """Loads questions and correct answers from a CSV file.
    Accepts either 'statement' or 'raw_question' as the question column.
    """

    path = f"datasets/{dataset_name}.csv"
    df = pd.read_csv(path)

    # Check for the question column
    if 'statement' in df.columns:
        question_col = 'statement'
    elif 'raw_question' in df.columns:
        question_col = 'raw_question'
    else:
        raise ValueError(f"Dataset {dataset_name}.csv must have either 'statement' or 'raw_question' column.")

    if 'correct_answer' not in df.columns:
        raise ValueError(f"Dataset {dataset_name}.csv must have a 'correct_answer' column.")

    return df[question_col].tolist(), df['correct_answer'].tolist()



def get_acts(statements, correct_answers, tokenizer, model, layers, layer_indices, device, extraction_model=None, extraction_tokenizer=None):
    """
    Attaches hooks and gets activations for exact answers using a two-step extraction process.
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

        # --- TWO-STEP EXTRACTION ---
        # 1. Try simple, fast string matching first.
        exact_answer_str = find_exact_answer_simple(model_answer_text, correct_ans)

        # 2. If it fails, and an extraction model is provided, use it as a fallback.
        if not exact_answer_str and extraction_model:
            exact_answer_str = extract_answer_with_llm(stmt, model_answer_text, extraction_model, extraction_tokenizer)

        if not exact_answer_str:
            print(f"\nWarning: Could not find/extract answer for: '{stmt[:60]}...'")
            continue

        exact_answer_ids = tokenizer.encode(exact_answer_str, add_special_tokens=False, return_tensors='pt').to(device)[0]
        answer_token_indices = find_answer_token_indices(generated_ids, exact_answer_ids)

        if answer_token_indices is None:
            print(f"\nWarning: Could not align tokens for answer '{exact_answer_str}'")
            continue

        with t.no_grad():
            model(generated_ids.unsqueeze(0))

        for l in layer_indices:
            a = layers[l].self_attn.register_forward_hook(hook_a).out[0][answer_token_indices].mean(dim=0).detach()
            m = layers[l].mlp.register_forward_hook(hook_m).out[0][answer_token_indices].mean(dim=0).detach()
            acts[2*l].append(a)
            acts[2*l+1].append(m)

    for k in list(acts.keys()):
        if acts[k]: acts[k] = t.stack(acts[k]).cpu().float()
        else: del acts[k]

    for h in handles: h.remove()
    return acts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract activations for exact answers using a dual-strategy approach.")
    parser.add_argument('--model_family', default='Llama3', help='Primary model family')
    parser.add_argument('--model_size', default='8B', help='Primary model size')
    parser.add_argument('--model_type', default='base', help='Primary model type (base or chat)')
    parser.add_argument('--layers', nargs='+', required=True, type=int, help='Layer indices to extract from (-1 for all)')
    parser.add_argument('--datasets', nargs='+', required=True, help='Dataset names (without .csv)')
    parser.add_argument('--output_dir', default='acts', help='Root directory for saving activations')
    parser.add_argument('--device', default='cpu', help='Device to run on (cpu or cuda)')
    # **NEW ARGUMENT** for the optional extraction model
    parser.add_argument('--extraction_model', type=str, default=None, help='Optional: Model name/path for LLM-based answer extraction (e.g., google/gemma-2-2b-it)')
    args = parser.parse_args()

    # --- Dataset expansion logic is unchanged ---
    ds = args.datasets
    if ds == ['all']: ds = [os.path.relpath(fp, 'datasets').replace('.csv','') for fp in glob.glob('datasets/**/*.csv', recursive=True)]

    t.set_grad_enabled(False)

    # Load primary model for activation extraction
    print(f"Loading primary model: {args.model_family} {args.model_size}...")
    tokenizer, model, layer_modules = load_model(args.model_family, args.model_size, args.model_type, args.device)
    
    # Load optional extraction model
    extraction_model, extraction_tokenizer = None, None
    if args.extraction_model:
        print(f"Loading extraction model: {args.extraction_model}...")
        extraction_tokenizer = AutoTokenizer.from_pretrained(args.extraction_model)
        extraction_model = AutoModelForCausalLM.from_pretrained(args.extraction_model)
        if extraction_tokenizer.pad_token is None:
            extraction_tokenizer.pad_token = extraction_tokenizer.eos_token
        extraction_model.to(args.device) # Move to the same device
        print("Extraction model loaded.")
    else:
        print("No extraction model provided. Using simple string matching only.")

    li = args.layers
    if li == [-1]: li = list(range(len(layer_modules)))

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
            
            # Pass the optional extraction model and tokenizer to the main function
            acts = get_acts(batch_stmts, batch_correct_ans, tokenizer, model, layer_modules, li, args.device, extraction_model, extraction_tokenizer)
            
            if not acts:
                print(f"No valid activations collected for batch in {dataset} (starts at index {start}).")
                continue

            for pseudo, tensor in acts.items():
                filename = os.path.join(save_base, f"layer_{pseudo}_{start}.pt")
                t.save(tensor, filename)