from utils import encode, generate, create_prompts, generate_model_answers, check_correctness, find_exact_answer_simple, extract_answer_direct, is_vague_or_non_answer, extract_answer_with_llm, _cleanup_extracted_answer, load_model, load_statements
from hook import Hook, get_resid_acts
import argparse
from tqdm import tqdm
import os
import glob
from thefuzz import process, fuzz
from transformers import (AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer,
                          LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList)
import torch as t
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract activations and check correctness from Hugging Face models.")
    parser.add_argument('--model_repo_id', type=str, required=True, help='The exact Hugging Face Hub repository ID (e.g., google/gemma-2b-it)')
    parser.add_argument('--layers', nargs='+', required=True, type=int, help='Layer indices to extract from (-1 for all)')
    parser.add_argument('--datasets', nargs='+', required=True, help='Dataset names (without .csv)')
    parser.add_argument('--output_dir', default='acts_output', help='Root directory for saving all outputs')
    parser.add_argument('--device', default='cuda' if t.cuda.is_available() else 'cpu', help='Device to run on (cpu or cuda)')
    parser.add_argument('--enable_llm_extraction', action='store_true', help='Enable LLM-based answer extraction as a fallback.')
    parser.add_argument('--early_stop', action='store_true', help='Process only two batches and save a subsampled CSV for quick checks.')
    parser.add_argument('--num_generations', type=int, default=30, help='Number of generations per prompt (default: 30)')
    parser.add_argument('--batch_size', type=int, default=25, help='Batch size for processing statements (default: 25)')
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
        
        batch_count = 0 
        for start in tqdm(range(0, len(stmts), args.batch_size), desc=f"Overall progress for {dataset}"):
            batch_stmts = stmts[start:start + args.batch_size]
            batch_correct_ans = correct_answers[start:start + args.batch_size]
            
            # Use the modified function for multiple generations
            acts, batch_correctness, batch_model_ans, batch_exact_ans = get_resid_acts(
                batch_stmts, batch_correct_ans, tokenizer, model, layer_modules, li, args.device, 
                num_generations=args.num_generations,
                enable_llm_extraction=args.enable_llm_extraction
            )
            
            all_correctness_results.extend(batch_correctness)
            all_model_answers.extend(batch_model_ans)
            all_exact_answers.extend(batch_exact_ans)

            if acts:
                for layer_idx, tensor in acts.items():
                    filename = os.path.join(save_base, f"layer_{layer_idx}_{start}.pt")
                    t.save(tensor, filename)
            
            batch_count += 1
            if args.early_stop and batch_count >= 2:
                print(f"\nEarly stopping after {batch_count} batches.")
                break
        
        num_results = len(all_model_answers)
        if args.early_stop:
            if num_results > 0:
                df_sub = df.iloc[:num_results].copy()
                df_sub['model_answers'] = all_model_answers  # Now contains lists
                df_sub['automatic_correctness'] = all_correctness_results  # Now contains lists
                df_sub['exact_answers'] = all_exact_answers  # Now contains lists
                
                output_csv_path = os.path.join(save_base, f"{dataset}_SUBSAMPLED_with_results.csv")
                df_sub.to_csv(output_csv_path, index=False, encoding='utf-8')
                print(f" Early stop: Saved subsampled results to: {output_csv_path}")
            else:
                print(" Warning: No results generated during early stop run. No CSV saved.")
        elif num_results == len(df):
            df['model_answers'] = all_model_answers  # Now contains lists
            df['automatic_correctness'] = all_correctness_results  # Now contains lists  
            df['exact_answers'] = all_exact_answers  # Now contains lists
            
            output_csv_path = os.path.join(save_base, f"{dataset}_with_results.csv")
            df.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"Saved full dataset with results to: {output_csv_path}")
        else:
            print(f"Warning: Mismatch between results ({num_results}) and dataset rows ({len(df)}). CSV not saved.")
            
        print(f"Dataset {dataset} summary:")
        print(f"   - Total statements processed: {num_results}")
        print(f"   - Generations per statement: {args.num_generations}")
        print(f"   - Total generations: {num_results * args.num_generations}")
        if acts:
            total_activations = sum(len(tensor) for tensor in acts.values())
            print(f"   - Total activations saved: {total_activations}")
            print(f"   - Layers with activations: {list(acts.keys())}")