import os
import json
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist
from datasets import load_dataset,load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.generation_utils import (
    set_seed,
    get_math_symbols_ids,
    generate_cot,
    generate_selar,
    generate_swir,
)
from src.grader import answer_match


def main(args):
    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    model_name = args.model_name
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens
    n_samples = args.n_samples
    method = args.method
    
    # SWIR-specific parameters
    alpha = args.alpha
    max_switch_count = args.max_switch_count
    
    # SeLaR-specific parameters
    selar_topk = args.selar_topk
    entropy_threshold = args.entropy_threshold

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={"": local_rank}
    )
    
    if dataset_name == "gsm8k":
        dataset = load_from_disk("datasets/gsm8k_test")
    elif dataset_name == "math500":
        dataset = load_from_disk("datasets/math_500_test")
    elif dataset_name == "aime_2024":
        dataset = load_from_disk("datasets/aime_2024_train")
    elif dataset_name == "aime_2025":
        dataset = load_from_disk("datasets/aime_2025")
    elif dataset_name == "gpqa_diamond":
        dataset = load_from_disk("datasets/gpqa_diamond_mc_test")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if n_samples is not None:
        dataset = dataset.select(range(n_samples))
    total_len = len(dataset)
    chunk_size = (total_len + world_size - 1) // world_size
    start = local_rank * chunk_size
    end = min(start + chunk_size, total_len)
    dataset = dataset.select(range(start, end))
    
    correct = 0
    total = 0
    details = []
    total_token_lens = []
    correct_token_lens = []
    wrong_token_lens = []

    math_symbols_ids = get_math_symbols_ids(tokenizer)
    math_ids_tensor = torch.tensor(list(math_symbols_ids), device=model.device)
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        if args.dataset_name == "gsm8k":
            questions = batch["question"]
            golds = [str(a).split("####")[-1].strip() for a in batch["answer"]]
        elif args.dataset_name == "math500":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["answer"]]
        elif args.dataset_name == "aime_2024":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["answer"]]
        elif args.dataset_name == "aime_2025":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["answer"]]
        elif args.dataset_name == "gpqa_diamond":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["solution"]]
        prompts = [
            f"{q}\nPlease reason step by step, and make sure put your final answer within \\boxed{{}}."
            for q in questions
        ]
        messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for messages in messages_batch
        ]
        model_inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
    
        with torch.no_grad():
            if method == "cot":
                generated_ids = generate_cot(
                    model,
                    tokenizer,
                    **model_inputs,   
                    **gen_kwargs,   
                )
            elif method == "cot_greedy":
                gen_kwargs["do_sample"] = False
                generated_ids = generate_cot(
                    model,
                    tokenizer,
                    **model_inputs,   
                    **gen_kwargs,   
                )
            elif method == "selar":
                # SeLaR-specific parameters
                model_inputs["selar_topk"] = selar_topk
                model_inputs["entropy_threshold"] = entropy_threshold
                model_inputs["math_ids_tensor"] = math_ids_tensor
                generated_ids = generate_selar(
                    model,
                    tokenizer,
                    **model_inputs,
                    **gen_kwargs,
                )
            elif method == "swir":
                # SWIR-specific parameters
                model_inputs["alpha_0"] = alpha
                model_inputs["max_switch_count"] = max_switch_count
                model_inputs["math_ids_tensor"] = math_ids_tensor
                model_inputs["convergence_words"] = "</think>" if "Qwen" in model_name else "\n\n</think>\n\n"
                generated_ids = generate_swir(
                    model,
                    tokenizer,
                    **model_inputs,   
                    **gen_kwargs,   
                )
        
        prompt_len = model_inputs["input_ids"].shape[1]
        preds = [
            tokenizer.decode(generated_ids[idx][prompt_len:], skip_special_tokens=True)
            for idx in range(len(questions))
        ]
    
        for idx in range(len(questions)):
            gold = golds[idx]
            question = questions[idx]
            pred = preds[idx]
            output_ids = generated_ids[idx][prompt_len:].tolist()
            try:
                eot_id = 128014 if "Llama" in model_name else 151668
                index = len(output_ids) - output_ids[::-1].index(eot_id)
            except ValueError:
                index = 0
            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
            answer_content = pred[len(thinking_content):]
            is_correct, prediction = answer_match(dataset_name, answer_content, gold)
            correct += int(is_correct)
            total += 1
            details.append({
                "question": question,
                "gold": gold,
                "prediction": prediction,
                "correct": is_correct,
                "thinking": thinking_content,
                "answer_content": answer_content,
            })
            if total % 20 == 0:
                print(f"Processed {total} examples, Accuracy: {correct/total:.2%}")
                
            output_token_ids = tokenizer.encode(pred, add_special_tokens=False)
            total_token_len = len(output_token_ids)
            total_token_lens.append(total_token_len)
            if is_correct:
                correct_token_lens.append(total_token_len)
            else:
                wrong_token_lens.append(total_token_len)

    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct/total:.2%}")
    
    avg = lambda l: float(sum(l)) / len(l) if l else 0.0
    length_stats = {
        "max_new_tokens": max_new_tokens,
        "avg_total_token_len": avg(total_token_lens),
        "correct_avg_total_token_len": avg(correct_token_lens),
        "wrong_avg_total_token_len": avg(wrong_token_lens),
    }
    
    result = {
        "accuracy": correct / total if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        "length_stats": length_stats,
        "details": details
    }
    
    os.makedirs("logs", exist_ok=True)
    model_name = model_name.split("/")[-1]
    
    # Create method-specific log filename
    if method == "selar":
        log_path = f"logs/{model_name}_{dataset_name}_{method}_k{selar_topk}_t{entropy_threshold}_{max_new_tokens}_rank{local_rank}.json"
    elif method == "swir":
        log_path = f"logs/{model_name}_{dataset_name}_{method}_a{alpha}_s{max_switch_count}_{max_new_tokens}_rank{local_rank}.json"
    else:
        log_path = f"logs/{model_name}_{dataset_name}_{method}_{max_new_tokens}_rank{local_rank}.json"
    
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Rank {local_rank}] log written: {log_path}")


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--dataset_name', type=str, default="gsm8k")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_samples', type=int, default=None) 
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--do_sample", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--max_new_tokens', type=int, default=38912)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument("--method", type=str, default="selar", choices=["selar", "swir", "cot", "cot_greedy"])
    
    # SWIR-specific parameters
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--max_switch_count', type=int, default=None)
    
    # SeLaR-specific parameters
    parser.add_argument('--selar_topk', type=int, default=3, help='Top-k for entropy computation in SeLaR')
    parser.add_argument('--entropy_threshold', type=float, default=0.5, help='Entropy threshold for SeLaR interventions')

    
    args = parser.parse_args()
    main(args)
