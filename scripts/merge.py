import os
import glob
import json
import argparse


def main(args):
    
    model_name = args.model_name
    model_name = model_name.split("/")[-1]
    dataset_name = args.dataset_name
    max_new_tokens = args.max_new_tokens
    method = args.method

    print("[Rank 0] All logs written, start merging...")
    all_details = []
    total_correct = 0
    total_samples = 0
    
    total_token_sum = 0
    correct_token_sum = 0
    wrong_token_sum = 0
    total_token_cnt = 0
    correct_token_cnt = 0
    wrong_token_cnt = 0

    all_log_paths = list(glob.glob(f"logs/{model_name}_{dataset_name}_{method}_{max_new_tokens}_rank*.json"))
    for path in all_log_paths:
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
            total_correct += result["correct"]
            total_samples += result["total"]
            all_details.extend(result["details"])
            ls = result["length_stats"]
            total_token_sum += ls.get("avg_total_token_len", 0) * result["total"]
            total_token_cnt += result["total"]
            correct_token_sum += ls.get("correct_avg_total_token_len", 0) * result["correct"]
            correct_token_cnt += result["correct"]
            wrong_cnt = result["total"] - result["correct"]
            wrong_token_sum += ls.get("wrong_avg_total_token_len", 0) * wrong_cnt
            wrong_token_cnt += wrong_cnt

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    merged_length_stats = {
        "max_new_tokens": max_new_tokens,
        "avg_total_token_len": float(total_token_sum) / total_token_cnt if total_token_cnt else 0.0,
        "correct_avg_total_token_len": float(correct_token_sum) / correct_token_cnt if correct_token_cnt else 0.0,
        "wrong_avg_total_token_len": float(wrong_token_sum) / wrong_token_cnt if wrong_token_cnt else 0.0,
    }
    merged_result = {
        "accuracy": accuracy,
        "total": total_samples,
        "correct": total_correct,
        "length_stats": merged_length_stats,
        "details": all_details,
    }
    with open(f"logs/{model_name}_{dataset_name}_{method}_{max_new_tokens}_merged.json", "w", encoding="utf-8") as f:
        json.dump(merged_result, f, ensure_ascii=False, indent=2)
    print(f"[Rank 0] Merged results saved. Accuracy: {accuracy:.2%}, Length: {merged_length_stats}")

    for path in all_log_paths:
        try:
            os.remove(path)
        except Exception as e:
            print(f"Failed to delete {path}: {e}")


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--dataset_name', type=str, default="gsm8k")
    parser.add_argument('--max_new_tokens', type=int, default=38912)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--method", type=str, default="swir", choices=["swir", "cot", "cot_greedy","selar"])
    args = parser.parse_args()
    main(args)
