from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
from latex2sympy2_extended import NormalizationConfig


def gsm8k_grader(solution_str: str, ground_truth: str) -> bool:
    # if not ground_truth.startswith("$"):
    #     ground_truth = f"${ground_truth}$"
    gold = parse(
        ground_truth,
        extraction_config=[ExprExtractionConfig()],
    )
    answer = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )
    if len(answer) == 0:
        return False, "No extracted answer"
    else:
        return verify(gold, answer), str(answer)


def math500_grader(solution_str: str, ground_truth: str) -> bool:
    if not ground_truth.startswith("$"):
        ground_truth = f"${ground_truth}$"
    gold = parse(
        ground_truth,
        extraction_config=[LatexExtractionConfig()],
    )
    answer = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )
    if len(answer) == 0:
        return False, "No extracted answer"
    else:
        return verify(gold, answer), str(answer)


def aime_grader(solution_str: str, ground_truth: str) -> bool:
    # if not ground_truth.startswith("$"):
    #     ground_truth = f"${ground_truth}$"
    gold = parse(
        ground_truth,
        extraction_config=[ExprExtractionConfig()],
    )
    answer = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )
    if len(answer) == 0:
        return False, "No extracted answer"
    else:
        return verify(gold, answer), str(answer)
            

def gpqa_grader(solution_str: str, ground_truth: str) -> bool:
    if not ground_truth.startswith("$"):
        ground_truth = f"${ground_truth}$"
    gold = parse(
        ground_truth,
        extraction_config=[LatexExtractionConfig()],
    )
    answer = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )
    if len(answer) == 0:
        return False, "No extracted answer"
    else:
        return verify(gold, answer), str(answer)


def answer_extraction(pred):
    return gsm8k_grader(pred, None)


def answer_match(dataset_name, pred, gold):
    if dataset_name == "gsm8k":
        return gsm8k_grader(pred, gold)
    elif dataset_name == "math500":
        return math500_grader(pred, gold)
    elif "aime" in dataset_name:
        return aime_grader(pred, gold)
    elif "gpqa" in dataset_name:
        return gpqa_grader(pred, gold)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    