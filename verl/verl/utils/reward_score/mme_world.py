import os
import re
import random
import json
from .code_exec_utils import rebuild_solution_str


def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']

    if method == 'strict':
        pattern = r"<ANSWER>(.*)</ANSWER>"
        solution = re.search(pattern, solution_str, re.DOTALL)

        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(1).strip()
    elif method == 'flexible':
        raise NotImplementedError
    return final_answer


def get_type(solution_str):
    if solution_str.startswith("<COT>"):
        return "COT"
    elif solution_str.startswith("<CODE>"):
        return "CODE"
    elif solution_str.startswith("<LONG_COT>"):
        return "LONG_COT"
    elif solution_str.startswith("<ANSWER>"):
        return "DIRECT"
    else:
        return "UNKNOWN"


def compute_score(solution_str, ground_truth, code_executalbe, output_file, method='strict', format_score=0., score=1.):
    _type = get_type(solution_str)
    code_executed = None
    if _type == "CODE" and code_executalbe:
        solution_str, code_executed = rebuild_solution_str(solution_str)
    answer = extract_solution(solution_str=solution_str, method=method)
    actual_score = 0
    do_print = random.randint(1, 512) == 1
    if do_print:
        print("--------------------")
        print(f"solution_str: {solution_str}")
        print(f"answer: {answer}, ground_truth: {ground_truth}")
        print("--------------------")

    if answer is None:
        if do_print:
            print("Invalid answer")
        actual_score = 0
    else:
        if answer == ground_truth:
            if do_print:
                print("Correct answer")
            actual_score = score
        else:
            if do_print:
                print("Incorrect answer")
            actual_score = format_score

    # format
    if not any(solution_str.startswith(tag) for tag in ("<CODE>", "<COT>", "<ANSWER>", "<LONG_COT>")):
        if do_print:
            print("Invalid format")
        actual_score = 0.
    else:
        invalid_combinations = {
            "<CODE>": {"<COT>", "<LONG_COT>", "</COT>", "</LONG_COT>"},
            "<COT>": {"<CODE>", "<LONG_COT>", "</CODE>", "</LONG_COT>"},
            "<LONG_COT>": {"<CODE>", "<COT>", "</CODE>", "</COT>"},
            "<ANSWER>": {"<CODE>", "<COT>", "<LONG_COT>", "</CODE>", "</COT>", "</LONG_COT>"},
        }

        for key, invalid_tags in invalid_combinations.items():
            if solution_str.startswith(key) and any(tag in solution_str for tag in invalid_tags):
                if do_print:
                    print("Invalid format")
                actual_score = 0.
                break
    # format    

    if do_print:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as f:
            output = json.dumps({"solution_str": solution_str, "answer": answer, "ground_truth": ground_truth, "actual_score": actual_score, "code_executed": code_executed})
            f.write(output + "\n")

    output_original = output_file.replace(".jsonl", "_original.jsonl")
    os.makedirs(os.path.dirname(output_original), exist_ok=True)
    with open(output_original, "a") as f:
        output = json.dumps({"t": _type, "s": actual_score})
        f.write(output + "\n")
    return actual_score


def compute_score_test(solution_str, ground_truth, method='pass', code_executalbe=False, format_score=0., score=1.):
    _type = get_type(solution_str)
    code_executed = None
    if _type == "CODE" and code_executalbe:
        solution_str, code_executed = rebuild_solution_str(solution_str)
    if method == 'pass':
        answer = extract_solution(solution_str=solution_str, method=method)
        actual_score = 0
        if answer is None:
            actual_score = 0
        else:
            if answer == ground_truth:
                actual_score = score
            else:
                actual_score = format_score
    elif method == 'sc':
        if solution_str == ground_truth:
            actual_score = score
        else:
            actual_score = format_score
    else:
        raise NotImplementedError
    return actual_score
