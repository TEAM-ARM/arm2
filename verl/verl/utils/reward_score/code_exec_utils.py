import math
import numpy as np
import collections
import functools
import itertools
import fractions
import sympy
import signal
import re
import psutil
import time
import sys


def handler(signum, frame):
    raise TimeoutError("Timeout")

def safe_str_convert(obj):
    """Safely convert an object to string, handling cases of very large integers"""
    try:
        return str(obj)
    except ValueError as e:
        if "Exceeds the limit" in str(e):
            # For dictionary type, try to convert each value
            if isinstance(obj, dict):
                safe_dict = {}
                for key, value in obj.items():
                    try:
                        safe_dict[key] = str(value)
                    except ValueError:
                        safe_dict[key] = f"<large_value_{type(value).__name__}>"
                return str(safe_dict)
            else:
                return f"<large_{type(obj).__name__}>"
        else:
            return repr(obj)

def execute_code(code):
    # Increase the limit for integer string conversion
    sys.set_int_max_str_digits(50000)
    
    # Check memory usage, wait if it exceeds 90%
    while psutil.virtual_memory().percent > 90:
        print(f"The memory usage is too high: {psutil.virtual_memory().percent}%，waiting for memory release...")
        time.sleep(1)
    
    custom_global_scope = {
        'math': math,
        'np': np,
        'Counter': collections.Counter,
        'comb': math.comb,
        'factorial': math.factorial,
        'gcd': math.gcd,
        'lcm': math.lcm,
        'sympy': sympy,
        'reduce': functools.reduce,
        'permutations': itertools.permutations,
        'combinations': itertools.combinations,
        'Fraction': fractions.Fraction,
    }
    local_scope = {}
    result = None
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(30)
        exec(code, custom_global_scope, local_scope)

        result = local_scope['result']
    except Exception as e:
        if "is not defined" in str(e):
            pass
        # print('execution error: ', e)
        pass
    finally:
        signal.alarm(0)
    return result

def extract_code_obs_ans(solution_str):
    try:
        code = re.search(r"<CODE>(.*)</CODE>", solution_str, re.DOTALL)
        obs = re.search(r"<OBSERVATION>(.*)</OBSERVATION>", solution_str, re.DOTALL)
        ans = re.search(r"<ANSWER>(.*)</ANSWER>", solution_str, re.DOTALL)
        if code is None and obs is None and ans is None:
            return None, None, None
        else:
            return code.group(1).strip(), obs.group(1).strip(), ans.group(1).strip()
    except Exception as e:
        # print('extract_code_obs_ans error: ', e)
        return None, None, None
    
def rebuild_solution_str(solution_str):
    code, obs, ans = extract_code_obs_ans(solution_str)
    if code is None:
        return solution_str, False
    if 'print' in code:
        code = re.sub(r"print\((.*)\)", r"result = \1", code)
    elif '>>>' in code:
        code = re.sub(r'>>>', 'result = ', code)
    result = execute_code(code)

    if result is not None and type(result) == dict and 'answer' in result:
        for key in result:
            val = result[key]
            if isinstance(val, float):
                result[key] = round(val, 2)
        ans = result['answer']
        # Use safe string conversion function
        result_str = safe_str_convert(result)
        
        return f"<CODE>\n{code}\n</CODE>\n<OBSERVATION>\nresult = {result_str}\n</OBSERVATION>\n<ANSWER>\n{ans}\n</ANSWER>", True
    else:
        return solution_str, False

if __name__ == "__main__":
    solution_str = "<CODE>\ndef find_angle_BDA():\n    # Given values\n    PB = 12\n    AB = 15\n    angle_ABD = 24\n\n    # In a rhombus, all sides are equal, so DA = AB = 15\n    # The diagonals of a rhombus bisect the angles\n    # So, angle ABD = 24, then angle BDA = 180 - 2*24 = 120\n\n    angle_BDA = 180 - 2 * angle_ABD\n\n    answer = angle_BDA\n\n    return {\n        'PB': PB,\n        'AB': AB,\n        'angle_ABD': angle_ABD,\n        'angle_BDA': angle_BDA,\n        'answer': answer\n    }\n\n>>> find_angle_BDA()</CODE>\n<OBSERVATION>\nresult = {\n    'PB': 12,\n    'AB': 15,\n    'angle_ABD': 24,\n    'angle_BDA': 120,\n    'answer': 120\n}</OBSERVATION>\n<ANSWER>\n120\n</ANSWER>"