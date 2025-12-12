# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code


code_executalbe = True

def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth, code_executalbe, output_file=f'./samples/{data_source}.jsonl')
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo":
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source.startswith("aime") and data_source.lower() not in ["aime"]:
        # Handle other "aime" variants (e.g., "aime24", "aime25") - use math_dapo for backward compatibility
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        from . import prime_code

        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth, code_executalbe, output_file=f'./samples/{data_source}.jsonl')
    elif data_source in ["FanqingM/MMK12"]:
        from . import mmk12
        res = mmk12.compute_score(solution_str, ground_truth, code_executalbe, output_file=f'./samples/{data_source}.jsonl')

    elif data_source in ["MME_RealWorld"]:
        from . import mme_world
        res = mme_world.compute_score(solution_str, ground_truth, code_executalbe, output_file=f'./samples/{data_source}.jsonl')
    
    elif data_source in ["AIME"] or data_source.lower() == "aime":
        # Handle both "AIME" (uppercase) and "aime" (lowercase) data sources
        # Use aime.compute_score which has format_score=0.0 by default and checks format
        from . import aime
        res = aime.compute_score(solution_str, ground_truth, code_executalbe, output_file=f'./samples/{data_source}.jsonl')
    
    elif data_source in ["CSQA"]:
        from . import csqa
        res = csqa.compute_score(solution_str, ground_truth, code_executalbe, output_file=f'./samples/{data_source}.jsonl')
        
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
