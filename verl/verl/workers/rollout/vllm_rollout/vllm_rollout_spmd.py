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
"""
The vllm_rollout that can be applied in different backend
"""

# 导入全局 Qwen vLLM 补丁
from verl.utils.qwen_vllm_patch import patch_qwen_weights_vllm

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union

import json5
import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
import verl.utils.torch_functional as verl_F
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.dataset import RLHFDataset
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length, get_eos_mask
from verl.workers.rollout.base import BaseRollout

import copy
import re

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)



OBS_START = '<observation>'
OBS_END = '</observation>'
CODE_START = '<code>'
CODE_END = '</code>'

def extract_program(result: str, last_only=True):
    """
    extract the program after "<code>", and before "</code>"
    """
    program = ''
    start = False
    for line in result.split('\n'):
        if line.strip() == '<code>':
            if last_only:
                program = ''  # only extract the last program
            else:
                program += '\n# ========\n'
            start = True
        elif line.strip() == '</code>':
            start = False
        elif start:
            program += line + '\n'
    if start:
        # the code is incomplete
        program = ''
    return program


def _detect_tool(text: str, image: str) -> Tuple[bool, str, str, str]:
    program = extract_program(text)
    if image:
        program = program.replace("sample_img.jpg", image)
    if program:
        program = json.dumps({'code': program}, ensure_ascii=False)
    return (program != ''),  program, text



def rebuild_inputs(datasets_config: RLHFDataset, row_dict):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """

        messages = copy.deepcopy(row_dict["messages"])

        for message in messages:
            content = message["content"]
            content_list = []
            for segment in re.split("(<image>|<video>)", content):
                if segment == "<image>":
                    content_list.append({"type": "image"})
                elif segment == "<video>":
                    content_list.append({"type": "video"})
                else:
                    content_list.append({"type": "text", "text": segment})

            message["content"] = content_list

        model_inputs = {}

        if datasets_config.processor is not None:
            from verl.utils.dataset.vision_utils import process_image
            assert messages[-1]["role"] == 'assistant'
            raw_prompt = datasets_config.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

            # delete the last token <|im_end|> in each message
            marker = "<|im_end|>\n"
            assert raw_prompt.count(marker) == 3
            assert raw_prompt.endswith(marker)
            parts = raw_prompt.rsplit(marker, 1)
            raw_prompt = parts[0] + parts[1]

            multi_modal_data = {}

            images = None
            if row_dict['multi_modal_data']['image']:
                images = [process_image(image) for image in row_dict['multi_modal_data']['image']]
                multi_modal_data["image"] = images

            videos = None


            model_inputs = datasets_config.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt",max_pixels=256 * 28 * 28)

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = datasets_config.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = datasets_config.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=datasets_config.max_prompt_length,
            pad_token_id=datasets_config.tokenizer.pad_token_id,
            left_pad=True,
            truncation=datasets_config.truncation,
        )

        if datasets_config.processor is not None and datasets_config.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    datasets_config.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            from verl.utils.model import compute_position_id_with_mask
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = datasets_config.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > datasets_config.max_prompt_length:
            if datasets_config.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-datasets_config.max_prompt_length :]
            elif datasets_config.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: datasets_config.max_prompt_length]
            elif datasets_config.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {datasets_config.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if datasets_config.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 32768)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, f"model context length {model_hf_config.max_position_embeddings} should be greater than total sequence length {config.prompt_length + config.response_length}"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                f"Enable chunked prefill, max_num_batched_tokens {max_num_batched_tokens} is smaller than max_model_len {max_model_len}, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = {"image": 10}
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=True,
            seed=config.get("seed", 0),
            mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 512 * 28 * 28,
            "fps": [1],
        },
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id


    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)


    def _tokenize_with_multi_modal_data(self, processor,tokenizer, prompt, multi_modal_data):
        if processor is not None:
            return processor(text=[prompt], images=multi_modal_data["image"], videos=multi_modal_data["video"], return_tensors="pt")
        else:
            return tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    
    def _tokenize_and_find_mask_token_indices(self, sample_info):
        response=sample_info['messages'][-1]['content']
        mask_str_ranges=sample_info['mask_info']

        encoding=self.datasets_config.tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
        
        response_token_ids=encoding['input_ids']

        offset_mapping_tensor=torch.tensor(encoding['offset_mapping'], dtype=torch.long)
        token_starts = offset_mapping_tensor[:,0]
        token_ends = offset_mapping_tensor[:,1]

        mask_tensor=torch.ones(len(response_token_ids))
        for mask_str_range in mask_str_ranges:
            start_index, end_index=mask_str_range[0], mask_str_range[1]
            mask = (token_starts < end_index) & (token_ends > start_index) & (token_starts >= start_index)
            mask_tensor[mask]=0 

        return response_token_ids, mask_tensor
    
    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                if multi_modal_data == {}:
                    vllm_inputs.append({"prompt_token_ids": raw_prompt_ids})
                else:
                    vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0.7,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": -1,
                "top_p": 1.0,
                "temperature": 0.7,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            for i, vllm_input in enumerate(vllm_inputs):
                if vllm_input['multi_modal_data']['image'] == []:
                    _ = vllm_input.pop('multi_modal_data')
            
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )


            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            response_length_ind_list = []


            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)
                    response_length_ind_list.append(len(response[-1]))
            
            response_list_tensor = torch.tensor(response_length_ind_list, dtype=torch.long)

                    
            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(non_tensor_batch["multi_modal_inputs"], self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "response_length_ind": response_list_tensor,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    def _get_unfinished_indices(self, samples_info):
        indices=[]
        for index, info in enumerate(samples_info):
            if len(info['image_files']) >= 10:
                continue
            if not info['stop']:
                indices.append(info['index'])

        return indices

    def set_attr(self, name, value):
        setattr(self, name, value)

class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)


if __name__ == "__main__":
    response = """To determine the EBIT Margin for 2023, we need to follow these steps:\n\n1. Identify the EBIT (Earnings Before Interest and Taxes) value for 2023 from the provided data.\n2. Identify the PAT (Profit After Tax) value for 2023 from the provided data.\n3. Calculate the EBIT Margin using the formula: EBIT Margin = (EBIT / PAT) * 100.\n\nLet\'s start by extracting the relevant data from the image.\n\n<code>\nimport pandas as pd\n\n# Load the data from the Excel file\ndf = pd.read_excel(\'sample_img.xlsx\')\n\n# Extract the EBIT and PAT values for 2023\nebit_2023 = df.loc[df[\'Year\'] == 2023, \'EBIT ($\' + \'000)\'].values[0]\npat_2023 = df.loc[df[\'Year\'] == 2023, \'PAT ($\' + \'000)\'].values[0]\n\nprint(f"EBIT for 2023: {ebit_2023}")\nprint(f"PAT for 2023: {pat_2023}")\n</code>\n"""
    image = "sample_img.jpg"
    print(_detect_tool(response, image))