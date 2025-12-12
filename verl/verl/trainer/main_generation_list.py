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
Generate responses given a dataset of prompts
"""

# 导入全局 Qwen vLLM 补丁
from verl.utils.qwen_vllm_patch import patch_qwen_weights_vllm

import ray
import numpy as np
import hydra
import os
import json

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask
from tqdm import tqdm

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from torchdata.stateful_dataloader import StatefulDataLoader
import verl.utils.torch_functional as verl_F
import copy
import re




@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path)
    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='actor_rollout')

    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    print(f"Start to generate responses for {len(config.data.path)} datasets.")
    for dataset_path in config.data.path:
        dataset_cls = RLHFDataset
        dataset = dataset_cls(
            data_files=[dataset_path],
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )

        dataloader = StatefulDataLoader(
            dataset=dataset,
            batch_size=config.data.batch_size,
            num_workers=8,  
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        total_samples = len(dataset)
        print(f"processing dataset: {dataset_path}")
        print(f"total_samples: {total_samples}")

        config_batch_size = config.data.batch_size

        # dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
        # num_batch = (total_samples // config_batch_size) + 1
        dp_size = wg.world_size
        num_batch = len(dataloader)

        output_lst = [[] for _ in range(config.data.n_samples)]
        print(f'world_size: {wg.world_size}, dp_size: {dp_size}')
        print(f'continue_final_message: {config.data.continue_final_message}')
        print(f'add_generation_prompt: {config.data.add_generation_prompt}')
        for idx, batch in tqdm(enumerate(dataloader)):

            test_batch = DataProto.from_single_dict(batch)

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs = []
            sample_inputs.extend(input_texts)
            if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids","messages"] + ["raw_prompt"],
                )

            test_gen_batch.meta_info = {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            data, pad_size = pad_dataproto_to_divisor(test_gen_batch, wg.world_size)

            # START TO GENERATE FOR n_samples TIMES
            for i in range(config.data.n_samples):

                print(f"Generating for {idx}th batch, {i}th sample")

                output = wg.generate_sequences(data)


                # Remove padding from the output data
                if pad_size > 0:
                    output = unpad_dataproto(output, pad_size)

                output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                                     skip_special_tokens=False)
                

                # remove the padding
                pad_token = tokenizer.pad_token
                output_text_unpad = []
                for text in output_text:
                    output_text_unpad.append(text.replace(pad_token, ''))
                    # print(output_text_unpad[-1])
                    # exit(0)

                output_lst[i].extend(output_text_unpad)

        output_lst = np.array(output_lst, dtype=object)
        output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()
        print(f"len of output_lst: {len(output_lst)}")
        print(f"dataset rows: {len(dataset.dataframe)}")
        # add to the data frame
        dataset.dataframe = dataset.dataframe.add_column('responses', output_lst)

        # write to a new parquet
        output_path = os.path.join(config.data.output_path, f"{config.rollout.temperature}_{config.data.n_samples}_{os.path.basename(dataset_path)}")
        output_dir = os.path.dirname(output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.dataframe.to_parquet(output_path)

    return None


if __name__ == '__main__':
     main()


