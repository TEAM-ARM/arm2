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
Qwen vLLM 权重映射补丁
解决 'visual.patch_embed.proj.weight' KeyError 问题
"""

import sys
import os

# 标记补丁是否已应用
_PATCH_APPLIED = False

def patch_qwen_weights_vllm():
    """
    Patch weight names of qwen multimodal models consistently with transformers==4.52
    See https://github.com/vllm-project/vllm/pull/19054
    """
    global _PATCH_APPLIED
    
    if _PATCH_APPLIED:
        return
    
    try:
        import vllm
        from vllm.model_executor.models.utils import WeightsMapper
        
        # 为 Qwen2.5 VL 模型应用补丁
        if "Qwen2_5_VLForConditionalGeneration" in vllm.model_executor.models.ModelRegistry.models:
            vllm.model_executor.models.ModelRegistry.models["Qwen2_5_VLForConditionalGeneration"].load_model_cls().hf_to_vllm_mapper = WeightsMapper(
                orig_to_new_prefix={
                    # mapping for new names in checkpoint saved after transformers v4.52
                    "model.language_model.": "language_model.model.",
                    "model.visual.": "visual.",
                    # mapping for original checkpoint
                    "lm_head.": "language_model.lm_head.",
                    "model.": "language_model.model.",
                }
            )
            print("✅ Qwen2.5 VL 模型权重映射补丁已应用")
        
        # 为 Qwen2 VL 模型应用补丁
        if "Qwen2VLForConditionalGeneration" in vllm.model_executor.models.ModelRegistry.models:
            vllm.model_executor.models.ModelRegistry.models["Qwen2VLForConditionalGeneration"].load_model_cls().hf_to_vllm_mapper = WeightsMapper(
                orig_to_new_prefix={
                    # mapping for new names in checkpoint saved after transformers v4.52
                    "model.language_model.": "language_model.model.",
                    "model.visual.": "visual.",
                    # mapping for original checkpoint
                    "lm_head.": "language_model.lm_head.",
                    "model.": "language_model.model.",
                }
            )
            print("✅ Qwen2 VL 模型权重映射补丁已应用")
        
        print("### Patch to vllm qwen modelling applied successfully.")
        _PATCH_APPLIED = True
        
    except ImportError as e:
        print(f"❌ 导入 vLLM 模块失败: {e}")
    except Exception as e:
        print(f"❌ 应用补丁时出错: {e}")

# 自动应用补丁
patch_qwen_weights_vllm()
