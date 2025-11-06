# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from verl.utils.megatron_utils import unwrap_model

from .util import postprocess_packed_seqs, preprocess_packed_seqs, recover_left_padding, remove_left_padding
import torch

iteration = 0


def gptmodel_forward(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    logits_processor=None,
    logits_processor_args: dict = None,
    **kwargs,
):
    """Default forward pass for GPT models with optional sequence packing."""
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()

        global iteration

        if iteration == 0:
            save_path = f'/data/logs/megatron-dump-gpu/gpt_model_forward_0_iter{iteration}_rank{torch.distributed.get_rank()}.pt'
            # torch.save({'input_ids_rmpad': input_ids_rmpad.cpu(), 'position_ids': position_ids.cpu(), 'packed_seq_params': packed_seq_params.cpu()}, save_path)
            cache_value = torch.load(save_path, map_location=input_ids_rmpad.device, weights_only=False)
            if torch.distributed.get_rank() == 0:
                breakpoint()
            input_ids_rmpad = cache_value['input_ids_rmpad']
            position_ids = cache_value['position_ids']
            packed_seq_params2 = cache_value['packed_seq_params']

        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )

        if iteration == 0:
            save_path = f'/data/logs/megatron-dump-gpu/gpt_model_forward_1_iter{iteration}_rank{torch.distributed.get_rank()}.pt'
            # torch.save({'output_orig': output_orig.cpu()}, save_path)
            cache_value = torch.load(save_path, map_location=input_ids_rmpad.device, weights_only=False)
            if torch.distributed.get_rank() == 0:
                breakpoint()
            output_orig = cache_value['output_orig']

        if post_process and logits_processor is not None:
            args = {k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0] for k, v in logits_processor_args.items()}
            output_dict = logits_processor(output_orig, **args)
            output = {k: postprocess_packed_seqs(v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process) for k, v in output_dict.items()}
        else:
            output = postprocess_packed_seqs(output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process)

        if iteration == 0:
            save_path = f'/data/logs/megatron-dump-gpu/gpt_model_forward_2_iter{iteration}_rank{torch.distributed.get_rank()}.pt'
            # torch.save({key: val.cpu() for key, val in output.items()}, save_path)
            cache_value = torch.load(save_path, map_location=input_ids_rmpad.device, weights_only=False)
            if torch.distributed.get_rank() == 0:
                breakpoint()
            for key, val in cache_value.items():
                output[key] = val

        iteration += 1

    else:
        assert logits_processor is None, "logits_processor is not supported for non-packed sequence"
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process)
        output = model(input_ids=new_input_ids, attention_mask=new_attention_mask, position_ids=new_position_ids)
        output = recover_left_padding(output, new_attention_mask, attention_mask, sequence_length, post_process=post_process)
    if value_model and post_process:
        output = output[..., 0]
    return output


def gptmodel_forward_qwen2_5_vl(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    multi_modal_inputs=None,
    logits_processor=None,
    logits_processor_args: dict = None,
    **kwargs,
):
    from megatron.core import parallel_state as mpu

    assert mpu.get_context_parallel_world_size() == 1, "qwen2_5_vl's context parallel is not accurate yet"
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    pixel_values = (multi_modal_inputs["pixel_values"].to(input_ids.device) if "pixel_values" in multi_modal_inputs else None)
    image_grid_thw = (multi_modal_inputs["image_grid_thw"].to(input_ids.device) if "image_grid_thw" in multi_modal_inputs else None)
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        if post_process and logits_processor is not None:
            args = {k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0] for k, v in logits_processor_args.items()}
            output_dict = logits_processor(output_orig, **args)
            output = {k: postprocess_packed_seqs(v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process) for k, v in output_dict.items()}
        else:
            output = postprocess_packed_seqs(output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process)
    else:
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process)
        output = model(
            input_ids=new_input_ids,
            position_ids=new_position_ids,
            attention_mask=new_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        output = recover_left_padding(output, new_attention_mask, attention_mask, sequence_length, post_process=post_process)
    if value_model and post_process:
        output = output[..., 0]
    return output
