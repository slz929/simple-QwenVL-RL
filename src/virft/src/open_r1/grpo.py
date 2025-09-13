# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json
from reasoning_dataset import RelationReasoningDataset
from reward_function import accuracy_reward_iou, accuracy_reward_confidence, format_reward

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    train_data_path: str = field(default="", metadata={"help": "Path to training data."})
    train_image_folder_path: str = field(default="", metadata={"help": "Path to training images."})


###  reward registry three parts
reward_funcs_registry = {
    "accuracy_iou": accuracy_reward_iou,
    "accuracy_confidence": accuracy_reward_confidence,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
# Format into conversation
def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

def make_conversation_image(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["problem"]},
                ],
            },
        ],
    }

def main(script_args, training_args, model_args):
    # Get reward functions
    script_args.reward_funcs = ['accuracy_iou','accuracy_confidence','format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset from huggingface
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    # Load the dataset from local disk
    # from datasets import DatasetDict
    # dataset = DatasetDict.load_from_disk(script_args.dataset_name)

    # if "image" in dataset[script_args.dataset_train_split].features:
    #     print("has image in dataset")
    #     dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    #     # dataset = dataset.remove_columns(["original_question", "original_answer"])

    # else:
    #     print("no image in dataset")
    #     dataset = dataset.map(make_conversation)
    #     dataset = dataset.remove_columns("messages")


    train_data_path= script_args.train_data_path
    train_image_folder_path= script_args.train_image_folder_path

    train_data_paths = train_data_path.split(',')
    train_image_folder_paths = train_image_folder_path.split(',')
    
    # prompt= (
    #         f"Detect all objects belonging to the category '{category}' in the image, and provide the bounding boxes (between 0 and 1000, integer) and confidence (between 0 and 1, with two decimal places).\n"
    #         f"If no object belonging to the category '{category}' in the image, return 'No Objects'.\n"
    #         "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    #         "The output answer format should be as follows:\n"
    #         "<think> ... </think> <answer>[{'Position': [x1, y1, x2, y2], 'Confidence': number}, ...]</answer>\n"
    #         "Please strictly follow the format."
    #     )
    
    
    category= '生物'
    prompt= (
        f"检测图像中所有的“{category}”，并提供边界框（整数）和置信度（0 到 1的小数）。\n"
        f"如果图像中没有“{category}”，则返回“无”。\n"
        "在 <think> </think> 标签中输出思考过程，在 <answer> </answer> 标签中输出最终答案。"
        "输出答案格式应如下：\n"
        "<think> ... </think> <answer>[{'Position': [x1, y1, x2, y2], 'Confidence': number}, ...]</answer>\n"
        "请严格遵循格式。"
    )

    prompt_suffix= ''
    train_dataset = RelationReasoningDataset(train_data_paths, train_image_folder_paths, prompt = prompt, prompt_suffix = prompt_suffix)
    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset= train_dataset,
        eval_dataset= None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Initialize the GRPO trainer
    # trainer = trainer_cls(
    #     model=model_args.model_name_or_path,
    #     reward_funcs=reward_funcs,
    #     args=training_args,
    #     train_dataset=dataset[script_args.dataset_train_split],
    #     eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
    #     peft_config=get_peft_config(model_args),
    #     attn_implementation=model_args.attn_implementation,
    #     max_pixels=script_args.max_pixels,
    #     min_pixels=script_args.min_pixels,
    # )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    # trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
