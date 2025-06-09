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

import logging
import os
import sys
import random
import json
from dataclasses import dataclass, field

# sys.path.append('/apdcephfs_qy3/share_1565115/jonxie/workspace/open-r1/rewards')

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig
from rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
    seqrec_accuracy_reward,
    seqrec_format_reward,
    seqrec_cosine_sim_reward,
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from seqrec_grpo_trainer_ours import SeqRecGRPOTrainer

from prompt import GRPO_SEQREC_TEMPLATES, grpo_prompt

from utils.model_utils import get_model

logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )


@dataclass
class SeqRecGRPOScriptArguments(GRPOScriptArguments):
    index_file: str = field(
        default='.index.json',
        metadata={"help": "Semantic item ID mapping file"},
    )
    item_file: str = field(
        default='.item.json',
        metadata={"help": "Text item information mapping file"},
    )
    dataset: str = field(
        default='Instruments',
        metadata={"help": "Dataset name"},
    )
    seqrec_data_path: str = field(
        default='',
        metadata={"help": "The data path for the sequential recommendation"}
    )
    max_his_len: int = field(
        default=20,
        metadata={"help": "Maximum number of recent interacted item"}
    )
    num_next_k: int = field(
        default=1,
        metadata={"help": "Max number of predicted item for next K task"}
    )
    his_sep: str = field(
        default=', ',
        metadata={"help": "The separator between items"},
    )
    use_conversation: bool = field(
        default=False,
        metadata={"help": "Whether to use the conversational format for prompting"}
    )
    task_types: list[str] = field(
        default_factory=lambda: ["reason_next_1"],
        metadata={
            "help": "The types of tasks for optimization"
        },
    )


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    seqrec_data_path = os.path.join(script_args.seqrec_data_path, script_args.dataset)
    with open(os.path.join(seqrec_data_path, script_args.dataset + script_args.index_file)) as f:
        index_mapping = json.load(f)
    with open(os.path.join(seqrec_data_path, script_args.dataset + script_args.item_file)) as f:
        item_mapping = json.load(f)
    all_item_ids = list(item_mapping.keys())

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)


    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "seqrec_accuracy": seqrec_accuracy_reward,
        "seqrec_format": seqrec_format_reward,
        "seqrec_cosine_sim": seqrec_cosine_sim_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
    valid_seqrec_task_types = list(set(script_args.task_types) & set(list(GRPO_SEQREC_TEMPLATES.keys())))
    print(f'Valid Tasks: {valid_seqrec_task_types}')
    # Format into conversation
    def convert_one_seqrec_prompt(src_data):
        his_inter_ids = src_data['his_inter_ids']
        tgt_item_id = src_data['tgt_item_id']
        one_data = dict()
        one_data["item_semantic_id"] = "".join(index_mapping[str(tgt_item_id)])
        one_data["item_title"] = item_mapping[str(tgt_item_id)]["title"].strip().strip(".!?,;:`")
        one_data["item_description"] = item_mapping[str(tgt_item_id)]["description"]
        one_data["item_brand"] = item_mapping[str(tgt_item_id)]['brand']
        one_data["item_category"] = item_mapping[str(tgt_item_id)]["categories"]
        history = his_inter_ids
        if script_args.max_his_len > 0:
            history = history[-script_args.max_his_len:]
        inters = ["".join(index_mapping[str(j)]) for j in history]
        inter_titles = ["\"" + item_mapping[str(j)]["title"].strip().strip(".!?,;:`") + "\"" for j in history]
        one_data["semantic_inter"] = script_args.his_sep.join(inters)
        one_data["text_inter"] = script_args.his_sep.join(inter_titles)
        full_inters = []
        for i in range(len(inters)):
            full_inters.append((inter_titles[i], inters[i]))
        one_data['full_inters'] = json.dumps(full_inters)
        # construct candidates for the multiple choice task
        cand_item_ids = random.sample(all_item_ids, 5)
        cand_item_ids.append(str(tgt_item_id))
        cand_item_ids = set(cand_item_ids)
        cand_items = []
        for i, x in enumerate(cand_item_ids, start=1):
            cand_items.append(f'{i}. [{item_mapping[str(x)]["title"].strip().strip(".!?,;:`")}, {"".join(index_mapping[str(x)])}]')
        one_data['cand_item_ids'] = cand_item_ids
        one_data['cand_items'] = '\n'.join(cand_items)
        return one_data
    
    def make_conversation(example):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        sample_info = convert_one_seqrec_prompt(example)
        seqrec_task_type = random.choice(list(GRPO_SEQREC_TEMPLATES.keys()))
        template = random.choice(GRPO_SEQREC_TEMPLATES[seqrec_task_type])
        instruction = template['instruction'].format(**sample_info)
        response = template['response'].format(**sample_info)
        target = sample_info['item_semantic_id']
        prompt.append({"role": "user", "content": instruction})
        return {"prompt": prompt, "response": response, "target": target, "task_type": seqrec_task_type}
   
    def make_prompt(example):
        sample_info = convert_one_seqrec_prompt(example)
        seqrec_task_type = random.choice(valid_seqrec_task_types)
        if 'next_k' in seqrec_task_type:
            next_k = script_args.num_next_k
            sample_info['next_k'] = next_k
        else:
            next_k = 1
        
        template = random.choice(GRPO_SEQREC_TEMPLATES[seqrec_task_type])
        instruction = template['instruction'].format(**sample_info)
        response = template['response'].format(**sample_info)
        target = sample_info['item_semantic_id']
        messages = [
            {"role": "user", "content": instruction},
        ]
        enable_thinking = False
        if 'reason' in seqrec_task_type:
            enable_thinking = True 
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking # Switches between thinking and non-thinking modes. Default is True.
        )
        sample_info.update({"prompt": prompt, "response": response, "target": target, "task_type": seqrec_task_type, "next_k": next_k})
        return sample_info

    if script_args.use_conversation:
        dataset = dataset.map(make_conversation)
    else:
        dataset = dataset.map(make_prompt)

    # breakpoint()
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # logger.info("*** Initializing model kwargs ***")
    # torch_dtype = (
    #     model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    # )
    # model_kwargs = dict(
    #     revision=model_args.model_revision,
    #     trust_remote_code=model_args.trust_remote_code,
    #     attn_implementation=model_args.attn_implementation,
    #     torch_dtype=torch_dtype,
    #     use_cache=False if training_args.gradient_checkpointing else True,
    # )
    # training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = SeqRecGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((SeqRecGRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(script_args.task_types)
    main(script_args, training_args, model_args)
