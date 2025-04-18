import json
import logging
import os
import random
import datetime

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from dataset import SeqRecDataset, ItemFeatDataset, ItemSearchDataset, FusionSeqRecDataset, SeqRecTestDataset, PreferenceObtainDataset

def load_train_datasets(args):

    tasks = args.tasks.split(",")

    train_prompt_sample_num = [int(_) for _ in args.train_prompt_sample_num.split(",")]
    assert len(tasks) == len(train_prompt_sample_num), "prompt sample number does not match task number"
    train_data_sample_num = [int(_) for _ in args.train_data_sample_num.split(",")]
    assert len(tasks) == len(train_data_sample_num), "data sample number does not match task number"

    train_datasets = []
    for task, prompt_sample_num,data_sample_num in zip(tasks,train_prompt_sample_num,train_data_sample_num):
        if task.lower() == "seqrec":
            dataset = SeqRecDataset(args, mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == "item2index" or task.lower() == "index2item":
            dataset = ItemFeatDataset(args, task=task.lower(), prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == "fusionseqrec":
            dataset = FusionSeqRecDataset(args, mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == "itemsearch":
            dataset = ItemSearchDataset(args, mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == "preferenceobtain":
            dataset = PreferenceObtainDataset(args, prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        else:
            raise NotImplementedError
        train_datasets.append(dataset)

    train_data = ConcatDataset(train_datasets)

    valid_data = SeqRecDataset(args,"valid",args.valid_prompt_sample_num)

    return train_data, valid_data

def load_test_dataset(args):

    if args.test_task.lower() == "seqrec":
        test_data = SeqRecDataset(args, mode="test", sample_num=args.sample_num)
        # test_data = SeqRecTestDataset(args, sample_num=args.sample_num)
    elif args.test_task.lower() == "itemsearch":
        test_data = ItemSearchDataset(args, mode="test", sample_num=args.sample_num)
    elif args.test_task.lower() == "fusionseqrec":
        test_data = FusionSeqRecDataset(args, mode="test", sample_num=args.sample_num)
    else:
        raise NotImplementedError

    return test_data

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data