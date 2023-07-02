from datasets import load_dataset
from typing import List
import os
import logging
from typing import Dict, Optional, Sequence, List, Union
import transformers
from functools import partial

logger = logging.getLogger(__name__)


def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []
    for (root, dir, file_name) in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list


def load_dataset_from_path(data_path: Optional[str] = None,
                           cache_dir: Optional[str] = "cache_data",
                           data_file_number: Optional[int] = 2,
                           use_streaming: bool = False):
    all_file_list = get_all_datapath(data_path)[:data_file_number]
    data_files = {'train': all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info("load files %d number", len(all_file_list))

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
        streaming=use_streaming
    )['train']
    return raw_datasets


def make_train_dataset(tokenizer: transformers.PreTrainedTokenizer,
                       data_path: str,
                       data_file_number: int,
                       data_proc_num: int,
                       use_streaming: bool):
    logging.warning("Loading data...")

    dataset = load_dataset_from_path(
        data_path=data_path,
        data_file_number=data_file_number,
        use_streaming=use_streaming
    )
    logging.warning("Formatting inputs...")

    def generate_sources_targets(examples: Dict, tokenizer: transformers.PreTrainedTokenizer):
        ins_data = examples['content']

        input_output = tokenizer(ins_data, return_tensors="np")
        examples['input_ids'] = input_output['input_ids']
        return examples

    generate_sources_targets_p = partial(
        generate_sources_targets, tokenizer=tokenizer)

    if use_streaming:
        dataset = dataset.map(
            function=generate_sources_targets_p,
            batched=True
        ).shuffle(42, buffer_size=50000)
    else:
        dataset = dataset.map(
            function=generate_sources_targets_p,
            batched=True,
            desc="Running tokenizer on train dataset",
            num_proc=data_proc_num
        ).shuffle()

    return dataset
