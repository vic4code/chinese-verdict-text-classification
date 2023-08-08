import argparse
import os
os.sys.path.append("..") # 待解

from tqdm import tqdm
import numpy as np
import json
import re
import yaml
from paddlenlp import Taskflow
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.log import logger
from paddlenlp.transformers import AutoTokenizer
from utils import filter_text, rule_based_processer, write_json
from utc.utils import get_template_tokens_len


def inference(args):
    """
    Shorten long texts with the finetuned uie model for utc model.
    """
    # setting
    processer = rule_based_processer()
    uie = Taskflow("information_extraction", task_path=args.uie_model_name_or_path, schema=args.schema, precision="fp16")
    tokenizer = AutoTokenizer.from_pretrained(args.utc_model_name_or_path)
    special_word_len = get_template_tokens_len(tokenizer, args.label_file)
    max_content_len = args.max_seq_len - special_word_len

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # processing to utc inputs
    total = 0
    for data_name in (args.train_file, args.dev_file, args.test_file):
        out_text = []
        logger.info(f"Start preprocessing {data_name}...")
        number_of_examples = 0

        with open(os.path.join(args.dataset_path, data_name), "r", encoding="utf8") as fp:
            number_of_examples = len(fp.readlines())
        total += number_of_examples

        with open(os.path.join(args.dataset_path, data_name), "r", encoding="utf8") as fp:
            for example in tqdm(fp, total=number_of_examples):
                example = json.loads(example.strip())
                uie_output = uie(example["text_a"])

                # if IS_RULE_BASED_POSTPROCESSING:
                uie_output = processer.postprocessing(
                    raw_text=example["text_a"], uie_output=uie_output, labels=example["labels"], schema=args.schema
                )

                new_text = filter_text(
                    raw_text=example["text_a"],
                    uie_output=uie_output,
                    max_len_of_new_text=max_content_len,
                    threshold=args.threshold,
                    dynamic_adjust_length=args.dynamic_adjust_length,
                )

                if len(new_text) == 0 and len(example["labels"]) == 0:
                    new_text = example["text_a"][:last_index]

                example["text_a"] = new_text
                out_text.append(example)

        write_json(out_text, out_path=os.path.join(args.output_path, data_name))
        logger.info(f"Finish {data_name} processing. Total samples: {len(out_text)}.")

    logger.info(f"Finish all preprocessing.")


def main():

    with open("configs/infer.yaml", "r") as f:
        logger.info(f"Loading the config file in {os.path.abspath(f.name)}...")
        configs = yaml.load(f, Loader=yaml.CSafeLoader)

    parser = argparse.ArgumentParser(description='Argument Parser from Dictionary')
    for key, value in configs["infer_args"].items():
        arg_name = '--' + key
        parser.add_argument(arg_name, default=value, type=type(value))

    infer_args = parser.parse_args()
    inference(infer_args)


if __name__ == "__main__":
    main()