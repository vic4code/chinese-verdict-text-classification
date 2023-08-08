import argparse
import json
import os
import time
import yaml
import numpy as np
from decimal import Decimal
from utils import convert_cls_examples, convert_ext_examples, set_seed

from paddlenlp.trainer.argparser import strtobool
from paddlenlp.utils.log import logger

def append_attrs(data, item, label_id, relation_id):

    mapp = {}

    for anno in data["annotations"][0]["result"]:
        if anno["type"] == "labels":
            label_id += 1
            item["entities"].append(
                {
                    "id": label_id,
                    "label": anno["value"]["labels"][0],
                    "start_offset": anno["value"]["start"],
                    "end_offset": anno["value"]["end"],
                }
            )
            mapp[anno["id"]] = label_id

    for anno in data["annotations"][0]["result"]:
        if anno["type"] == "relation":
            relation_id += 1
            item["relations"].append(
                {
                    "id": relation_id,
                    "from_id": mapp[anno["from_id"]],
                    "to_id": mapp[anno["to_id"]],
                    "type": anno["labels"][0],
                }
            )

    return item, label_id, relation_id


def convert(dataset, task_type):
    results = []
    outer_id = 0
    if task_type == "ext":
        label_id = 0
        relation_id = 0
        for data in dataset:
            outer_id += 1
            item = {"id": outer_id, "text": data["data"]["text"], "entities": [], "relations": []}
            item, label_id, relation_id = append_attrs(data, item, label_id, relation_id)
            results.append(item)
    # for the classification task
    else:
        for data in dataset:
            outer_id += 1
            results.append(
                {
                    "id": outer_id,
                    "text": data["data"]["text"],
                    "label": data["annotations"][0]["result"][0]["value"]["choices"],
                }
            )
    return results


def do_convert(args):

    if not os.path.exists(args.label_studio_file):
        raise ValueError("Please input the correct path of label studio file.")

    with open(args.label_studio_file, "r", encoding="utf-8") as infile:
        for content in infile:
            dataset = json.loads(content)
        results = convert(dataset, args.task_type)

    return results
    

def do_split(args, converted_results):

    set_seed(args.seed)

    tic_time = time.time()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if len(args.splits) != 0 and len(args.splits) != 3:
        raise ValueError("Only []/ len(splits)==3 accepted for splits.")

    def _check_sum(splits):
        return Decimal(str(splits[0])) + Decimal(str(splits[1])) + Decimal(str(splits[2])) == Decimal("1")

    if len(args.splits) == 3 and not _check_sum(args.splits):
        raise ValueError("Please set correct splits, sum of elements in splits should be equal to 1.")

    raw_examples = converted_results

    def _create_ext_examples(
        examples,
        negative_ratio,
        prompt_prefix="情感倾向",
        options=["正向", "负向"],
        separator="##",
        shuffle=False,
        is_train=True,
        schema_lang="ch",
    ):
        entities, relations, aspects = convert_ext_examples(
            examples, negative_ratio, prompt_prefix, options, separator, is_train, schema_lang
        )
        examples = entities + relations + aspects
        if shuffle:
            indexes = np.random.permutation(len(examples))
            examples = [examples[i] for i in indexes]
        return examples

    def _create_cls_examples(examples, prompt_prefix, options, shuffle=False):
        examples = convert_cls_examples(examples, prompt_prefix, options)
        if shuffle:
            indexes = np.random.permutation(len(examples))
            examples = [examples[i] for i in indexes]
        return examples

    def _save_examples(save_dir, file_name, examples):
        count = 0
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        logger.info("Save %d examples to %s." % (count, save_path))

    if len(args.splits) == 0:
        if args.task_type == "ext":
            examples = _create_ext_examples(
                raw_examples,
                args.negative_ratio,
                args.prompt_prefix,
                args.options,
                args.separator,
                args.is_shuffle,
                schema_lang=args.schema_lang,
            )
        else:
            examples = _create_cls_examples(raw_examples, args.prompt_prefix, args.options, args.is_shuffle)
        _save_examples(args.save_dir, "train.txt", examples)
    else:
        if args.is_shuffle:
            indexes = np.random.permutation(len(raw_examples))
            index_list = indexes.tolist()
            raw_examples = [raw_examples[i] for i in indexes]
        else:
            index_list = list(range(len(raw_examples)))

        i1, i2, _ = args.splits
        p1 = int(len(raw_examples) * i1)
        p2 = int(len(raw_examples) * (i1 + i2))

        train_ids = index_list[:p1]
        dev_ids = index_list[p1:p2]
        test_ids = index_list[p2:]

        with open(os.path.join(args.save_dir, "sample_index.json"), "w") as fp:
            maps = {"train_ids": train_ids, "dev_ids": dev_ids, "test_ids": test_ids}
            fp.write(json.dumps(maps))

        if args.task_type == "ext":
            train_examples = _create_ext_examples(
                raw_examples[:p1],
                args.negative_ratio,
                args.prompt_prefix,
                args.options,
                args.separator,
                args.is_shuffle,
                schema_lang=args.schema_lang,
            )
            dev_examples = _create_ext_examples(
                raw_examples[p1:p2],
                -1,
                args.prompt_prefix,
                args.options,
                args.separator,
                is_train=False,
                schema_lang=args.schema_lang,
            )
            test_examples = _create_ext_examples(
                raw_examples[p2:],
                -1,
                args.prompt_prefix,
                args.options,
                args.separator,
                is_train=False,
                schema_lang=args.schema_lang,
            )
        else:
            train_examples = _create_cls_examples(raw_examples[:p1], args.prompt_prefix, args.options)
            dev_examples = _create_cls_examples(raw_examples[p1:p2], args.prompt_prefix, args.options)
            test_examples = _create_cls_examples(raw_examples[p2:], args.prompt_prefix, args.options)

        _save_examples(args.save_dir, "train.txt", train_examples)
        _save_examples(args.save_dir, "dev.txt", dev_examples)
        _save_examples(args.save_dir, "test.txt", test_examples)

    logger.info("Finished! It takes %.2f seconds" % (time.time() - tic_time))


def data_prepare(label_studio_args):

    # Convert label studio data to the format of model inputs
    converted_results = do_convert(label_studio_args)
    
    # Split data as train, dev, test
    do_split(label_studio_args, converted_results)


if __name__ == "__main__":

    with open("configs/label_studio.yaml", "r") as f:
        logger.info(f"Loading the config file in {os.path.abspath(f.name)}...")
        configs = yaml.load(f, Loader=yaml.CSafeLoader)

    parser = argparse.ArgumentParser(description='Argument Parser from Dictionary')
    for key, value in configs["label_studio_args"].items():
        arg_name = '--' + key
        parser.add_argument(arg_name, default=value, type=type(value))

    label_studio_args = parser.parse_args()
    data_prepare(label_studio_args)
    