# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import json
import argparse
import yaml
from functools import partial

import paddle
from utils import (
    convert_example,
    create_data_loader,
    get_relation_type_dict,
    reader,
    unify_prompt_name,
)

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import MapDataset, load_dataset
from paddlenlp.metrics import SpanEvaluator
from paddlenlp.transformers import UIE, UIEM, AutoTokenizer
from paddlenlp.utils.log import logger


@paddle.no_grad()
def compute_metrics(model, metric, data_loader, multilingual=False):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        multilingual(bool): Whether is the multilingual model.
    """
    model.eval()
    metric.reset()
    for batch in data_loader:
        if multilingual:
            start_prob, end_prob = model(batch["input_ids"], batch["position_ids"])
        else:
            start_prob, end_prob = model(
                batch["input_ids"], batch["token_type_ids"], batch["position_ids"], batch["attention_mask"]
            )

        start_ids = paddle.cast(batch["start_positions"], "float32")
        end_ids = paddle.cast(batch["end_positions"], "float32")
        num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)
        metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    model.train()
    return precision, recall, f1


def evaluate(args):
    paddle.set_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.multilingual:
        model = UIEM.from_pretrained(args.model_path)
    else:
        model = UIE.from_pretrained(args.model_path)

    test_ds = load_dataset(reader, data_path=args.test_path, max_seq_len=args.max_seq_len, lazy=False)
    class_dict = {}
    relation_data = []
    if args.debug:
        for data in test_ds:
            class_name = unify_prompt_name(data["prompt"])
            # Only positive examples are evaluated in debug mode
            if len(data["result_list"]) != 0:
                p = "的" if args.schema_lang == "ch" else " of "
                if p not in data["prompt"]:
                    class_dict.setdefault(class_name, []).append(data)
                else:
                    relation_data.append((data["prompt"], data))

        relation_type_dict = get_relation_type_dict(relation_data, schema_lang=args.schema_lang)
    else:
        class_dict["all_classes"] = test_ds

    trans_fn = partial(
        convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len, multilingual=args.multilingual
    )
    
    
    for key in class_dict.keys():
        if args.debug:
            test_ds = MapDataset(class_dict[key])
        else:
            test_ds = class_dict[key]
        test_ds = test_ds.map(trans_fn)

        data_collator = DataCollatorWithPadding(tokenizer)

        test_data_loader = create_data_loader(test_ds, mode="test", batch_size=args.batch_size, trans_fn=data_collator)

        metric = SpanEvaluator()
        precision, recall, f1 = compute_metrics(model, metric, test_data_loader, args.multilingual)
        logger.info("-----------------------------")
        logger.info("Class Name: %s" % key)
        logger.info("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" % (precision, recall, f1))

        metrics = {
            "precision":precision,
            "recall":recall,
            "f1":f1
        }

        if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as fp:
            json.dump(metrics, fp)

    if args.debug and len(relation_type_dict.keys()) != 0:
        for key in relation_type_dict.keys():
            test_ds = MapDataset(relation_type_dict[key])
            test_ds = test_ds.map(trans_fn)
            test_data_loader = create_data_loader(
                test_ds, mode="test", batch_size=args.batch_size, trans_fn=data_collator
            )

            metric = SpanEvaluator()
            precision, recall, f1 = compute_metrics(model, metric, test_data_loader)
            logger.info("-----------------------------")
            if args.schema_lang == "ch":
                logger.info("Class Name: X的%s" % key)
            else:
                logger.info("Class Name: %s of X" % key)
            logger.info("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" % (precision, recall, f1))

            metrics = {
                "precision":precision,
                "recall":recall,
                "f1":f1
            }

            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as fp:
                json.dump(metrics, fp)


def main():

    with open("configs/eval.yaml", "r") as f:
        logger.info(f"Loading the config file in {os.path.abspath(f.name)}...")
        configs = yaml.load(f, Loader=yaml.CSafeLoader)

    parser = argparse.ArgumentParser(description='Argument Parser from Dictionary')
    for key, value in configs["eval_args"].items():
        arg_name = '--' + key
        parser.add_argument(arg_name, default=value, type=type(value))

    eval_args = parser.parse_args()
    evaluate(eval_args)

if __name__ == "__main__":
    
    main()