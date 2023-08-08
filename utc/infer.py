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

import json
import os
os.sys.path.append("..") # 待解

import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from paddlenlp.utils.log import logger
from metric import MetricReport
import yaml
import paddle
from paddlenlp import Taskflow
from utils import UTCLoss, read_inference_dataset, get_template_tokens_len

from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import (
    PromptModelForSequenceClassification,
    PromptTrainer,
    PromptTuningArguments,
    UTCTemplate,
)
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import UTC, AutoTokenizer, PretrainedTokenizer
from uie.utils import filter_text, rule_based_processer, write_json
from typing import Any, Dict, List, Optional


@dataclass
class DataArguments:
    data_file_to_inference: str = field(default="../data_to_inference/data.json", metadata={"help": "Test dataset file name."})
    threshold: float = field(default=0.5, metadata={"help": "The threshold to produce predictions."})
    single_label: str = field(default=False, metadata={"help": "Predict exactly one label per sample."})
    label_file: str = field(default="../label_studio_data/text_classification_labels.txt", metadata={"help": "Predict exactly one label per sample."})
    do_uie_preprocess: str = field(default=True, metadata={"help": "Data preprocess with uie before feeding to utc model"})
    last_index: int = field(default=10, metadata={"help": "The last index to slice the text example when the output from ruled based processor is empty."})


@dataclass
class ModelArguments:
    uie_model_name_or_path: str = field(default="uie-base", metadata={"help": "The schema for the uie preprocessor"})
    uie_output_path: str = field(default="uie-base", metadata={"help": "The output path for uie preprocessed data"})
    schema: str = field(default=None, metadata={"help": "The schema for the uie preprocessor"})
    dynamic_adjust_length: str = field(default=True, metadata={"help": "The dynamic_adjust_length for the uie preprocessor"})
    utc_model_name_or_path: str = field(default="utc-base", metadata={"help": "The utc model name"})
    utc_model_path: str = field(default="checkpoints/utc", metadata={"help": "The path of the finetuned utc model"})
  
class InferenceUTCTemplate(UTCTemplate):

    template_special_tokens = ["text", "hard", "sep", "cls", "options"]

    def __init__(self, tokenizer: PretrainedTokenizer, max_length: int, prompt: str = None):
        prompt = (
            (
                "{'options': 'choices', 'add_omask': True, 'position': 0, 'token_type': 1}"
                "{'sep': None, 'token_type': 0, 'position': 0}{'text': 'text_a'}{'sep': None, 'token_type': 1}{'text': 'text_b'}"
            )
            if prompt is None
            else prompt
        )
        super(UTCTemplate, self).__init__(prompt, tokenizer, max_length)
        self.max_position_id = self.tokenizer.model_max_length - 1
        self.max_length = max_length
        if not self._has_options():
            raise ValueError(
                "Expected `options` and `add_omask` are in defined prompt, but got {}".format(self.prompt)
            )

    def _has_options(self):
        for part in self.prompt:
            if "options" in part and "add_omask" in part:
                return True
        return False

    def build_inputs_with_prompt(
        self, example: Dict[str, Any], prompt: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        inputs = super(UTCTemplate, self).build_inputs_with_prompt(example, prompt)
        for index, part in enumerate(inputs):
            if "cls" in part:
                inputs[index] = self.tokenizer.cls_token
        
        return inputs

    def encode(self, example: Dict[str, Any], use_mask: bool = False):
        input_dict = super(UTCTemplate, self).encode(example)

        # Set OMASK and MASK positions and labels for options.
        omask_token_id = self.tokenizer.convert_tokens_to_ids("[O-MASK]")
        input_dict["omask_positions"] = (
            np.where(np.array(input_dict["input_ids"]) == omask_token_id)[0].squeeze().tolist()
        )

        sep_positions = (
            np.where(np.array(input_dict["input_ids"]) == self.tokenizer.sep_token_id)[0].squeeze().tolist()
        )
        input_dict["cls_positions"] = sep_positions[0]

        # Limit the maximum position ids.
        position_ids = np.array(input_dict["position_ids"])
        position_ids[position_ids > self.max_position_id] = self.max_position_id
        input_dict["position_ids"] = position_ids.tolist()

        return input_dict

    def create_prompt_parameters(self):
        return None

    def process_batch(self, input_dict):
        return input_dict


def uie_preprocess(
    model_args,
    data_args,
    infer_args
):
    """
    Shorten long texts with the finetuned uie model for utc model.
    """

    # setting
    processer = rule_based_processer()
    uie = Taskflow("information_extraction", task_path=model_args.uie_model_name_or_path, schema=model_args.schema)
    tokenizer = AutoTokenizer.from_pretrained(model_args.utc_model_name_or_path)
    special_word_len = get_template_tokens_len(tokenizer, data_args.label_file)
    max_content_len = infer_args.max_seq_length - special_word_len

    if not os.path.exists(model_args.uie_output_path):
        os.makedirs(model_args.uie_output_path)

    # processing
    out_text = []
    logger.info(f"Start to preprocess inference_data with uie...")
    number_of_examples = 0

    with open(data_args.data_file_to_inference, "r", encoding="utf8") as fp:
        data = json.load(fp) 
        number_of_examples = len(data)
    
    for example in tqdm(data[:10], total=number_of_examples):

        uie_output = uie(example["jfull_compress"])
        uie_output = processer.postprocessing(
            raw_text=example["jfull_compress"], uie_output=uie_output, labels=None, schema=model_args.schema
        )

        new_text = filter_text(
            raw_text=example["jfull_compress"],
            uie_output=uie_output,
            max_len_of_new_text=max_content_len,
            threshold=data_args.threshold,
            dynamic_adjust_length=model_args.dynamic_adjust_length,
        )

        if len(new_text) == 0:
            new_text = example["jfull_compress"][:data_args.last_index]

        example["jfull_compress"] = new_text
        out_text.append(example)

    write_json(out_text, out_path=os.path.join(model_args.uie_output_path, 'uie_processed_data.json'))
    logger.info(f"Finish inference_data preprocessing. Total samples: {len(out_text)}.")


def inference(
    model_args,
    data_args,
    infer_args
):

    paddle.set_device(infer_args.device)

    # Load the pretrained language model.
    tokenizer = AutoTokenizer.from_pretrained(model_args.utc_model_name_or_path)
    model = UTC.from_pretrained(model_args.utc_model_name_or_path)

    # Define template for preprocess and verbalizer for postprocess.
    template = InferenceUTCTemplate(tokenizer, infer_args.max_seq_length)

    # Load and preprocess dataset.
    if data_args.data_file_to_inference is not None:
        test_ds = load_dataset(read_inference_dataset, data_path=data_args.data_file_to_inference, lazy=False, options=data_args.label_file)

    # Initialize the prompt model.
    prompt_model = PromptModelForSequenceClassification(
        model, template, None, freeze_plm=infer_args.freeze_plm, freeze_dropout=infer_args.freeze_dropout
    )
    if model_args.utc_model_path is not None:
        model_state = paddle.load(os.path.join(model_args.utc_model_path, "model_state.pdparams"))
        prompt_model.set_state_dict(model_state)

    # Define the metric function.
    trainer = PromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=infer_args,
        criterion=UTCLoss(),
        train_dataset=None,
        eval_dataset=None,
        callbacks=None,
    )

    # Inference
    if data_args.data_file_to_inference is not None:
        with open(data_args.label_file, "r", encoding="utf-8") as fp:
            choices = [x.strip() for x in fp]

        test_data = open(data_args.data_file_to_inference, "r", encoding="utf-8") 
        test_ret = trainer.predict(test_ds)

        if not os.path.exists(infer_args.output_dir):
            os.makedirs(infer_args.output_dir)

        with open(os.path.join(infer_args.output_dir, "inference_results.json"), "w", encoding="utf-8") as fp:    
            preds = paddle.nn.functional.sigmoid(paddle.to_tensor(test_ret.predictions))

            logger.info(f"Start to inference data with utc...")
            for index, example in enumerate(test_data):
                result = json.loads(example)
                try:
                    del(result["jfull_compress"])
                    del(result["text_a"])
                
                except:
                    pass

                pred_ids = paddle.where(preds[index] > data_args.threshold)[0].tolist()    

                result["pred_labels"] = {}
                for choice in choices:
                    result["pred_labels"][choice] = 0

                if pred_ids:
                    for pred_id in pred_ids:
                        result["pred_labels"][choices[pred_id[0]]] = 1
                        
                fp.write(json.dumps(result, ensure_ascii=False) + "\n")

def main():

    # Load config file
    with open("configs/infer.yaml", "r") as f:
        logger.info(f"Loading the config file in {os.path.abspath(f.name)}...")
        configs = yaml.load(f, Loader=yaml.CSafeLoader)
    
    # Parse the arguments.
    model_args, data_args, infer_args = ModelArguments(**configs["model_args"]), DataArguments(**configs["data_args"]), PromptTuningArguments(**configs["infer_args"]),
    infer_args.print_config(model_args, "Model")
    infer_args.print_config(data_args, "Data")

    # UIE preprocess
    if data_args.do_uie_preprocess:
        uie_preprocess(
            model_args,
            data_args,
            infer_args
        )

    # Inference
    inference(
        model_args,
        data_args,
        infer_args
    )

if __name__ == "__main__":
    main()
