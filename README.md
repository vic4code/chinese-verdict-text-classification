# Two-Stage Chinese Verdict Text Classification

- [2023/7/31] 目前長文本的文本分類問題對於 PaddleNLP 官方的 Pretrained Model 來說是一大挑戰，有鑑於此，我們先使用 label-studio 標註一些小樣本資料，訓練一個 UIE 模型來擷取跟分類標籤有關的文字段落，再將擷取出來的文字丟給 UTC 模型做訓練。

## Architecture

## Procedure
### Data 準備
- 使用 label studio 分別依照 NER, Text Classification 兩個任務進行資料標注

### Stage 1 - UIE 模型訓練
1. 將 label studio NER 任務標注完成的 `.json` 資料放至 `uie/label_studio_data` 底下，並使用 `labelstudio2doccano.py` 把資料轉成 UIE 模型 input 格式，預設的檔名為 `./doccano_ext.jsonl`，詳細說明可参考 [doccano官方文檔](https://github.com/doccano/doccano)。

```
python uie/labelstudio2doccano.py --doccano_file ./label_studio_data/doccano_ext.jsonl --labelstudio_file ./label_studio_data/data.json
```

2. Run `doccano.py` 將 `.jsonl` data 分成 train, valid, test splits，這裡預設只分成 train, test splits:

```
python uie/doccano.py --splits [0.9, 0, 0.1]
```

3. 開始訓練 UIE 模型
- Note: 跑越久不一定越好，模型可能反而容易預測空值
```
python uie/finetune.py  \
    --device gpu \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 1000 \
    --model_name_or_path uie-base \
    --output_dir ./checkpoint/model_best \
    --train_path data/train.txt \
    --dev_path data/dev.txt  \
    --max_seq_length 512  \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size  16 \
    --num_train_epochs 30  \
    --learning_rate 1e-05 \
    --label_names "start_positions" "end_positions" \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir ./checkpoint/model_best \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_recall \
    --load_best_model_at_end  True \
    --save_total_limit 1 
```
4. 看一下模型在 test data 上的表現:
```
python uie/evaluate.py --model_path checkpoint/model_best/checkpoint-700 --test_path data/test.txt
```
5. 準備一些實際的文本，直接看模型文本擷取的結果：
```
python uie/filter_text.py --model_name_or_path checkpoint/model_best/ --index_of_toy_data 0
```

### Stage 2 - UTC 模型訓練
1. 定義文本分類標籤至 `label.txt`:

```
頭頸部
臉
胸部
腹部
背部
骨盆
上肢
下肢
...
```

2. 將 label studio Text Classification 任務標注完成的 `.json` 資料放至 `utc/label_studio_data` 底下，並使用 `label_studio.py` 把資料切分成 `train.txt`, `valid.txt`, `test.txt` 並配合預先定義好的 `label.txt` 轉成 UTC 模型 input 格式:

```
python utc/label_studio.py  --label_studio_file ./label_studio_exported_data/classification.json --options ./labelstudio_data/label.txt
```

3. 使用 Stage 1 訓練好的 UIE 模型將長文本截短:

```
python utc/uie_preprocessing.py --dataset_path data/data_1000 --max_seq_len 768 --threshold 0.0 --uie_model_name_or_path uie_model/checkpoint-2790/ --out_folder_name processed_data_1000
```

4. 開始訓練 UTC 模型:

```
python utc/run_train.py  \
    --device gpu \
    --logging_steps 5 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 1000 \
    --model_name_or_path utc-base \
    --output_dir ./checkpoint/lr_1e-4 \
    --dataset_path ./data/processed_data_1000/data_1000 \
    --max_seq_length 768  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 25 \
    --learning_rate 1e-4 \
    --do_train \
    --do_eval \
    --do_predict \
    --do_export \
    --export_model_dir ./checkpoint/lr_1e-4 \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model macro_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1 \
    --save_plm 
```

5. 看一下模型在 test data 上的表現:

```
python run_eval.py --test_path ./data/processed_data_1000/data_1000/test.txt --max_seq_len 768 --per_device_eval_batch_size 8  --model_path ./checkpoint/lr_3e-4 --output_dir ./checkpoint/lr_3e-4/test_results
```

6. Inference data

```
python utc_inference.py --test_path ./data/processed_data_8000/processed_data.json --max_seq_len 768 --per_device_eval_batch_size 8  --model_path ./checkpoint/seed_30678 --output_dir ./inference_results/data_8000/test_results
```