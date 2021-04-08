<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#Argument Impact Classification">Argument Impact Classification</a>
    </li>
    <li>
      <a href="#prerequisites">prerequisites</a>
    </li>
    <li><a href="#Installation">Installation</a></li>
    <li><a href="#Usage">Usage</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
# Argument Impact Classification
This project is for MSBD6000H project 1 Argument Impact Classification.

It contains two models: fast-text and Roberta.

## Prerequisites
python 3+
### For FastText Model 
* nltk
* pandas
* fasttext
* re
* sklearn
### For Bert Model
* transformers
* datasets

## Installation
### For FastText Model
Use the package manager [pip](https://pypi.org/project/fasttext/) to install fasttext.

```bash
pip install fasttext
```
### For Bert Model
Run in **Colab**(which can use the free gpu resource)
```bash
!pip install git+https://github.com/huggingface/transformers
!pip install datasets
```

## Usage
### For FastText Model
* change the dataset file path and run the script in python3
### For Bert Model
```
!python run_bert.py --model_name_or_path roberta-large --use_fast_tokenizer False --train_file 'train.csv' --validation_file 'valid.csv' --do_train --do_eval --do_predict --test_file 'test.csv' --max_seq_length 100 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --output_dir 'roberta'
```
parameters:
- model_name_or_path: model(BERT/Roberta/Albert...)
- train_file: training file
- validation_file: validation file
- test_file: test file
- output_dir: output direction
- max_seq_length: max sequence length
- per_device_train_batch_size: batch size
- learning_rate: learning rate
- num_train_epochs: epochs

