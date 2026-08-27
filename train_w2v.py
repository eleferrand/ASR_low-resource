'''
 # @ Author: Éric Le Ferrand
 # @ Create Time: 2025-07-08 19:00:00
 # @ Modified by: Your name
 # @ Modified time: 2025-07-08 10:34:10
 # @ Description: Script to train a wav2vec ASR model. The script takes as argument the path to a folder 
 that contains two csv files. One named train.csv, the other test.csv. The csv should have two column: one with 
 the paths to the wav files, the other for the transcription. The function clean_sent line 38 needs to be modified according to the language. 
 '''
import numpy as np
import re, os
import json
import csv
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from datasets import load_metric
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import soundfile as sf
import argparse
from tqdm import tqdm
from random import shuffle
from cleaning import clean_sent


os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """

    wav, sr = sf.read(fname)
    return wav, sr


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def get_data(data_path, part, lang):
    data = []
    long_wav = np.array([])
    long_transc = ""
    duration = 0
    words = []
    left_out = 0
    with open(f"{data_path}/{part}.csv", mode='r', encoding='utf-8') as cfile:
        reader = csv.reader(cfile, delimiter=",", quotechar="|")
        cpt_w = 0
        for row in tqdm(reader):
            if len(row)>1:
                wav_path = row[0]
                sent = row[1]
                sent = sent.replace(" ", " ")

                if len(sent.split())>1:
                    try:
                        w, sr = sf.read(wav_path)
                        
                        entry = {}
                        duration+=len(w)/sr
                        words = words+sent.split()
                        entry["sentence"] = clean_sent(sent, lang)
                        entry["audio"] = {"sampling_rate" : sr, "array" : w}
                        data.append(entry)
                        
                    except KeyboardInterrupt:
                        exit()
                    # except:
                    #     print("wrong format wav")
                    cpt_w+=1


    print(len(words))
    print(len(set(words)))
    print(f"duration : {duration/60}min")
    print(left_out/60)
    return data


def train(data_path, lang, model_type):
    print("loading data")
    train_data = get_data_reg(data_path,"train")
    test_data = get_data_reg(data_path,"dev")
    print(test_data[0])
    if model_type == 'xlsr-53':
        model_name = "facebook/wav2vec2-large-xlsr-53"
    elif model_type == 'xls-r-300m':
        model_name = 'facebook/wav2vec2-xls-r-300m'
    elif model_type == 'xls-r-1b':
        model_name = 'facebook/wav2vec2-xls-r-1b'

    print("creating vocab")
    vocab_train = set(y for x in train_data for y in x["sentence"])
    
    vocab_test = set(y for x in test_data for y in x["sentence"])
    vocab = vocab_train.union(vocab_test)
    if "\n" in vocab:
        vocab.remove("\n")
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False)
    print(vocab)

    ###Creation of the tokeniser###
    print("setting up tokeniser")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    print("tokeniser saved")

    repo_name = f"{lang}_{model_name.replace("facebook/", "")}" 
    tokenizer.save_pretrained(repo_name)
    ###Extraction of speech features###

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)

        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def prepare_dataset(batch):
        audio = batch["audio"]


        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    #### Setting up data for training###

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = load_metric("wer")
    train_data = list(map(prepare_dataset, train_data))
    test_data = list(map(prepare_dataset, test_data))
 
    print("preparing model")
    #### Training ####
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name, 
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.config.ctc_zero_infinity = True #to prevent the loss of getting lost
    from transformers import TrainingArguments

    epochs = 30
    batch_size = 16


    training_args = TrainingArguments(
    output_dir=repo_name,
    group_by_length=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    num_train_epochs=epochs,
    gradient_checkpointing=True,
    fp16=True,
    learning_rate=3e-4,
    metric_for_best_model="wer",
    save_total_limit=2,
    greater_is_better=False,
    push_to_hub=False,
    )
    from transformers import Trainer
    print("start training")

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=processor.feature_extractor
    )

    trainer.train()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="smp")
    parser.add_argument("--lang", type=str, default="smp")
    parser.add_argument("--model_type", type=str, default="smp")

    #your data path should contain a train and test folders with inside a set of wavs and txts files
    args = parser.parse_args()
    data_path = args.data_path
    lang = args.lang
    model_type = args.model_type

    train(data_path, lang, model_type)

if __name__ == "__main__":
    main()
