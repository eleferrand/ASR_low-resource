'''
 # @ Author: Éric Le Ferrand
 # @ Create Time: 2025-07-08 19:00:00
 # @ Modified by: Your name
 # @ Modified time: 2025-07-08 10:34:10
 # @ Description: Script to train a MMS ASR model. The script takes as argument the path to a folder that contains two csv files. 
 One named train.csv, the other test.csv. The csv should have two column: one with the paths to the wav files, the other for the transcription. 
 The function clean_sent line 39 needs to be modified according to the language. 
 '''

import numpy as np
import re, os
import json
import csv
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
import soundfile as sf
import argparse
from tqdm import tqdm
from datasets import Dataset, Audio
from IPython.display import display, HTML
import random
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import Trainer
from transformers import TrainingArguments
from evaluate import load
from safetensors.torch import save_file as safe_save_file
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

wer_metric = load("wer")

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
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """

    wav, sr = sf.read(fname)
    return wav, sr

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}




def clean_sent(sent):
    sent = sent.lower()
    repl_dict = {("à", "á", "â", "ǎ") : "a", ("ì","ǐ", "í", "î") : "i", ("ń", "ǹ","ň"): "n", ("ú","ǔ", "ù") : "u", 
                 ("é","è", "ê", "ě"," ́e"): "e", ("m̀", "ḿ","m̌"): "m", ("r̀") : "r", ("ó", "ô"): "o"}
    for tones in repl_dict:
        for char in tones:
            if char in sent:
                sent = sent.replace(char, repl_dict[tones])
    sent = re.sub(r"[\.\-\[\],\n;:?]"," ", sent)
    sent = sent.replace("!", "ǃ")
    sent = sent.replace('"', '')
    return sent





def get_data_reg(data_path,part):
    data = []
    long_wav = np.array([])
    long_transc = ""
    duration = 0
    words = []
    letters = []
    with open(f"{data_path}{part}.csv", mode='r', encoding='utf-8') as cfile:
        reader = csv.reader(cfile, delimiter=",", quotechar="|")
        cpt_w = 0
        for row in tqdm(reader):
            if len(row)>1:
                wav_path = row[0]
                sent = row[1]
                sent = sent.replace(" ", " ")
                if len(sent.split())>3:
                    w, sr = sf.read(wav_path)
                    if len(w)/sr<30:
                        long_transc += sent +" "
                        long_wav = np.concatenate((w, long_wav), axis=0)
                        entry = {}
                        duration+=len(long_wav)/sr
                        words = words+long_transc.split()
                        long_transc = clean_sent(long_transc)
                        letters = letters+[l for l in long_transc]
                        entry["sentence"] = long_transc
                        entry["audio"] = {"sampling_rate" : sr, "array" : long_wav}
                        if len(long_wav)/sr>20:
                            print(long_transc)
                            input()
                        data.append(entry)
                        long_wav = np.array([])
                        long_transc = ""

                        cpt_w+=1
                else:
                    w, sr = sf.read(wav_path)
                    if (len(long_wav)+len(w))/sr>20:
                        continue
                    long_wav = w
                    long_transc = sent
    print(len(words))
    print(len(set(words)))
    print(f"duration : {duration/60}min")
    print(set(letters))

    # ---- convert dynamically to Dataset ----
    ds = Dataset.from_list(data)
    ds = ds.cast_column("audio", Audio())
    return ds

def train(data_path, lang, model_type, frozen):
    print("loading data")

    train_data = get_data_reg(data_path,"train")
    test_data = get_data_reg(data_path,"test")

    if model_type == 'mms-1b':
        model_name = "facebook/mms-1b"
    elif model_type == 'mms-1b-all':
        model_name = "facebook/mms-1b-all"

    
    vocab_train = train_data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train_data.column_names)
    vocab_test = test_data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=test_data.column_names)

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "] 
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    target_lang = 'uni'
    new_vocab_dict = {target_lang: vocab_dict}
    with open('vocab.json', 'w') as vocab_file:
        json.dump(new_vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", target_lang='uni')

    scratch = '/scratch/leferran/clicks_models/'

    if frozen:
        repo_name = f"{scratch}{lang}_{model_name}_frozen" 
    else:
        repo_name = f"{scratch}{lang}_{model_name}"
    tokenizer.save_pretrained(repo_name)
    print(repo_name)

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched"
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

        
    train_data = train_data.map(prepare_dataset, remove_columns=train_data.column_names)
    test_data = test_data.map(prepare_dataset, remove_columns=test_data.column_names)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2ForCTC.from_pretrained(
    model_name,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True,
    )   
    model.config.ctc_zero_infinity = True

    if model_type=='mms-1b-all':
        model.init_adapter_layers()
        # model.freeze_base_model()
        adapter_weights = model._get_adapters()
        for param in adapter_weights.values():
            param.requires_grad = True

    batch_size = 16
    epochs = 30
    training_args = TrainingArguments(
    output_dir=repo_name,
    group_by_length=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=epochs,
    gradient_checkpointing=True,
    fp16=True,
    learning_rate=3e-4,
    metric_for_best_model="wer",
    save_total_limit=2,
    greater_is_better=False,
    push_to_hub=False,
    )

    trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=processor.feature_extractor,
    )

    trainer.train()
    if model_type=='mms-1b-all':
        adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format('uni')
        adapter_file = os.path.join(training_args.output_dir, adapter_file)

        safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="smp")
    parser.add_argument("--lang", type=str, default="smp")
    parser.add_argument("--model_type", type=str, default="smp")
    parser.add_argument("--frozen", action="store_true")

    #your data path should contain a train and test folders with inside a set of wavs and txts files
    args = parser.parse_args()
    data_path = args.data_path
    lang = args.lang
    model_type = args.model_type
    frozen = args.frozen

    train(data_path, lang, model_type, frozen)

if __name__ == "__main__":
    main()