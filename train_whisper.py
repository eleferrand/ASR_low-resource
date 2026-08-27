from transformers import Seq2SeqTrainer
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import soundfile as sf
import os, re
import numpy as np
import argparse
from tqdm import tqdm

print("grabbing models")

metric = evaluate.load("wer")

def clean_sent(sent):
    sent = sent.lower()
    sent = re.sub(r"[\.\-\[\],\n;:?]"," ", sent)
    return sent


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


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch





def train(data_path, lang, model_type):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/{model_type}")
    tokenizer = WhisperTokenizer.from_pretrained(f"openai/{model_type}", task="transcribe")
    processor = WhisperProcessor.from_pretrained(f"openai/{model_type}", task="transcribe")


    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
        
    print("loading data")
    train_data = list(map(prepare_dataset, get_data(data_path, "train"))) 
    dev_data = list(map(prepare_dataset, get_data(data_path, "dev")))

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    print("initializing training")
    epochs = 30
    batch_size = 16

    repo_name = f"{lang}_{model_type}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=repo_name,  # change to a repo name of your choice
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size = 8,
        group_by_length=True,
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        learning_rate=3e-4,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        metric_for_best_model="wer",
        load_best_model_at_end=True,
        save_total_limit=2,
        greater_is_better=False,
        save_strategy="epoch",
        push_to_hub=False,)

    model = WhisperForConditionalGeneration.from_pretrained(f"openai/{model_type}")
    print("training")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_data,
        eval_dataset=dev_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor)

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