from transformers import Wav2Vec2Processor, AutoModelForCTC
from transformers import Wav2Vec2ProcessorWithLM
import soundfile as sf
import torch
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm
from datasets import load_metric
import re
import csv
import os
import shutil
from unidecode import unidecode
from cleaning import clean_sent

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"


wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

def prepare_dataset(batch):
    audio = batch["audio"]

    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        # print(batch["sentence"])
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

def eval_model(data_path, lang, model_type, order):
    repo_name = f"{lang}_{model_name.replace("facebook/", "")}"
    checkpoint = max([x for x in os.listdir(f'{repo_name}') if "checkpoint" in x], key=lambda y: int(y.split('-')[1]))

    path_checkpoint = f'{repo_name}/{checkpoint}/'

    if "tokenizer_config.json" not in [x for x in os.listdir(path_checkpoint)]:
        shutil.copy(f'{scratch}{model_type}/tokenizer_config.json', path_checkpoint+"tokenizer_config.json")
        shutil.copy(f'{scratch}{model_type}/vocab.json', path_checkpoint+"vocab.json")


    model = AutoModelForCTC.from_pretrained(path_checkpoint).to(device)#change
    processor = Wav2Vec2Processor.from_pretrained(path_checkpoint)

    vocab = processor.tokenizer.get_vocab()
    vocab[' '] = vocab['|']
    del vocab[' ']
    sorted_dict = {k.lower(): v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
    print(sorted_dict)

    b_size = 10
    signals = []
    sentences = []
    test_data = []
    refs = []
    letters = set()
    with open(f'{data_path}/test.csv', "r") as in_file:
        reader = csv.reader(in_file, delimiter=",", quotechar="|")
        for row in reader:
            if len(row)>1:
                sent = clean_sent(row[1])
                for l in sent:
                    if l not in vocab and l!=" ":
                        print(l)

                w, sr = sf.read(row[0])
                test_data.append({"audio" : {"sampling_rate" : 16000, "array" : w}, "sentence" : sent})
                signals.append(w)
                sentences.append(sent)
                refs.append(row[0])
                for l in sent:
                    letters.add(l)
    print(letters)

    test_data = list(map(prepare_dataset, test_data))

    preds = []

    for ind in tqdm(range(len(test_data))):
        input_dict = processor(test_data[ind]["input_values"], return_tensors="pt",sampling_rate=16000, padding=True)
        logits = model(input_dict.input_values.to(device)).logits
        decoded = processor.decode(torch.argmax(logits, dim=-1)[0]).replace('[UNK]', '')
        preds.append(decoded)


    for i in range(len(preds)):
        print(preds[i])
        print(sentences[i])

    print(wer_metric.compute(predictions=preds, references=sentences))
    print(cer_metric.compute(predictions=preds, references=sentences))
    if not os.path.isdir("outputs/"):
        os.mkdir("outputs/")
    with open(f"outputs/{lang}_{model_type}_greedy.csv", mode="w", encoding="utf-8") as tfile:
        writer = csv.writer(tfile,delimiter=",", quotechar="|")
        for i in range(len(preds)):
            pred = preds[i]
            gold = sentences[i]
            ref = refs[i]
            writer.writerow([ref,gold,pred])
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="smp")
    parser.add_argument("--lang", type=str, default="smp")
    parser.add_argument("--model_type", type=str, default="smp")

    args = parser.parse_args()
    data_path = args.data_path
    lang = args.lang
    model_type = args.model_type
    eval_model(data_path, lang, model_type)

if __name__ == "__main__":
    main()