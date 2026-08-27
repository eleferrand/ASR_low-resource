from transformers import Wav2Vec2Processor, HubertForCTC
import soundfile as sf
import torch
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm
from evaluate import load
import re
import csv
import os
import shutil
from unidecode import unidecode
if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

lang = 'West_Xoon'
model_type = 'hubert-large-ll60k'
# model_type = 'hubert-xlarge-ll60k'

def clean_sent(sent):
    sent = sent.lower()
    repl_dict = {("à", "á", "â", "ǎ") : "a", ("ì","ǐ", "í", "î") : "i", ("ń", "ǹ","ň"): "n", ("ú","ǔ", "ù") : "u", 
                 ("é","è", "ê", "ě"," ́e"): "e", ("m̀", "ḿ","m̌"): "m", ("r̀") : "r", ("ó", "ô"): "o"}
    for tones in repl_dict:
        for char in tones:
            if char in sent:
                sent = sent.replace(char, repl_dict[tones])
    sent = re.sub(r"[\.\-\[\],\n;:?]"," ", sent)
    sent = sent.replace("|", "")
    return sent

wer_metric = load("wer")
cer_metric = load("cer")

def prepare_dataset(batch):
    audio = batch["audio"]

    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        # print(batch["sentence"])
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch


scratch = f'/scratch/leferran/clicks_models/{lang}_facebook/'
checkpoint = max([x for x in os.listdir(f'{scratch}{model_type}') if "checkpoint" in x], key=lambda y: int(y.split('-')[1]))

path_checkpoint = f'{scratch}{model_type}/{checkpoint}/'

if "tokenizer_config.json" not in [x for x in os.listdir(path_checkpoint)]:
    shutil.copy(f'{scratch}{model_type}/tokenizer_config.json', path_checkpoint+"tokenizer_config.json")
    shutil.copy(f'{scratch}{model_type}/vocab.json', path_checkpoint+"vocab.json")


model = HubertForCTC.from_pretrained(path_checkpoint).to(device)#change
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
with open(f'data/{lang}/test.csv', "r") as in_file:
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

with open(f"outputs/{lang}_{model_type}_greedy.csv", mode="w", encoding="utf-8") as tfile:
    writer = csv.writer(tfile,delimiter=",", quotechar="|")
    for i in range(len(preds)):
        pred = preds[i]
        gold = sentences[i]
        ref = refs[i]
        writer.writerow([ref,gold,pred])
