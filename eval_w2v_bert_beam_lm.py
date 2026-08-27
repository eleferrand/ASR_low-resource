from transformers import Wav2Vec2BertProcessor
from transformers import Wav2Vec2ProcessorWithLM
from transformers import Wav2Vec2BertForCTC
from transformers import SeamlessM4TFeatureExtractor
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
order = 5
model_type = 'w2v-bert-2.0'



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



lm_model= f"{lang}_{order}gram.arpa"
scratch = f'/scratch/leferran/clicks_models/{lang}_facebook/'
checkpoint = max([x for x in os.listdir(f'{scratch}{model_type}') if "checkpoint" in x], key=lambda y: int(y.split('-')[1]))

path_checkpoint = f'{scratch}{model_type}/{checkpoint}/'
print(lm_model)

if "tokenizer_config.json" not in [x for x in os.listdir(path_checkpoint)]:
    shutil.copy(f'{scratch}{model_type}/tokenizer_config.json', path_checkpoint+"tokenizer_config.json")
    shutil.copy(f'{scratch}{model_type}/vocab.json', path_checkpoint+"vocab.json")


model = Wav2Vec2BertForCTC.from_pretrained(path_checkpoint).to(device)#change
processor = Wav2Vec2BertProcessor.from_pretrained(path_checkpoint)

vocab = processor.tokenizer.get_vocab()
vocab[' '] = vocab['|']
del vocab[' ']
sorted_dict = {k.lower(): v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
print(sorted_dict)

decoder = build_ctcdecoder(
    list(sorted_dict.keys()),
    lm_model,
    alpha = 0.5,
    beta = 1.5
)

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)


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

def prepare_dataset(batch):
    audio = batch["audio"]
    # print(processor(audio["array"], sampling_rate=audio["sampling_rate"]))
    # exit()
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_length"] = len(batch["input_features"])
    
    # with processor.as_target_processor():
        # print(batch["sentence"])
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

test_data = list(map(prepare_dataset, test_data))

preds = []

for ind in tqdm(range(len(test_data))):

    # input_dict = processor_with_lm(test_data[ind]["input_features"], return_tensors="pt",sampling_rate=16000, padding=True)
    input_dict = {"input_features": torch.tensor(test_data[ind]["input_features"]).unsqueeze(0).to(device)}
    with torch.no_grad():
        logits = model(**input_dict).logits
        # logits = model(input_dict.input_values.to(device)).logits
        logits_np = logits.cpu().numpy()
        pred_str = processor_with_lm.batch_decode(logits_np)
        decoded = processor_with_lm.decode(logits[0].cpu().detach().numpy()).text
    print(decoded)
    preds.append(decoded)


for i in range(len(preds)):
    print(preds[i])
    print(sentences[i])

print(wer_metric.compute(predictions=preds, references=sentences))
print(cer_metric.compute(predictions=preds, references=sentences))

with open(f"outputs/{lang}_{model_type}_{order}gram.csv", mode="w", encoding="utf-8") as tfile:
    writer = csv.writer(tfile,delimiter=",", quotechar="|")
    for i in range(len(preds)):
        pred = preds[i]
        gold = sentences[i]
        ref = refs[i]
        writer.writerow([ref,gold,pred])
