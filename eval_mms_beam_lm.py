from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2CTCTokenizer
import soundfile as sf
import torch
from tqdm import tqdm
from evaluate import load
import re
import csv
from pyctcdecode import build_ctcdecoder
import json
import tempfile
import os
import shutil
from unidecode import unidecode
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
from safetensors.torch import save_file as safe_save_file


if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
lang = 'West_Xoon'
iso = 'uni'
model_type = 'mms-1b-all'
# model_type = 'mms-1b-all'
order = 5
frozen = False

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

wer_metric = load("wer")
cer_metric = load("cer")
lm_model= f"{lang}_{order}gram.arpa"
def prepare_dataset(batch):
    audio = batch["audio"]

    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        # print(batch["sentence"])
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch
if frozen:
    scratch = f'/scratch/leferran/clicks_models/{lang}_facebook/{model_type}_frozen/'
else:
    scratch = f'/scratch/leferran/clicks_models/{lang}_facebook/{model_type}/'
checkpoint = max([x for x in os.listdir(f'{scratch}') if "checkpoint" in x], key=lambda y: int(y.split('-')[1]))

path_checkpoint = f'{scratch}{checkpoint}/'

if "tokenizer_config.json" not in [x for x in os.listdir(path_checkpoint)]:
    shutil.copy(f'{scratch}tokenizer_config.json', path_checkpoint+"tokenizer_config.json")
    shutil.copy(f'{scratch}vocab.json', path_checkpoint+"vocab.json")



if model_type=='mms-1b':
    model = Wav2Vec2ForCTC.from_pretrained(path_checkpoint).to(device)
    processor = Wav2Vec2Processor.from_pretrained(path_checkpoint)
else:
    model = Wav2Vec2ForCTC.from_pretrained(path_checkpoint, target_lang="uni").to(device)#change
    processor = Wav2Vec2Processor.from_pretrained(path_checkpoint)
    # processor.tokenizer.set_target_lang('uni')

orig_tokenizer = processor.tokenizer

vocab = orig_tokenizer.get_vocab()

# unwrap MMS vocab
if "uni" in vocab:
    vocab = vocab["uni"]


tmp_dir = tempfile.mkdtemp()
vocab_path = os.path.join(tmp_dir, "vocab.json")

with open(vocab_path, "w") as f:
    json.dump(vocab, f)

flat_tokenizer = Wav2Vec2CTCTokenizer(
    vocab_path,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)
vocab[' '] = vocab['|']
del vocab[' ']

# sorted_dict = {k.lower(): v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
vocab = [token for token, idx in sorted(vocab.items(), key=lambda x: x[1])]

# Now add the special tokens with their correct indices
vocab += ["<s>", "</s>"]

decoder = build_ctcdecoder(
    list(vocab),
    lm_model,
    alpha = 0.5,
    beta = 1.5
)

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=flat_tokenizer,
    decoder=decoder,
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
    input_dict = processor_with_lm(test_data[ind]["input_values"], return_tensors="pt",sampling_rate=16000, padding=True)
    logits = model(input_dict.input_values.to(device)).logits
    decoded = processor_with_lm.decode(logits[0].cpu().detach().numpy()).text
    decoded = decoded.replace('</s>', '')
    decoded = re.sub(r' +', ' ', decoded)

    preds.append(decoded)


    # preds.append(processor.decode(pred_ids))


final_pred = []
final_sent = []
for i in range(len(preds)):
    if len(preds[i].strip())>1 and len(sentences[i].strip())>1:
        print(preds[i])
        print(sentences[i])
        final_pred.append(preds[i])
        final_sent.append(sentences[i])


print(wer_metric.compute(predictions=final_pred, references=final_sent))
print(cer_metric.compute(predictions=final_pred, references=final_sent))
with open(f"outputs/{lang}_{model_type}_{order}gram.csv", mode="w", encoding="utf-8") as tfile:
    writer = csv.writer(tfile,delimiter=",", quotechar="|")
    for i in range(len(preds)):
        pred = preds[i]
        gold = sentences[i]
        ref = refs[i]
        writer.writerow([ref,gold,pred])