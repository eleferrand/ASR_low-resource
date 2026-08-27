from transformers import Wav2Vec2Processor, HubertForCTC
from transformers import Wav2Vec2ProcessorWithLM
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
from cleaning import clean_sent

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"


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
def eval_model(data_path, lang, model_type, order):


    repo_name = f"{lang}_{model_name.replace("facebook/", "")}"
    lm_model= f"{lang}_{order}gram.arpa"
    checkpoint = max([x for x in os.listdir(f'{repo_name}') if "checkpoint" in x], key=lambda y: int(y.split('-')[1]))

    path_checkpoint = f'{repo_name}/{checkpoint}/'
    print(lm_model)

    if "tokenizer_config.json" not in [x for x in os.listdir(path_checkpoint)]:
        shutil.copy(f'{scratch}{model_type}/tokenizer_config.json', path_checkpoint+"tokenizer_config.json")
        shutil.copy(f'{scratch}{model_type}/vocab.json', path_checkpoint+"vocab.json")


    model = HubertForCTC.from_pretrained(path_checkpoint).to(device)#change
    processor = Wav2Vec2Processor.from_pretrained(path_checkpoint)

    vocab = processor.tokenizer.get_vocab()

    vocab_dict = processor.tokenizer.get_vocab()

    model_vocab_size = model.config.vocab_size

    # EXACT alignment with model logits
    vocab_list = [""] * model_vocab_size

    for token, idx in vocab_dict.items():
        if idx < model_vocab_size:
            vocab_list[idx] = token

    print(len(vocab_list))
    print(vocab_list)
    for i, tok in enumerate(vocab_list):
        if tok == "":
            print("Missing token id:", i)
    blank_token = processor.tokenizer.pad_token

    for i in range(len(vocab_list)):
        if vocab_list[i] == "":
            vocab_list[i] = blank_token


    vocab[' '] = vocab['|']
    del vocab[' ']
    sorted_dict = {k.lower(): v for k, v in sorted(vocab.items(), key=lambda item: item[1])}


    decoder = build_ctcdecoder(
        # list(sorted_dict.keys()),
        vocab_list,
        lm_model,
        alpha = 0.5,
        beta = 1.5
    )

    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder
    )

    print(model.config.vocab_size)
    print(len(processor.tokenizer.get_vocab()))

    b_size = 10
    signals = []
    sentences = []
    test_data = []
    refs = []
    letters = set()
    with open(f'{data_path}test.csv', "r") as in_file:
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
        input_dict = processor_with_lm(test_data[ind]["input_values"], return_tensors="pt",sampling_rate=16000, padding=True)
        logits = model(input_dict.input_values.to(device)).logits
        preds.append(processor_with_lm.decode(logits[0].cpu().detach().numpy()).text)


    for i in range(len(preds)):
        print(preds[i])
        print(sentences[i])

    print(wer_metric.compute(predictions=preds, references=sentences))
    print(cer_metric.compute(predictions=preds, references=sentences))
    if not os.path.isdir("outputs/"):
        os.mkdir("outputs/")
    with open(f"outputs/{lang}_{model_type}_{order}gram.csv", mode="w", encoding="utf-8") as tfile:
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
    order = args.order
    model_type = args.model_type
    eval_model(data_path, lang, model_type, order)

if __name__ == "__main__":
    main()