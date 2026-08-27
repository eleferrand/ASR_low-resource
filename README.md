### Training ASR models for low-resource languages
This repository provides implementations to train and evaluate transformers-based ASR models. 4 architectures are included: wav2vec2 for CTC, HuBERT for CTC, MMS for CTC, wav2vec-BERT for CTC and Whisper.
The scripts are built with the same basis with small differences. Additionally several decoding methods are included


Dependencies
You first need to install kenlm: https://github.com/kpu/kenlm
To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Training

You need to create a folder where you put your data. The data folder should contain 3 CSVs train.csv, dev.csv and test.csv. Each csv has two columns the first one is the absolute path to audio files and the second the corresponding transcriptions. (delimiters = "," quotechars = '"')
cleaning.py contains a single function used in every script to strip the transcription form punctation and possible diacritics, you should adapt the function to your language. 

You can train your model with the following lines:
```
python train_w2v.py --data_path /path/to/your/data --lang target_language --model_type wav2vec2-large-xlsr-53 ##you can also choose wav2vec2-xls-r-1b, wav2vec2-xls-r-300m and mms-1b
```
```
python train_hubert.py --data_path /path/to/your/data --lang target_language --model_type hubert-large-ll60k ##you can also choose hubert-xlarge-ll60k
```

```
python train_whisper.py --data_path /path/to/your/data --lang target_language --model_type whisper-medium ## you can also choose whisper-small whisper-tiny etc.
```

```
python train_w2v-bert.py --data_path /path/to/your/data --lang target_language 
```

```
python train_MMS.py --data_path /path/to/your/data --lang target_language 
```
## Evaluation
It's been shown that an small ngram model drastically helps WER. You can train a simple ngram lm with the command below. by default the order is 5, 3 is good as well. The text from the training data is used. Modify the script directly to change the path
```
python make_lm.py
```


