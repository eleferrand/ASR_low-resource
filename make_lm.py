import re, os
import subprocess
import csv
from cleaning import clean_sent

text = ""
speakers = set()
lang = 'West_Xoon'
order = 5
with open(f"data/{lang}/train.csv", "r") as cfile:
    reader = csv.reader(cfile, delimiter=",", quotechar="|")
    for row in reader:
        speakers.add(os.path.basename(row[0]).split('_')[1])
        # text.append(row[1])
        print(row[1])
        text+=clean_sent(row[1])+"\n"
with open(f"text_{lang}.txt", "w") as tfile:
    tfile.write(text)
subprocess.call(f"kenlm/build/bin/lmplz -o {order} -S 1G <text_{lang}.txt >{lang}_{order}gram.arpa", shell=True)
