import re

def clean_sent(sent):
    sent = sent.lower()
    sent = re.sub(r"[\.\-\[\],\n;:?]"," ", sent)
    return sent