#coding: utf-8
"""
Preprocess dataset

usage: text_preprocessing.py <in_dir> 

"""
import re
from docopt import docopt

def txt_prepro(path):
    with open(path+"train.txt", "rb") as f:
        lines = f.readlines()
    l = lines[0].decode("utf-8").split("|")
    texts = list(map(lambda l: l.decode("utf-8").split("|")[3], lines))
    with open(path+"train_ch.txt","w",encoding="utf-8") as w:
        for line,text in zip(lines,texts):
            line = line.decode("utf-8").split("|")
            line.pop()
            text=text.upper()
            rep=[]
            text = re.sub(", |:|;|,","%",text)
            if text[-3] == ".":
                #import pdb; pdb.set_trace()
                text = text.replace(".","%.")
            elif text[-3] == "?":
                text = text.replace("?","%?")
            elif "%\r\n" in text:
                text = text.replace("%\r\n","%.\r\n")
            else:
                text = text.replace("\r\n","%.\r\n")
            w.write('|'.join([str(x) for x in line])+ '|' + text)

if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args["<in_dir>"]
    txt_prepro(in_dir)
