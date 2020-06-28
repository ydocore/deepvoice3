# coding: utf-8
from deepvoice3_pytorch.frontend.text.symbols import symbols
from deepvoice3_pytorch.frontend.text.numbers import normalize_numbers

import nltk
from random import random

n_vocab = len(symbols)

_arpabet = nltk.corpus.cmudict.dict()


def _maybe_get_arpabet(word, p):
    #if word contains punctuation, it cannot change phonemes.
    #e.g. word='Printing,' cannot change, but word='Printing' can change phonemes
    if len(word) != 0 and word[-1] in '!,.:;?':
        punc = ' %' + word[-1] if word[-1] in '!.?' else ' %'
        word = word[:-1]
    else:
        punc = None
    try:
        phonemes = _arpabet[word.lower()][0]
        phonemes = " ".join(phonemes)
        phonemes = '{%s}' %phonemes
        phonemes = phonemes + punc if punc is not None else phonemes
    except KeyError:
        return word + punc if punc is not None else word

    word = word + punc if punc is not None else word
    return phonemes if random() < p else word


def mix_pronunciation(text, p):
    #text = '%'.join(word for word in text.split(', '))
    text = ' '.join(_maybe_get_arpabet(word, p) for word in text.split(' '))
    return text


def text_to_sequence(text, p=0.0):
    text = normalize_numbers(text)
    text = text.replace('\r', '')
    text = text + '.' if text[-1] not in '!,.:;?' else text
    if p >= 0:
        text = mix_pronunciation(text, p)
    from deepvoice3_pytorch.frontend.text import text_to_sequence
    text = text_to_sequence(text, ["english_cleaners"])
    return text


from deepvoice3_pytorch.frontend.text import sequence_to_text

#test
if __name__ == '__main__':
    print('input ratio:')
    p = float(input())
    print('input English sentence:')
    text = input()
    seq = text_to_sequence(text, p)
    print('sequence:{}'.format(seq))
    seq2text = sequence_to_text(seq)
    print('sequence to text:{}'.format(seq2text))