#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import argparse
import json
import os
import csv
import pandas as pd


def get_verb(utterance, translations):
    for word in utterance.split():
        if word in translations:
            return word
    return None


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description='Expand JSGF grammars.')
    argparse.add_argument('-i', '--input',
                        help='Output of MT that returned multiple variants: en, tgt1, ...')
    argparse.add_argument('-d', '--dictionary',
                        help='Dictionary with possible verb translation from src to trg (list)')
    argparse.add_argument('-m', '--mode', type=str, default='variant',
                        help='Either single or variant.')
    args = argparse.parse_args()

    with open(args.dictionary) as f:
        translations = json.load(f)

    mt_output = pd.read_csv(args.input, sep='\t')

    for idx, mt_out in mt_output.iterrows():
        src_verb = get_verb(mt_out['en'], translations)

        if args.mode == 'variant':
            match_verb = 0
            if src_verb:
                for trg_verb in translations[src_verb]:
                    if trg_verb in " ".join([mt_out['pl1'], mt_out['pl2'], mt_out['pl3'], mt_out['pl4'], mt_out['pl5'], mt_out['pl6']]).split():
                        match_verb += 1
                match_percent = match_verb / len(translations[src_verb])
                print("pass\t",src_verb, "\t", ",".join(translations[src_verb]), "\t", match_percent)
            else:
                print("fail\t", mt_out['en'], "\t", ",".join([mt_out['pl1'], mt_out['pl2'], mt_out['pl3'], mt_out['pl4'], mt_out['pl5'], mt_out['pl6']]), "\t 0.0")
        elif args.mode == 'single':
            match_verb = 0
            if src_verb:
                for trg_verb in translations[src_verb]:
                    if trg_verb in mt_out['pl'].split():
                        match_verb += 1
                match_percent = match_verb / len(translations[src_verb])
                print("pass\t",src_verb, "\t", ",".join(translations[src_verb]), "\t", match_percent)
            else:
                print("fail\t", mt_out['en'], "\t", mt_out['pl'], "\t 0.0")
        else:
            print('This is neither single or variant mode.')
