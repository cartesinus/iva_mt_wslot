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
    argparse.add_argument('-s', '--src',
                        help='Source tsv with JSGF expands or patterns, same as generate_patterns.py output but with columns separated by tab.')
    argparse.add_argument('-t', '--trg',
                        help='Target tsv with JSGF expands or patterns, same as generate_patterns.py output but with columns separated by tab.')
    argparse.add_argument('-d', '--dictionary',
                        help='Dictionary with possible verb translation from src to trg (list)')
    args = argparse.parse_args()

    eval = {
        "Airconditioner": {"pass": 0, "fail": 0, "no_verb": 0},
        "Calendar": {"pass": 0, "fail": 0, "no_verb": 0},
        "Console": {"pass": 0, "fail": 0, "no_verb": 0},
        "Contacts": {"pass": 0, "fail": 0, "no_verb": 0},
        "Email": {"pass": 0, "fail": 0, "no_verb": 0},
        "Facebook": {"pass": 0, "fail": 0, "no_verb": 0},
        "Fitbit": {"pass": 0, "fail": 0, "no_verb": 0},
        "Gdrive": {"pass": 0, "fail": 0, "no_verb": 0},
        "Instagram": {"pass": 0, "fail": 0, "no_verb": 0},
        "News": {"pass": 0, "fail": 0, "no_verb": 0},
        "Phone": {"pass": 0, "fail": 0, "no_verb": 0},
        "Slack": {"pass": 0, "fail": 0, "no_verb": 0},
        "Speaker": {"pass": 0, "fail": 0, "no_verb": 0},
        "Spotify": {"pass": 0, "fail": 0, "no_verb": 0},
        "Translate": {"pass": 0, "fail": 0, "no_verb": 0},
        "Twitter": {"pass": 0, "fail": 0, "no_verb": 0},
        "Weather": {"pass": 0, "fail": 0, "no_verb": 0},
        "Websearch": {"pass": 0, "fail": 0, "no_verb": 0},
        "Wikipedia": {"pass": 0, "fail": 0, "no_verb": 0},
        "Yelp": {"pass": 0, "fail": 0, "no_verb": 0},
        "Youtube": {"pass": 0, "fail": 0, "no_verb": 0}
    }

    with open(args.dictionary) as f:
        translations = json.load(f)

    trg_corpus = pd.read_csv(args.trg, sep='\t')
    src_corpus = pd.read_csv(args.src, sep='\t')
    for k, v in eval.items():
        eval[k]['total'] = src_corpus[src_corpus['domain'] == k]['domain'].count()

    i = 0
    for idx, src_row in src_corpus.iterrows():
        filtered_trg = trg_corpus[(trg_corpus['domain']==src_row['domain']) & (trg_corpus['intent']==src_row['intent']) & (trg_corpus['level']==src_row['level'])]
        src_verb = get_verb(src_row['utterance'], translations)
        if not src_verb:
            eval[src_row['domain']]['no_verb'] += 1

        for idx, trg_row in filtered_trg.iterrows():
            matching_trans_verb = False
            if src_verb:
                for trg_verb in translations[src_verb]:
                    if trg_verb in trg_row['utterance'].split():
                        matching_trans_verb = True
#                        print("pass\t", "\t".join([src_row['domain'], src_row['intent'], src_row['level'], src_verb, src_row['utterance'], trg_row['utterance']]))

            if matching_trans_verb:
                eval[src_row['domain']]['pass'] += 1
            else:
                eval[src_row['domain']]['fail'] += 1
#                print("fail\t", "\t".join([src_row['domain'], src_row['intent'], src_row['level'], "-", src_row['utterance'], trg_row['utterance']]))

    print(eval)

