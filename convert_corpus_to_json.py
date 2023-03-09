#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import argparse
import json
import os
import csv
import pandas as pd
import re
import string


def get_slots(annot_utt):
    #example: wake me up at [time : five am] [date : this week]
    slots = []
    for m in re.findall(r'\[\w+ \:', annot_utt):
        slots.append(m.replace('[', '').replace(' :', ''))
    d = dict.fromkeys(string.ascii_lowercase, 0)

    return zip(d, set(slots))


def convert_to_flat_slots(en_utt, pl_utt):
    #from: wake me up at [time : five am] [date : this week]
    #to: wake me up at <a>five am<a> <b>this week<b>
    for k, v in get_slots(en_utt):
        p = re.compile('\[' + v + '\s:\s([^\]]*)\]', re.VERBOSE)
        en_utt = p.sub(r"<" + k + '>\\1<' + k + ">", en_utt)
        pl_utt = p.sub(r"<" + k + '>\\1<' + k + ">", pl_utt)
    return en_utt, pl_utt



if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description='Convert TSV corpus to JSON (Huggingface compatible).')
    argparse.add_argument('-i', '--input',
                        help='')
    argparse.add_argument('-o', '--output',
                        help='')
    argparse.add_argument('-s', '--flatten_slots', type=str, default="false",
                    help='')
    args = argparse.parse_args()

    tsv_corpus = pd.read_csv(args.input, sep='\t')
    json_corpus = {"data": []}

    for idx, row in tsv_corpus.iterrows():
        if args.flatten_slots == "true":
            src_utt, tgt_utt = convert_to_flat_slots(row['src_utt'], row['tgt_utt'])
        else:
            src_utt = row['src_utt']
            tgt_utt = row['tgt_utt']

        json_corpus['data'].append({"id": str(idx), "locale": str(row['locale']), "origin": str(row['origin']),
            "partition": str(row['partition']), "translation_utt": {"src": str(src_utt), "tgt": str(tgt_utt)},
            "translation_xml": {"src": str(row['src_xml']), "tgt": str(row['tgt_xml'])},
            "src_bio": str(row['src_bio']), "tgt_bio": str(row['tgt_bio'])}
        )

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(json_corpus, f, ensure_ascii=False, indent=4)
