#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converter from simple-xml to BIO format.
"""

import argparse
import json
import os
import pandas as pd
import re
import string


def get_slots(src_utt_bio, tgt_utt_bio):
    #example: wake me up at [time : five am] [date : this week]
    slots = []
    for slot in src_utt_bio.split() + tgt_utt_bio.split():
        if slot != 'o':
            slots.append(slot[2:])
    d = dict.fromkeys(string.ascii_lowercase, 0)
    return dict(zip(set(slots), d))


def convert_to_xml_slots(src_utt, src_bio, tgt_utt, tgt_bio):
    #from: [wake me up at five am this week] & [o o o o b-time i-time b-date i-date]
    #to: wake me up at <a>five am<a> <b>this week<b>
    bio_repl_dict = get_slots(src_bio, tgt_bio)
    src_utt_tokens = src_utt.split()
    tgt_utt_tokens = tgt_utt.split()

    if len(src_utt_tokens) == len(src_bio.split()):
        for idx, slot in enumerate(src_bio.split()):
            if slot != "o":
                affix = "<" + bio_repl_dict[slot[2:]] + ">"
                src_utt_tokens[idx] = affix + src_utt_tokens[idx] + affix

    if len(tgt_utt_tokens) == len(tgt_bio.split()):
        for idx, slot in enumerate(tgt_bio.split()):
            if slot != "o":
                affix = "<" + bio_repl_dict[slot[2:]] + ">"
                tgt_utt_tokens[idx] = affix + tgt_utt_tokens[idx] + affix

    src_utt = " ".join(src_utt_tokens).replace('<b> <b>', ' ').replace('<a> <a>', ' ') \
                                      .replace('<d> <d>', ' ').replace('<c> <c>', ' ')
    tgt_utt = " ".join(tgt_utt_tokens).replace('<b> <b>', ' ').replace('<a> <a>', ' ') \
                                      .replace('<d> <d>', ' ').replace('<c> <c>', ' ')
    return [src_utt, tgt_utt]


def convert_to_bio(utt):
    #from: wake me up at <a>five am<a> <b>this week<b>
    #to:   o    o  o  o     b-a  i-a      b-b  i-b
    utt_tokens = utt.split()
    bio_tokens = []
    slot_begin = False
    slot_name = ""

    for idx, token in enumerate(utt_tokens):
        if token.startswith('<'):
            tag_end = token.index('>')
            slot_name = token[1:tag_end]
            bio_tokens.append('b-'+slot_name)
            slot_begin = True
        elif slot_begin:
            bio_tokens.append('i-'+slot_name)
        else:
            bio_tokens.append('o')

        if token.endswith('>'):
            slot_begin = False

    return " ".join(bio_tokens)


def get_massive_slots(annot_utt):
    #example: wake me up at [time : five am] [date : this week]
    slots = []
    for m in re.findall(r'\[\w+ \:', annot_utt):
        slots.append(m.replace('[', '').replace(' :', ''))
    d = dict.fromkeys(string.ascii_lowercase, 0)

    return zip(d, set(slots))


def convert_massive_to_xml_slots(src_utt, tgt_utt):
    #from: wake me up at [time : five am] [date : this week]
    #to: wake me up at <a>five am<a> <b>this week<b>
    for k, v in get_massive_slots(src_utt):
        p = re.compile('\[' + v + '\s:\s([^\]]*)\]', re.VERBOSE)
        src_utt = p.sub(r"<" + k + '>\\1<' + k + ">", src_utt)
        tgt_utt = p.sub(r"<" + k + '>\\1<' + k + ">", tgt_utt)
    return src_utt, tgt_utt


def convert_massive_to_bio_slots(src_utt, tgt_utt):
    #from: wake me up at [time : five am] [date : this week]
    #to: wake me up at <a>five am<a> <b>this week<b>
    for k, v in get_massive_slots(src_utt):
        p = re.compile('\[' + v + '\s:\s([^\]]*)\]', re.VERBOSE)
        src_utt = p.sub(r"<" + v + '>\\1<' + v + ">", src_utt)
        tgt_utt = p.sub(r"<" + v + '>\\1<' + v + ">", tgt_utt)

    src_utt = convert_to_bio(src_utt)
    tgt_utt = convert_to_bio(tgt_utt)
    return src_utt, tgt_utt


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description='Convert TSV corpus to JSON (Huggingface compatible).')
    argparse.add_argument('-i', '--input',
                        help='')
    argparse.add_argument('-c', '--conversion_direction', type=str, default="bio_to_xml",
                    help='')
    argparse.add_argument('-u', '--uniq_src', type=str, default="true",
                    help='')
    argparse.add_argument('-f', '--format', type=str, default="leyzer",
                    help='')
    args = argparse.parse_args()

    tsv_corpus = pd.read_csv(args.input, sep='\t')

    src_utts = []
    for idx, row in tsv_corpus.iterrows():
        if not row['src_utt'] in src_utts:
            if args.uniq_src == "true":
                src_utts.append(row['src_utt'])

            if args.format == 'leyzer':
                if args.conversion_direction == "bio_to_xml":
                    en_utt_xml, pl_utt_xml = convert_to_xml_slots(row['src_utt'], row['src_bio'],
                            row['tgt_utt'], row['tgt_bio'])
                    print("\t".join([row['domain'], row['intent'], row['level'], row['verb'],
                        row['src_utt'], en_utt_xml, row['src_bio'], row['tgt_utt'], pl_utt_xml, row['tgt_bio']]))
                elif args.conversion_direction == "xml_to_bio":
                    src_bio = convert_to_bio(row['src_xml'])
                    tgt_bio = convert_to_bio(row['tgt_xml'])
                    print("\t".join([row['domain'], row['intent'], row['level'], row['verb'],
                        row['src_utt'], row['src_xml'], src_bio, row['tgt_utt'], row['tgt_xml'], tgt_bio]))
                else: #if no conversion is expected (only uniq utts are expected)
                    print("\t".join([row['domain'], row['intent'], row['level'], row['verb'],
                        row['src_utt'], row['src_bio'], row['tgt_utt'], row['tgt_bio']]))
            elif args.format == 'simple': #two columns: src_(xml|bio), tgt_(xml|bio)
                if args.conversion_direction == "bio_to_xml":
                    src_utt_xml, tgt_utt_xml = convert_to_xml_slots(row['src_utt'], row['src_bio'],
                            row['tgt_utt'], row['tgt_bio'])
                    print("\t".join([src_xml, tgt_xml]))
                elif args.conversion_direction == "xml_to_bio":
                    src_bio = convert_to_bio(row['src_xml'])
                    tgt_bio = convert_to_bio(row['tgt_xml'])
                    print("\t".join([str(row['id']), row['src_utt'], src_bio, row['tgt_utt'], tgt_bio]))
            elif args.format == 'massive':
                if args.conversion_direction == "massive_to_bio":
                    src_annot = row['src_annot']
                    tgt_annot = row['tgt_annot']
                    src_xml, tgt_xml = convert_massive_to_xml_slots(src_annot, tgt_annot)
                    src_bio, tgt_bio = convert_massive_to_bio_slots(src_annot, tgt_annot)

                    print("\t".join([str(row['id']), row['src_utt'], src_xml, src_bio, row['tgt_utt'], tgt_xml, tgt_bio]))
            else:
                print("Format unknown.")
