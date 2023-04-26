#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converter from TOP format to IVA_MT input format (xml-like slot annotations, no intents).
"""

import argparse
import json
import os
import csv
import pandas as pd
import re
import string


def convert_top_to_massive(annot_utt):
    #from: Cancel alarm for [SL:PERIOD thursdays ]
    #to: Cancel alarm for [PERIOD : thursdays]
    p = re.compile('\[SL:(\w*) ', re.VERBOSE)
    annot_utt = p.sub(r"[\1 :", annot_utt)

    return annot_utt.replace(' ]', ']')

def get_top_slots_dict(sentence):
    #example: wake me up at [time : five am] [date : this week]
    slots = []
    slot_ints = []
    for m in re.findall(r'\[SL:(\w*) \[IN:(\w*) ', sentence):
        slot = m[0]
        intent = m[1]
        matching_string = slot + "#" + intent
        if matching_string not in slots:
            slots.append(matching_string)
            slot_ints.append(slot)
    #for m in re.findall(r'\[SL:(\w*) ([^\[])', sentence):
    for m in re.findall(r'\[SL:(\w*) ', sentence):
        if m not in slots and m not in slot_ints:
            slots.append(m)

    d = dict.fromkeys(string.ascii_lowercase, 0)

    return list(zip(d, slots))


def convert_to_xml_slots(utt, slot_dict):
    #from: wake me up at [time : five am] [date : this week]
    #to: wake me up at <a>five am<a> <b>this week<b>
    for k, v in reversed(slot_dict): #because of "small" span slots
        if "#" in v:
            v_split = v.split("#")
            p = re.compile('\[SL:' + v_split[0] + '\s\[IN:' + v_split[1] + '\s([^\]]*)\s\]', re.VERBOSE)
            utt = p.sub(r"<" + k + '>\\1<' + k + ">", utt)
        else:
            if v+"#" in slot_dict:
                p = re.compile('\[SL:' + v + '\s(\w*)\s\]', re.VERBOSE)
                #utt = p.sub(r"<" + k + '>\\1<' + k + ">", utt)
            else:
                p = re.compile('\[SL:' + v + '\s([^\]]*)\s\]', re.VERBOSE)
                utt = p.sub(r"<" + k + '>\\1<' + k + ">", utt)
    #that's probably bad idea, but...
    utt = re.sub('\[SL:(\w*)', '', utt)
    utt = utt.replace('<a> <a>', ' ').replace('<b> <b>', ' ')\
            .replace('<c> <c>', ' ').replace('<d> <d>', ' ')
    return utt

def get_top_intent_and_sentence(annot_sentence):
    tokenized_s = annot_sentence.split()
    intent = tokenized_s[0].replace('[IN:', '')
    sentence = " ".join(tokenized_s[1:-1])

    return intent, sentence


def restore_slots(sentence, slot_dict):
    """
    Restore slots sentence converted to xml back to MTOP format
    """
    for k, v in reversed(slot_dict):
        if "#" in v:
            v_split = v.split("#")
            p = re.compile('<' + k + '>([^<]*)<' + k + '>', re.VERBOSE)
            sentence = p.sub(r"[SL:" + v_split[0] + ' [IN:' + v_split[1] + ' \\1 ] ]', sentence)
        else:
            p = re.compile('<' + k + '>([^<]*)<' + k + '>', re.VERBOSE)
            sentence = p.sub(r"[SL:" + v + ' \\1 ] ', sentence)
    return sentence


def restore_intents(sent, slot_rest):
    """
    Restore intents sentence converted to xml back to MTOP format
    """
    tokenized = sent.split()
    main_int = tokenized[0]
    sentence = main_int + " " + slot_rest.strip() + " ]"

    return sentence.replace('  ', ' ')


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description='Convert MTOP corpus to IVA_MT input.')
    argparse.add_argument('-i', '--input', help='')
    argparse.add_argument('-o', '--output', help='')
    args = argparse.parse_args()

    top = pd.read_csv(args.input, sep='\t')
    for idx, row in top.iterrows():
        sentence = row['sentence_annotated']
        slot_dict = get_top_slots_dict(sentence)
        if slot_dict:
            conv_sentence = convert_to_xml_slots(sentence, slot_dict)
            conv_sentence = re.sub('\[IN:(\w*) ', "", conv_sentence).replace(']', '').strip()
            rest_sentence = restore_slots(conv_sentence, slot_dict).replace('  ', ' ').replace('  ', ' ')
            rest_sentence = restore_intents(sentence, rest_sentence)
            if sentence != rest_sentence:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write("id\tsentence\trestore_utt\txml_converted_utt\tslot_dict\n")
                    result = [str(idx), sentence, rest_sentence.strip(), conv_sentence, str(slot_dict)]
                    f.write("\t".join(result) + "\n")
                    f.flush()
