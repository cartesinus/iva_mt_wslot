#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import argparse
import json
import os
import csv
import pandas as pd
import numpy as np
import re
from evaluate import load
import sys


def get_verb(utterance, translations):
    for word in utterance.split():
        if word in translations:
            return word
    return None


def read_jsonl(filename):
    with open(filename, 'r') as json_file:
        json_list = list(json_file)

    return [json.loads(x) for x in json_list]


def get_slots(annot_utt):
    #example: wake me up at [time : five am] [date : this week]
    slots = []
    for m in re.findall(r'\[\w+ \:', annot_utt):
        slots.append(m.replace('[', '').replace(' :', ''))

    return set(slots)


def find_best_bitext_candidate(memory):
    best_score = 0.0
    best_candidate = []

    for candidate in memory:
        if candidate[8] > best_score:
            best_candidate = candidate
            best_score = candidate[8]

    return best_candidate


def sent_similarity(a, b, model_type="USE_multi"):
    """Returns the similarity scores"""
    if model_type == "USE_multi":
      sts_encode1 = tf.nn.l2_normalize(model([a]), axis=1)
      sts_encode2 = tf.nn.l2_normalize(model([b]), axis=1)
    elif model_type == "XLR_multi":
      sts_encode1 = tf.nn.l2_normalize(model(tf.constant([a])), axis=1)
      sts_encode2 = tf.nn.l2_normalize(model(tf.constant([b])), axis=1)
    else:
        return False

    cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
    clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)

    scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
    return float(scores.numpy())


def sub_identical_slots(src_utt, src_bio, tgt_utt, tgt_bio):
    ident_slots = ['value', 'pathname', 'from', 'to', 'filename', 'phone_number', 'email',
            'sender_address', 'picture_url', 'weight', 'mime_type', 'count', 'filter',
            'hashtag', 'portal', 'sender', 'channel', 'channel_id', 'username', 'percent',
            'playlist', 'song', 'artist', 'seek_time', 'review_count', 'name']
    src_tokens = src_utt.split()
    src_bio_tokens = src_bio.split()
    tgt_bio_tokens = tgt_bio.split()
    tgt_utt_tokens = tgt_utt.split()
    slot_values = {}
    slot_bios = {}
    utt_tokens = {}

    slot_begin = False
    for idx, src_slot in enumerate(src_bio_tokens):
        if src_slot != "o" and src_slot[2:] in ident_slots:
            if src_slot[0:2] == "b-":
                slot_begin = src_slot[2:]
                current_slot = src_slot[2:]
                slot_values[src_slot[2:]] = src_tokens[idx]
                slot_bios[src_slot[2:]] = [src_bio_tokens[idx]]
                utt_tokens[src_slot[2:]] = [src_tokens[idx]]
                src_bio_len = 1
                src_start = idx
            elif src_slot[0:2] == "i-" and current_slot == src_slot[2:]:
                slot_values[src_slot[2:]] += " " + src_tokens[idx]
                slot_bios[src_slot[2:]].append(src_bio_tokens[idx])
                utt_tokens[src_slot[2:]].append(src_tokens[idx])
                src_bio_len +=1
        if slot_begin and (src_slot == 'o' or
                           idx == len(src_bio_tokens)-1 or
                           (src_slot[2:] and src_slot[2:] != slot_begin)): #slot finished
            if src_slot == 'o' or src_slot[2:] != slot_begin:
                src_slot = src_bio_tokens[idx-1]
            if not 'b-'+src_slot[2:] in tgt_bio_tokens:
                continue

            tgt_start = tgt_bio_tokens.index('b-'+src_slot[2:])
            tgt_bio_len_buffer = 0
            if 'o' in tgt_bio_tokens[tgt_start:]:
                tgt_slot_bios = tgt_bio_tokens[tgt_start:tgt_start+tgt_bio_tokens[tgt_start:].index('o')]
            else:
                tgt_slot_bios = tgt_bio_tokens[tgt_start:]
            if len(slot_bios[src_slot[2:]]) != len(tgt_slot_bios):
                tgt_bio_len_buffer = len(tgt_slot_bios) - len(slot_bios[src_slot[2:]])
            tgt_bio_tokens = tgt_bio_tokens[0:tgt_start] + slot_bios[src_slot[2:]] + tgt_bio_tokens[tgt_start+src_bio_len+tgt_bio_len_buffer:]
            tgt_utt_tokens = tgt_utt_tokens[0:tgt_start] + utt_tokens[src_slot[2:]] + tgt_utt_tokens[tgt_start+src_bio_len+tgt_bio_len_buffer:]
            slot_begin = ""

    return [" ".join(tgt_utt_tokens), " ".join(tgt_bio_tokens)]


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description='Expand JSGF grammars.')
    argparse.add_argument('-s', '--src',
                        help='Input file representing source language. If layzer format selected '\
                        'then same as generate_patterns.py output (columns separated by tab). If '\
                        'massive format selected then identical to massive 1.1.')
    argparse.add_argument('-t', '--tgt',
                        help='Input file representing target language. If layzer format selected '\
                        'then same as generate_patterns.py output (columns separated by tab). If '\
                        'massive format selected then identical to massive 1.1.')
    argparse.add_argument('-o', '--output',
                        help='File with bitext.')
    argparse.add_argument('-d', '--dictionary',
                        help='Dictionary with possible verb translation from src to tgt (list)')
    argparse.add_argument('-f', '--format', type=str, default="leyzer",
                        help='Format if input.')
    argparse.add_argument('-m', '--match_criteria', type=str, default="n_best_use",
                        help='Options: "n_best_embed" is 1 best candidate selected with multilingual '\
                             'embedding (here USE), "all_possible" (default) all possible combinations.')
    argparse.add_argument('-l', '--token_diff', type=int, default=3,
                        help='Reject bitext candidates with difference in tokens larger than this value. '\
                             'Default = 3.')
    argparse.add_argument('-c', '--utt_clean_strategy', type=str, default="no_strategy",
                        help='If "remove_stopwords" then some stopwords are removed before filtering ' \
                             'sentences with --token_diff option')
    argparse.add_argument('-e', '--embed_model', type=str, default="USE_multi",
                        help='Embedding model. Either: "USE_multi" which is Universal Sentence ' \
                             'Encoder or "XLR_multi" which is XLM Roberta.')
    argparse.add_argument('-b', '--with_bio', type=str, default="false",
                        help='')

    args = argparse.parse_args()

    if args.match_criteria in ['n_best_embed', 'n_best_embed_wslotsub'] :
        #@title Load the Universal Sentence Encoder's TF Hub module
        from absl import logging
        import tensorflow as tf
        import tensorflow_hub as hub
        import matplotlib.pyplot as plt
        import math
        import tensorflow_text

        if args.embed_model == "USE_multi":
            module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
            model = hub.load(module_url)
        elif args.embed_model == "XLR_multi":
            text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
            preprocessor = hub.KerasLayer("https://tfhub.dev/jeongukjae/xlm_roberta_multi_cased_preprocess/1")
            encoder_inputs = preprocessor(text_input)

            encoder = hub.KerasLayer("https://tfhub.dev/jeongukjae/xlm_roberta_multi_cased_L-12_H-768_A-12/1", trainable=True)
            encoder_outputs = encoder(encoder_inputs)
            pooled_output = encoder_outputs["pooled_output"]      # [batch_size, 768].
            sequence_output = encoder_outputs["sequence_output"]  # [batch_size, seq_length, 768].

            model = tf.keras.Model(text_input, pooled_output)
        else:
            sys.exit('Unknown embedding model selected.')

    with open(args.dictionary) as f:
        translations = json.load(f)

    with open(args.output, 'w') as out_file:
        if args.format == 'leyzer':
            src_corpus = pd.read_csv(args.src, sep='\t')
            tgt_corpus = pd.read_csv(args.tgt, sep='\t')
            memory = {'last_utt': ''}

            for idx, src_row in src_corpus.iterrows():
                src_verb = get_verb(src_row['utterance'], translations)
                src_utt = src_row['utterance']
                if args.utt_clean_strategy == "remove_stopwords":
                    src_tokens = [i for i in src_utt.split() if i != "the"]
                else:
                    src_tokens = src_utt.split()

                print(idx, '/', len(src_corpus))

                for idx, tgt_row in tgt_corpus[(tgt_corpus['domain']==src_row['domain'])
                                    & (tgt_corpus['intent']==src_row['intent'])].iterrows():

                    tgt_utt = tgt_row['utterance']
                    tgt_tokens = tgt_utt.split()

                    if src_verb:
                        for tgt_verb in translations[src_verb]:
                            token_diff = abs(len(src_tokens) - len(tgt_tokens))
                            if tgt_verb in tgt_tokens and token_diff <= int(args.token_diff):

                                if args.match_criteria == 'all_possible':
                                    out_file.write("\t".join([src_row['domain'], src_row['intent'], \
                                                              src_row['level'], src_verb, src_utt, \
                                                              tgt_utt]) + "\t" +  str(token_diff) + "\n")
                                    out_file.flush()
                                elif args.match_criteria == 'all_possible_wslotsub':

                                    tgt_bio = tgt_row['bio']
                                    tgt_utt_sub, tgt_bio_sub = sub_identical_slots(src_utt, src_row['bio'], tgt_utt, tgt_bio)
                                    if tgt_utt_sub != tgt_utt and tgt_bio_sub != tgt_bio:
                                        out_file.write("\t".join([src_row['domain'], src_row['intent'], \
                                                                  src_row['level'], src_verb, src_utt, \
                                                                  src_row['bio'], tgt_utt_sub, tgt_bio_sub]) + \
                                                                  "\t" +  str(token_diff) + "\n")
                                    out_file.flush()
                                elif args.match_criteria == 'n_best_embed':
                                    mem_idx = tgt_verb + src_utt
                                    use_similarity = sent_similarity(tgt_utt, src_utt)
                                    if not mem_idx in memory:
                                        memory[mem_idx] = []
                                    if mem_idx != memory['last_utt']:
                                        if memory['last_utt'] != '':
                                            result = find_best_bitext_candidate(memory[memory['last_utt']])
                                            out_file.write("\t".join(result[0:6]) + "\t" + \
                                                           str(result[6]) + "\n")
                                            out_file.flush()
                                            memory = {'last_utt': ''}
                                        memory[mem_idx] = []
                                        memory['last_utt'] = mem_idx
                                    memory[mem_idx].append([src_row['domain'], src_row['intent'], \
                                                            src_row['level'], src_verb, src_utt, \
                                                            tgt_utt, use_similarity])
                                elif args.match_criteria == 'n_best_embed_wslotsub':
                                    mem_idx = tgt_verb + src_utt
                                    tgt_bio = tgt_row['bio']
                                    tgt_utt_sub, tgt_bio = sub_identical_slots(src_utt, src_row['bio'],
                                            tgt_utt, tgt_bio)
                                    if tgt_utt_sub != tgt_utt:
                                        tgt_utt = tgt_utt_sub
                                        use_similarity = sent_similarity(tgt_utt, src_utt)
                                        if not mem_idx in memory:
                                            memory[mem_idx] = []
                                        if mem_idx != memory['last_utt']:
                                            if memory['last_utt'] != '':
                                                result = find_best_bitext_candidate(memory[memory['last_utt']])
                                                out_file.write("\t".join(result[0:8]) + "\t" + \
                                                               str(result[8]) + "\n")
                                                out_file.flush()
                                                memory = {'last_utt': ''}
                                            memory[mem_idx] = []
                                            memory['last_utt'] = mem_idx
                                        memory[mem_idx].append([src_row['domain'], src_row['intent'], \
                                                                src_row['level'], src_verb, src_utt, \
                                                                src_row['bio'], tgt_utt, tgt_bio, use_similarity])
                                else:
                                    sys.exit('Matching criteria unknown')


        elif args.format == 'massive':
            src_corpus = pd.DataFrame.from_dict(read_jsonl(args.src))
            tgt_corpus = pd.DataFrame.from_dict(read_jsonl(args.tgt))
            memory = {'last_utt': ''}

            for idx, src_row in src_corpus.iterrows():
                src_slots = get_slots(src_row['annot_utt'])
                src_utt = src_row['utt']
                src_verb = get_verb(src_utt, translations)
                if args.utt_clean_strategy == "remove_stopwords":
                    src_tokens = [i for i in src_utt.split() if i != "the"]
                else:
                    src_tokens = src_utt.split()

                print(idx, '/', len(src_corpus))

                for idx, tgt_row in tgt_corpus[(tgt_corpus['scenario']==src_row['scenario'])
                                    & (tgt_corpus['intent']==src_row['intent'])].iterrows():

                    tgt_utt = tgt_row['utt']
                    tgt_tokens = tgt_utt.split()
                    tgt_slots = get_slots(tgt_row['annot_utt'])
                    token_diff = abs(len(src_tokens) - len(tgt_tokens))

                    if src_verb and not src_slots and src_slots == tgt_slots and token_diff < args.token_diff:
                        for tgt_verb in translations[src_verb]:
                            if tgt_verb in tgt_row['utt'].split():
                                if args.match_criteria == 'all_possible':
                                    out_file.write("\t".join([src_row['partition'], src_row['scenario'], src_row['intent'],
                                          src_verb, src_utt, tgt_utt]) + "\t" + str(token_diff) + "\n")
                                    out_file.flush()
                                elif args.match_criteria == 'n_best_embed':
                                    mem_idx = tgt_verb + src_utt
                                    use_similarity = sent_similarity(tgt_utt, src_utt)
                                    if not mem_idx in memory:
                                        memory[mem_idx] = []
                                    if mem_idx != memory['last_utt']:
                                        if memory['last_utt'] != '':
                                            result = find_best_bitext_candidate(memory[memory['last_utt']])
                                            out_file.write("\t".join(result[0:8]) + "\t" + \
                                                           str(result[7]) + "\n")
                                            out_file.flush()
                                            memory = {'last_utt': ''}
                                        memory[mem_idx] = []
                                        memory['last_utt'] = mem_idx
                                    memory[mem_idx].append([src_row['partition'], src_row['scenario'], src_row['intent'], \
                                                            src_verb, src_utt, tgt_utt, use_similarity])
                                else:
                                    sys.exit('Matching criteria unknown')

        else:
            print('Input format unknown. Please have a look at --leyzer and --massive expected output '\
                  'and adjust your input.')


