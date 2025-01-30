#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: msra2mrc.py

import sys
sys.path.append("/home/tlf/Documents/mestrado/ner_llm/mrc-for-flat-nested-ner/src")

import os
import utils
from utils.bmes_decode import bmes_decode
import json


def convert_file(input_file, output_file, tag2query_file):
    """
    Convert MSRA raw data to MRC format
    """
    origin_count = 0
    new_count = 1
    tag2query = json.load(open(tag2query_file, encoding='utf-8'))
    mrc_samples = []
    contexts, labels = [], []
    with open(input_file, encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line:
                #print(line)
                context, label = line.split(" ")
                contexts.append(context)
                labels.append(label)
            else:

                tags = bmes_decode(char_label_list=[(char, label) for char, label in zip(contexts, labels)])
    #         tags = bmes_decode(char_label_list=[(char, label) for char, label in zip(src.split(), labels.split())])
                for label, query in tag2query.items():
                    start_position = [tag.begin for tag in tags if tag.tag == label]
                    end_position = [tag.end-1 for tag in tags if tag.tag == label]
                    mrc_samples.append(
                        {
                            "qas_id": "{}.{}".format(origin_count, new_count),
                            "context": " ".join(contexts),
                            "start_position": start_position,
                            "end_position": end_position,
                            "query": query,
                            "impossible": False if start_position and end_position else True,
                            "entity_label": label,
                            "span_position": [f"{start};{end}" for start, end in zip(start_position, end_position)]
                        }
                    )
                    new_count += 1

                contexts, labels = [], []
                origin_count += 1
                new_count = 1

    json.dump(mrc_samples, open(output_file, encoding="utf-8", mode='w'), ensure_ascii=False, sort_keys=True, indent=2)
    print(f"Convert {origin_count} samples to {new_count} samples and save to {output_file}")


def main():
    
    #msra_raw_dir = r"C:\Users\tarci\OneDrive\Área de Trabalho\mrc-for-flat-nested-ner\data"
    #msra_mrc_dir = r"C:\Users\tarci\OneDrive\Área de Trabalho\mrc-for-flat-nested-ner\mrc_data"
    #tag2query_file = r"C:\Users\tarci\OneDrive\Área de Trabalho\mrc-for-flat-nested-ner\src\ner2mrc\queries\lungrads.json"
    #os.makedirs(msra_mrc_dir, exist_ok=True)
    
    msra_raw_dir = "/home/tlf/Documents/mestrado/ner_llm/mrc-for-flat-nested-ner/data"
    msra_mrc_dir = "/home/tlf/Documents/mestrado/ner_llm/mrc-for-flat-nested-ner/mrc_data"
    tag2query_file = "/home/tlf/Documents/mestrado/ner_llm/mrc-for-flat-nested-ner/src/ner2mrc/queries/lungrads.json"
    os.makedirs(msra_mrc_dir, exist_ok=True)

    #msra_raw_dir = r"C:\Users\tarci\Downloads\en_conll03\en_conll03"
    #msra_mrc_dir = r"C:\Users\tarci\Downloads\en_conll03\en_conll03"
    #tag2query_file = r"C:\Users\tarci\OneDrive\Área de Trabalho\mrc-for-flat-nested-ner\src\ner2mrc\queries\en_conll_03.json"
    #os.makedirs(msra_mrc_dir, exist_ok=True)
    
    #msra_raw_dir = r"C:\Users\tarci\Downloads\flat_zh_msra\zh_msra"
    #msra_mrc_dir = r"C:\Users\tarci\Downloads\flat_zh_msra\zh_msra"
    #tag2query_file = r"C:\Users\tarci\OneDrive\Área de Trabalho\mrc-for-flat-nested-ner\src\ner2mrc\queries\zh_msra.json"
    #os.makedirs(msra_mrc_dir, exist_ok=True)
    
    
    #old_file = os.path.join(msra_raw_dir, f"teste.word.bmes")
    #new_file = os.path.join(msra_mrc_dir, f"mrc-ner.teste")
    #convert_file(old_file, new_file, tag2query_file)

    #for phase in ["train", "dev", "test"]:
    #    old_file = os.path.join(msra_raw_dir, f"{phase}.word.bmes.txt")
    #    new_file = os.path.join(msra_mrc_dir, f"mrc-ner.{phase}")
    #    convert_file(old_file, new_file, tag2query_file)
    
    #for phase in ["train", "dev", "test"]:
    #    old_file = os.path.join(msra_raw_dir, f"{phase}.char.bmes")
    #    new_file = os.path.join(msra_mrc_dir, f"mrc-ner.{phase}")
    #    convert_file(old_file, new_file, tag2query_file)

    #for phase in ["train", "dev", "test"]:
    #    old_file = os.path.join(msra_raw_dir, f"{phase}.word.bmes")
    #for phase in ["laudos_1_101", "laudos_102_202", "laudos_400_500"]:
    for phase in ["laudos_1_862", "laudos_863_962"]:
        old_file = os.path.join(msra_raw_dir, f"{phase}.bmes")
        new_file = os.path.join(msra_mrc_dir, f"mrc-ner.{phase}")
        convert_file(old_file, new_file, tag2query_file)


if __name__ == '__main__':
    main()
