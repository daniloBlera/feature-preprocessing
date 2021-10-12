#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from typing import NoReturn

import stanza

import preprocess.parsing


DATA_ROOT_DIR = './bionlp-dataset/BioNLP-OST-2019_BB-rel_dev'
MODEL_DIR = './stanza_resources'


def main(rootdir: str) -> NoReturn:
    samples = preprocess.parsing.load_dataset_files(rootdir, verbosity_level=0)
    samples = samples[:2]   # Reducing the samples for faster tests

    stanza.download(lang='en', model_dir=MODEL_DIR)
    nlp = stanza.Pipeline('en', dir=MODEL_DIR, preprocessors='tokenize')
    documents = preprocess.parsing.annotate(samples=samples, pipeline=nlp)
    graph = preprocess.parsing.create_feature_graph(documents, samples)

    for sample in samples:
        preprocess.parsing.export_subgraph(sample.fname, graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage='python3 generate_features PATH/TO/DATASET/ROOT',
        description=(
            'Process the BioNLP 2019 BB-Rel dataset into relational features '
            'in a format suitable for propositionalization.'
        )
    )

    parser.add_argument(
        'ROOTDIR',
        type=str,
        default=MODEL_DIR,
        help=('The path to the directory containing the text and annotation '
              'files from BB-Rel.')
    )
    args = parser.parse_args()
    main(args.ROOTDIR)
