#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from typing import NoReturn

import stanza

import preprocess.parsing


DATA_ROOT_DIR = './bionlp-dataset/BioNLP-OST-2019_BB-rel_dev'
MODEL_DIR = './stanza_resources'


def main(root_dir: str, graphs_dir: str) -> NoReturn:
    samples = preprocess.parsing.load_dataset_files(
        root_dir, verbosity_level=0)
    stanza.download(lang='en', package='craft', model_dir=MODEL_DIR)
    nlp = stanza.Pipeline(lang='en', package='craft', dir=MODEL_DIR)
    documents = preprocess.parsing.annotate(samples=samples, pipeline=nlp)
    graph = preprocess.parsing.create_feature_graph(
        documents=documents, samples=samples)

    for sample in samples:
        fpath = os.path.join(graphs_dir, 'full', sample.fname + '.html')
        preprocess.parsing.export_subgraph(sample.fname, graph, fpath)

    pos_tags = ['PUNCT']
    deprels = ['aux', 'det', 'punct', 'discourse']
    graph = preprocess.parsing.create_feature_graph(documents=documents,
                                                    samples=samples,
                                                    ignore_pos=pos_tags,
                                                    ignore_deprel=deprels)
    for sample in samples:
        fpath = os.path.join(graphs_dir, 'simplified', sample.fname + '.html')
        preprocess.parsing.export_subgraph(sample.fname, graph, fpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage=('python3 generate_features.py '
               '--graphs-dir PATH/TO/GRAPHS/OUTPUT '
               'PATH/TO/DATASET/ROOT'),
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

    parser.add_argument(
        '--graphs-dir',
        type=str,
        default='graphs',
        help=("The path to where the graph files should be generated. "
              "DEFAULT: './graphs'")
    )

    args = parser.parse_args()
    main(args.ROOTDIR, args.graphs_dir)
