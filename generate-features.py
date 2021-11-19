#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import shutil

import networkx as nx
import stanza
from tqdm import tqdm

from preprocess.utils import Samples
import preprocess.parsing


MODEL_DIR = './stanza_resources'


def export_all_subgraphs(samples: Samples,
                         graph: nx.DiGraph,
                         graphs_dir: str) -> None:
    for sample in tqdm(samples):
        fpath = os.path.join(graphs_dir, sample.fname + '.html')
        preprocess.parsing.export_subgraph(sample.fname, graph, fpath)


def main(root_dir: str) -> None:
    root_dir = root_dir.removeprefix('./')
    samples = preprocess.parsing.load_dataset_files(root_dir,
                                                    ignore_no_relations=True,
                                                    verbosity_level=2)

    nlp = stanza.Pipeline(
        lang='en', package='craft', dir=MODEL_DIR, logging_level='WARN')

    documents = preprocess.parsing.annotate(samples=samples, pipeline=nlp)
    graph = preprocess.parsing.create_feature_graph(
        documents=documents, samples=samples,
        ignore_intra_sent_relations=True)

    graphs_dir = os.path.join('graphs', root_dir, 'full')
    print(f"EXPORTING FULL GRAPHS TO '{graphs_dir}'...")
    export_all_subgraphs(samples, graph, graphs_dir)

    pos_tags = ['PUNCT']
    deprels = ['aux', 'det', 'punct', 'discourse']
    graph = preprocess.parsing.create_feature_graph(
        documents=documents, samples=samples,
        ignore_pos=pos_tags,ignore_deprel=deprels,
        ignore_intra_sent_relations=True)

    graphs_dir = os.path.join('graphs', root_dir, 'simple')
    print(f"EXPORTING SIMPLIFIED GRAPHS TO '{graphs_dir}'...")
    export_all_subgraphs(samples, graph, graphs_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage=('python3 generate_features.py PATH/TO/DATASET/ROOT'),
        description=('Process the BioNLP 2019 BB-Rel dataset into relational '
                     'features in a format suitable for propositionalization.')
    )

    parser.add_argument(
        'ROOTDIR',
        type=str,
        default=MODEL_DIR,
        help=('The path to the directory containing the text and annotation '
              'files from BB-Rel.')
    )

    args = parser.parse_args()
    stanza.download(lang='en', package='craft', model_dir=MODEL_DIR)
    main(args.ROOTDIR)
