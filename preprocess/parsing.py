#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This modules provides the top-level pre-processing functions."""
import os
import re
from typing import Optional

from networkx import DiGraph
from networkx.algorithms.components import node_connected_component
from pyvis.network import Network
from stanza import Pipeline
from tqdm import tqdm

from . import utils


def load_dataset_files(rootdir: str,
                       no_labels: bool = False,
                       ignore_no_relations: bool = False,
                       verbosity_level: int = 1) -> utils.Samples:
    """Read all texts and annotations files from a directory

    Build a dataset of samples, storing the files' name, raw text and
    annotations. Note that on validation datasets the A2 annotations, i.e. the
    labels, will be unavailable.

    Arguments:
        rootdir: str
            The path to the root of the dataset files.

        no_labels: True | False
            Ignore missing labels (a2 annotations).

        ignore_no_relations: True | False
            Skip loading the files without entity relations.

        verbosity_level: int
            Print information about the processing - 0: no output, 1: progress
            bar, 2: detailed information.

    Return: util.Samples
        A list of samples containing the document's file name, the text and the
        annotations. Note that label annotations (a2) will be set to `None` on
        validation datasets.
    """
    samples = list()
    fnames = [e.strip('.txt') for e in os.listdir(rootdir)
              if e.endswith('.txt')]

    if verbosity_level == 1:
        print('Processing samples...')
        fnames_wrapper = tqdm(fnames)
    else:
        fnames_wrapper = fnames

    for fname in fnames_wrapper:
        a1 = utils.get_annotations_a1(os.path.join(rootdir, fname) + '.a1')
        a2 = utils.get_annotations_a2(os.path.join(rootdir, fname) + '.a2')
        txt = utils.read_rawtext(os.path.join(rootdir, fname) + '.txt')

        if ignore_no_relations and isinstance(a2, list) and not a2:
            continue

        samples.append(utils.Sample(fname=fname, txt=txt, a1=a1, a2=a2))

    if verbosity_level == 2:
        num_a1 = 0
        num_a2 = 0
        print('IDX: FILENAME (without extension)')
        for (i, s) in enumerate(samples):
            if s.a1 is not None:
                num_a1 += 1
            if s.a2 is not None:
                num_a2 += 1

            print(f'{i:>3}: {s.fname}')

        print(f'\nNUMBER OF TEXT FILES:     {len(samples)}')
        print(f'NUMBER OF A1 ANNOTATIONS: {num_a1}')
        print(f'NUMBER OF A2 ANNOTATIONS: {num_a2}')

    return samples


def annotate(samples: utils.Samples,
             pipeline: Pipeline,
             verbosity_level: int = 1) -> utils.Documents:
    """Annotate the samples with the given pipeline.

    Annotate the list of text documents and, for internal purposes, add a few
    extra attributes to the document, sentence and word objects:

    *   The text document's filename into `document.fname`;
    *   The sentence's index relative to the document as `sentence.idx`;
    *   A reference to the sentence from the word and token, as `word.sent`
        and `token.sent`.

    Arguments:
        samples: utils.Samples
            A list of text documents.

        nlp: Pipeline
            A configured stanza Pipeline object.

        verbosity_level: 0 | 1 | 2
            How much information should be printed - 0: silence, 1: progress
            bar, 2: document sentences and tokens.

    Return: utils.Documents
        A list of stanza-annotated documents.
    """
    docs = list()
    if verbosity_level == 1:
        print('Annotating documents...')
        samples_wrapper = tqdm(samples)
    else:
        samples_wrapper = samples

    for (sample_idx, sample) in enumerate(samples_wrapper):
        if verbosity_level == 2:
            print(f'\n# TEXT {sample_idx+1:0>3}\t{sample.fname}')

        doc = pipeline(sample.txt)
        # Adding the document's filename (without the extension)
        doc.fname = sample.fname
        doc.entities = dict()
        doc.relations = dict()
        doc_entities = dict()

        for (sent_idx, sent) in enumerate(doc.sentences):
            sent.idx = sent_idx # Adding the sentence's index in the document

            if verbosity_level == 2:
                print(f'Sent {sent_idx+1:0>2},',
                      f'Words: {[w.text for w in sent.words]}')

            for word in sent.words:
                # Adding a reference to the sentence
                word.sent = sent
                word.parent.sent = sent
                token = word.parent

                # Storing words from annotated entities
                for ent in sample.a1:
                    if ent.id not in doc_entities:
                        doc_entities[ent.id] = {
                            'fname': doc.fname,
                            'type': ent.type,
                            'text': ent.text,
                            'words': list(),
                            'is_discontinuous': len(ent.idxs) > 1
                        }

                    for boundary in ent.idxs:
                        if utils.check_boundaries_overlap(
                                token.start_char, token.end_char,
                                boundary.start, boundary.end):
                            doc_entities[ent.id]['words'].append(word)

        for (ent_id, ent) in doc_entities.items():
            ent_type = ent['type']
            sent_idx = ent['words'][0].sent.idx
            entity = utils.Entity(
                fname=ent['fname'],
                id=ent_id,
                type=ent_type,
                text=ent['text'],
                sent_idx=sent_idx,
                words=ent['words'],
                is_discontinuous=ent['is_discontinuous']
            )
            doc.entities[ent_id] = entity

        for rel in sample.a2:
            e1 = doc.entities[rel.e1_id]
            e2 = doc.entities[rel.e2_id]
            relation = utils.Relation(
                fname=sample.fname, type=rel.type, id=rel.id, e1=e1, e2=e2)
            doc.relations[rel.id] = relation

        docs.append(doc)

    return docs


def create_feature_graph(documents: utils.Documents,
                         samples: utils.Samples,
                         ignore_pos: list[str] = None,
                         ignore_deprel: list[str] = None,
                         ignore_intra_sent_relations: bool = False
                         ) -> DiGraph:
    """Build a graph with morphological and relational features.

    Create and populate a directed graph with the morphological and relational
    features extracted from the list of annotated documents. The graph structure
    is roughly:

        Document --contains-> Sentence ...  (A document contains sentences)
        Sentence --contains-> Token ...     (A sentence contains tokens)

        Relation --entity_1-> Entity
        Relation --entity_2-> Entity
          Entity --contains-> Token         (An entity has (rightmost) token)

    Where each node has the following structure:

    Document
    key:
        f'{fname}'
    properties:
        * node_prop:        'DOCUMENT'
        * fname:            str
        * text:             str
        * label:            str                     (for pyvis)

    Sentence
    key:
        f'{fname}:SENT-{sent_idx}'
    properties:
        * node_prop:        'SENTENCE'
        * fname:            str
        * sent_idx:         int
        * text:             str
        * label:            str                     (for pyvis)

    Token
    key:
        f'{fname}:WORD-{end_idx}'
    properties:
        * node_prop:        'TOKEN'
        * fname:            str
        * start_idx:        int
        * end_idx:          int
        * text:             str
        * upos:             str
        * lemma:            str
        * label:            str                     (for pyvis)

    Relation
    key:
        f'{fname}:RELATION-id'
    properties:
        * node_prop:        'RELATION'
        * id:               'R[0-9]+'
        * type:             'Lives_In|Exhibits'
        * label:            str                     (for pyvis)

    Entity
    key:
        f'{fname}:ENTITY-{id}'
    properties:
        * node_prop:        'ENTITY'
        * id:               'E[0-9]+'
        * type:             'Phenotype|Microorganism|Habitat|Geographical'
        * boundaries:       utils.Boundaries
        * text:             str
        * is_discontinuous: True | False
        * label:            str                     (for pyvis)

    Arguments:
        documents: utils.Documents
            A list of stanza annotated documents.

        samples: utils.Samples
            A list of text and annotation samples.

        ignore_pos: list[str]
            A list of Part-of-Speech tags to ignore.

        ignore_deprel: list[str]
            A list of (universal) dependency relation tags to ignore.

        ignore_intra_sent_relations: True | False
            If relations between entities from different sentences should be
            skipped.

    Return: DiGraph
        A graph of features built from a list of (stanza) annotated documents.
    """
    feature_graph = DiGraph()
    utils.insert_annotated_document_features(documents,
                                             feature_graph,
                                             ignore_pos,
                                             ignore_deprel)
    utils.insert_entity_relation_features(documents,
                                          feature_graph,
                                          ignore_intra_sent_relations)

    return feature_graph


def export_subgraph(node_label, graph: DiGraph, filepath=None) -> None:
    """Export the document's subgraph for visualization.

    Given a node label, export to file a subgraph of all connected nodes.
    Normally you would use this function to export a document's subgraph
    containing its sentences, tokens, entities and relations.

    Arguments:
        node_label: str
            The label of a node in the feature graph - most likely a document's
            filename.

        graph: DiGraph
            A graph of features built from a list of (stanza) annotated
            documents.
    """
    ug = graph.to_undirected()
    connected_nodes = node_connected_component(ug, node_label)
    sg = graph.subgraph(connected_nodes)
    net = Network('1000px', '1000px', directed=True)
    net.from_nx(sg)

    if filepath is None:
        filepath = os.path.join('graphs', node_label + '.html')

    if not os.path.exists((dirname := os.path.dirname(filepath))):
        os.makedirs(dirname)

    net.save_graph(filepath)
