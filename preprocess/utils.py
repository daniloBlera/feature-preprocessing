#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provide utility functions and type hints for the `parsing` module.

The contents of this module shouldn't be imported directly, instead, you should
use the functions from the `preprocessing.parsing` module (the parsing.py file)
instead.
"""
from collections import namedtuple
from enum import Enum
import re

from typing import NoReturn
from typing import Optional
from typing import Union

import networkx as nx
from stanza.models.common.doc import Document
from stanza.models.common.doc import Sentence
from stanza.models.common.doc import Word


# Data structures
Sample = namedtuple('Sample', ['fname', 'txt', 'a1', 'a2'])

# Type Hints, hopefully to help understanding input-output
## Document and annotations
Samples = list[Sample]
AnnotationsA1 = list[str]
AnnotationsA2 = Optional[list[str]]
Annotations = Union[AnnotationsA1, AnnotationsA2]
Documents = list[Document]

## Feature-graph
DocumentNodeAttrs = dict[str, str]
SentenceNodeAttrs = dict[str, Union[str, int]]
WordNodeAttrs = dict[str, Union[str, int]]
EntityNodeAttrs = dict[str, str]
RelationNodeAttrs = dict[str, str]

## Entity and relation property
Boundary = dict[str, int]   # The boundary structure: {'start': int, 'end': int}
Boundaries = list[Boundary]
EntityParams = dict[str, Union[str, bool, Boundaries]]
RelationParams = dict[str, str]

# Property enums
class GraphEdgeType(Enum):
    DOC_TO_SENT = 'DOC-CONTAINS-SENT'
    SENT_TO_TOKEN = 'SENT-CONTAINS-TOKEN'
    DEPREL = 'TOKEN-DEPREL-HEAD'
    REL_TO_E1= 'RELATION-CONTAINS-ENTITY1'
    REL_TO_E2= 'RELATION-CONTAINS-ENTITY2'
    ENTITY = 'ENTITY-CONTAINS-RIGHTMOST-TOKEN'


class GraphNodeType(Enum):
    DOCUMENT = 'DOCUMENT'
    SENTENCE = 'SENTENCE'
    TOKEN = 'TOKEN'
    RELATION = 'RELATION'
    ENTITY = 'ENTITY'


class GraphColor(Enum):
    DOCUMENT = '#DC122A'
    SENTENCE = '#6666FF'
    TOKEN = '#66FF66'


def get_document_node_attrs(document: Document) -> DocumentNodeAttrs:
    """Create the graph's node attribute for a `Document`.

    Given an annotated document, build a property dict for a document node in
    the feature graph.

    Arguments:
        document: Document
            A stanza-annotated document.

    Return: DocumentNodeAttrs
        A dictionary containing the documents's features to be used by
        networkx's feature graph.
    """
    return {
        'fname': document.fname,
        'text': document.text,
        'label': f"DOC: {document.fname}",  # for pyvis
        'color': GraphColor.DOCUMENT.value # for pyvis
    }


def get_sentence_node_attrs(sentence: Sentence) -> SentenceNodeAttrs:
    """Create the graph's node attribute for a `Sentence`.

    Given an annotated sentence, build a property dict for a sentence node in
    the feature graph.

    Arguments:
        word: Sentence
            A stanza-annotated sentence.

    Return: SentenceNodeAttrs
        A dictionary containing the sentence's features to be used by
        networkx's feature graph.
    """
    return {
        'fname': sentence.doc.fname,
        'sent_idx': sentence.idx,
        'text': sentence.text,
        'label': f'SENT-{sentence.idx}',    # for pyvis
        'color': GraphColor.SENTENCE.value # for pyvis
    }


def get_word_node_attrs(word: Word) -> WordNodeAttrs:
    """Create the graph's node attribute for a `Word`.

    Build an attribute dict with the word's features. Note that we're using the
    term `Word` instead of `Token` to be closer to the implementation of these
    data structures in stanza. From stanza's documentation, a `Token` might
    hold more the a single word in the case of multi-word tokens. For more
    information please refer to
    'https://stanfordnlp.github.io/stanza/data_objects.html#token'.

    Arguments:
        word: Word
            A stanza-annotated word.

    Return: WordNodeAttrs
        A dictionary containing the word's features to be used by networkx's
        feature graph.
    """
    return {
        'fname': word.sent.doc.fname,
        'start_idx': word.parent.start_char,
        'end_idx': word.parent.end_char,
        'text': word.text,
        'upos': word.upos,
        'lemma': word.lemma,
        'label': f"WORD: '{word.text}'",    # for PyVis
        'color': GraphColor.TOKEN.value    # for pyvis
    }


def get_entity_node_attrs(entity: EntityParams) -> EntityNodeAttrs:
    """Create the graph's node attribute for an `Entity`.

    Given an entity annotation, build a property dict for an entity node in
    the feature graph.

    Arguments:
        entity: EntityParams
            An entity's parameter dictionary.

    Return: EntityNodeAttrs
        A dictionary containing the entity's features to be used by networkx's
        feature graph.
    """
    return {
        'id': entity['id'],
        'type': entity['type'],
        'boundaries': entity['boundaries'],
        'text': entity['text'],
        'is_discontinuous': entity['is_discontinuous'],
        'label': entity['id'] # for PyVis
    }


def get_relation_node_attrs(relation: RelationParams) -> RelationNodeAttrs:
    """Create the graph's node attribute for a `Relation`.

    Given a relation annotation, build a property dict for an relation node in
    the feature graph.

    Arguments:
        relation: RelationParams
            A relation's parameter dictionary.

    Return: RelationNodeAttrs
        A dictionary containing the relation's features to be used by networkx's
        feature graph.
    """
    return {
        'id': relation['id'],
        'type': relation['type'],
        'label': relation['id'] # for PyVis
    }


def is_valid_entity(line: str) -> bool:
    """Check if the line is a valid entity annotation."""
    regex = r'^T\d+\t\w+ \d+ \d+(;\d+ \d+)*\t.+$'
    return re.search(regex, line) is not None


def is_valid_relation(line: str) -> bool:
    """Check if the line is a valid relation annotation."""
    regex = r'^R\d+\t\w+ \w+:T\d+ \w+:T\d+$'
    return re.fullmatch(regex, line) is not None


def get_entity_param_dict(entity: str, filename: str) -> EntityParams:
    """Get the entity's parameters structure

    Given an entity string with with the format

        ID\tTYPE BOUNDARY[(; BOUNDARY)...]\tTEXT

    Create a dictionary with the following entity properties:

        * fname:            str
        * id:               str
        * type:             str
        * boundaries:       list[tuple[int, int]]
        * text:             str
        * is_discontinuous: True | False

    For example, given the entity line, from the document 'BB-rel-10496597'

        T14\tHabitat 603 611;657 685\tpatients with low-grade MALT lymphoma

    create a dictionary in with the structure

        {
            'fname':            'BB-rel-10496597',
            'id':               'T14',
            'type':             'Habitat',
            'boundaries':       [(603, 611), (657, 685)],
            'text':             'patients with low-grade MALT lymphoma',
            'is_discontinuous': True
        }

    Arguments:
        entity: str
            A single entity annotation line.

        filename: str
            The name of the document (without the extension) where this entity
            is from.

    Return: EntityParams
        A dictionary containing the parameters from the entity annotation line.
    """
    if not is_valid_entity(entity):
        raise ValueError("The argument doesn't match the expected format")

    [entity_id, data, text] = entity.split('\t')
    entity_type = re.search(r'^\w+', data).group()
    boundary_str = re.search(r'\d+ \d+(;\d+ \d+)*', data).group()
    boundaries = boundary_str.split(';')
    boundaries = [tuple(b.split(' ')) for b in boundaries]
    boundaries = [{'start': int(s), 'end': int(e)} for (s, e) in boundaries]

    return {
        'fname': filename,
        'id': entity_id,
        'type': entity_type,
        'boundaries': boundaries,
        'text': text,
        'is_discontinuous': len(boundaries) > 1
    }


def get_sample_entity_param_dicts(sample: Sample) -> list[EntityParams]:
    """Get the sample's entity parameter dictionaries.

    This is the list version of the `get_entity_param_dict` function.

    Arguments:
        sample: Sample
            A document and annotation sample.

    Return: list[EntityParams]
        A list of the sample's entity annotation parameters.
    """
    return [get_entity_param_dict(e, sample.fname) for e in sample.a1]


def get_relation_param_dict(relation: str, filename: str) -> RelationParams:
    """'Get the relation line parameters.

    Given a relation string with with the format

        ID\tREL_TYPE E1_TYPE:E1_ID E2_TYPE:E2_ID'

    Create a dictionary with the following entity properties:

        * fname:    str
        * id:       str
        * type:     str
        * e1_id:    str
        * e2_id:    str

    For example, given the relation line from the document 'BB-rel-11989773'

        R1\tLives_In Microorganism:T3 Location:T5'

    create a dictionary in with the structure

        {
            'fname':    'BB-rel-11989773'
            'id':       'R1',
            'type':     'Lives_In',
            'e1_id':    'T3',
            'e2_id':    'T5'
        }

    Note that this function ignores the `Equiv` relation annotations.

    Arguments:
        relation: str
            A single relation annotation line.

        filename: str
            The name of the document (without the extension) where this relation
            is from.

    Return: RelationParams
        A dictionary containing the parameters from the relation annotation
        line.
    """
    if not is_valid_relation(relation):
        return None

    [rel_id, data] = relation.split('\t')
    [rel_type, e1, e2] = data.split(' ')
    e1_id = e1.split(':')[1]
    e2_id = e2.split(':')[1]

    return {
        'fname': filename,
        'id': rel_id,
        'type': rel_type,
        'e1_id': e1_id,
        'e2_id': e2_id
    }


def get_sample_relation_param_dicts(sample: Sample) -> list[RelationParams]:
    """Get the sample's relation parameter dictionaries.

    This is the list version of the `get_relation_param_dict` function.

    Arguments:
        sample: Sample
            A document and annotation sample.

    Return: list[RelationParams]
        A list of the sample's relation annotation parameters.
    """
    param_dicts = list()

    for rel in sample.a2:
        param_dict = get_relation_param_dict(rel, sample.fname)
        if param_dict is not None:
            param_dicts.append(param_dict)

    return param_dicts


def insert_document_features(
        document: Document, feature_graph: nx.DiGraph) -> NoReturn:
    """Insert a document node into the features graph.

    Arguments:
        document: Document
            A stanza-annotated document.

        feature_graph: nx.DiGraph
            A graph of features to populate.

    Return: nothing
    """
    node_key = document.fname
    node_attrs = get_document_node_attrs(document)
    feature_graph.add_node(
        node_key, **node_attrs, node_prop=GraphNodeType.DOCUMENT.value)


def insert_sentence_features(
        sentence: Sentence, feature_graph: nx.DiGraph) -> NoReturn:
    """Insert a sentence node into the features graph.

    Arguments:
        sentence: Sentence
            A stanza-annotated sentence.

        feature_graph: nx.DiGraph
            A graph of features to populate.
    """
    node_key = f'{sentence.doc.fname}:{sentence.idx}'
    node_attrs = get_sentence_node_attrs(sentence)

    feature_graph.add_node(
        node_key, **node_attrs, node_prop=GraphNodeType.SENTENCE.value)

    doc_node_key = sentence.doc.fname
    feature_graph.add_edge(
        doc_node_key, node_key, edge_prop=GraphEdgeType.DOC_TO_SENT.value)


def insert_word_features(word: Word, feature_graph: nx.DiGraph) -> NoReturn:
    """Insert a token node into the features graph.

    Given a `Word`, this fuction will add a `Token` node and its edges into the
    features graph. Although the stanza implementation differentiates between
    `Token` and `Word` objects, for the purpose of simplicity, the feature graph
    will only use the term `Token` as if we were talking about stanza's `Word`
    object. Note that this assumption could be problematic with languages that
    have multi-word tokens. For more information:

        https://stanfordnlp.github.io/stanza/data_objects.html#word

    Arguments:
        word: Word
            A stanza-annotated word.

        feature_graph: nx.DiGraph
            A graph of features to populate.
    """
    node_key = f'{word.sent.doc.fname}:{word.end_char}'
    node_attrs = get_word_node_attrs(word)

    feature_graph.add_node(
        node_key, **node_attrs, node_prop=GraphNodeType.TOKEN.value)

    sent_node_key = f'{word.sent.doc.fname}:{word.sent.idx}'
    feature_graph.add_edge(
        sent_node_key, node_key, edge_prop=GraphEdgeType.SENT_TO_TOKEN.value)


def insert_deprel_features(
        word: Word, head: Word, feature_graph: nx.DiGraph) -> NoReturn:
    """Insert dependency relation edge into the features graph.

    Arguments:
        word: Word
            A stanza-annotated word

        head: Word
            The syntactic head of the `word` argument.
    """
    word_key = f'{word.sent.doc.fname}:{word.end_char}'
    head_key = f'{head.sent.doc.fname}:{head.end_char}'

    feature_graph.add_edge(
        word_key,
        head_key,
        deprel=word.deprel,
        edge_prop=GraphEdgeType.DEPREL.value
    )


def insert_annotated_document_features(
        documents: Documents, feature_graph: nx.DiGraph) -> NoReturn:
    """Populate the feature graph with document, sentence and word features.

    For each document in the list, add its features, as well as its respective
    sentences and words features.

    Arguments:
        documents: Documents
            A list of stanza-annotated documents.

        feature_graph: nx.DiGraph
            A graph of features to populate.
    """
    for doc in documents:
        insert_document_features(doc, feature_graph)

        for sent in doc.sentences:
            insert_sentence_features(sent, feature_graph)

            for word in sent.words:
                insert_word_features(word, feature_graph)

                if (head_id := word.head) != 0:
                    head_word = sent.words[head_id-1]
                    insert_word_features(head_word, feature_graph)
                    insert_deprel_features(word, head_word, feature_graph)


def insert_entity_features(
        entity: EntityParams, feature_graph: nx.DiGraph) -> NoReturn:
    """Insert the entity node into the features graph. 

    Insert into the graph a node and an edge connecting the entity to the
    rightmost token of its sequence of tokens. For example, given the entity
    annotation

        'T9\tHabitat 306 324\tmonoclonal B cells'

    Add a node to the graph with an edge connected to the token `cells`, out of
    the sequence [`monoclonal`, `B`, `cells`].

    Arguments:
        entity: EntityParams
            A dict containing the entity's parameters.

        feature_graph: nx.DiGraph
            A graph of features to populate.
    """
    if entity['type'] in {'Title', 'Paragraph'}:    # Ignore these entities
        return

    node_key = f'{entity["fname"]}:{entity["id"]}'
    node_attrs = get_entity_node_attrs(entity)
    feature_graph.add_node(
        node_key, **node_attrs, node_prop=GraphNodeType.ENTITY.value)

    rightmost_boundary = entity['boundaries'][-1]
    end_char = rightmost_boundary['end']
    word_key = f'{entity["fname"]}:{end_char}'
    feature_graph.add_edge(
        node_key, word_key, edge_prop=GraphEdgeType.ENTITY.value)


def insert_relation_features(
        relation: RelationParams, feature_graph: nx.DiGraph) -> NoReturn:
    """Insert the relation node into the features graph. 

    Insert into the graph a node and edges representing the relation and its two
    entities. For example, given the relation annotation

        'R1\tLives_In Microorganism:T19 Location:T15'


    add anode to the graph with edges connected to the nodes of entities `T19`
    and `T15`.

    Arguments:
        relation: RelationParams
            A dict containing the relation's parameters.

        feature_graph: nx.DiGraph
            A graph of features to populate.
    """
    node_key = f'{relation["fname"]}:{relation["id"]}'
    node_attrs = get_relation_node_attrs(relation)
    feature_graph.add_node(
        node_key, **node_attrs, node_prop=GraphNodeType.RELATION.value)

    entity1_key = f'{relation["fname"]}:{relation["e1_id"]}'
    feature_graph.add_edge(
        node_key, entity1_key, edge_prop=GraphEdgeType.REL_TO_E1.value)

    entity2_key = f'{relation["fname"]}:{relation["e2_id"]}'
    feature_graph.add_edge(
        node_key, entity2_key, edge_prop=GraphEdgeType.REL_TO_E2.value)


def insert_entity_relation_features(
        samples: Samples, feature_graph: nx.DiGraph) -> NoReturn:
    """Insert (inplace) the entity relation features into a graph.

    Populate the feature graph with entity relation features, extracted from the
    list of annotated samples. The entity and relation edges are as follows:

        Relation --entity_1--> Entity
        Relation --entity_2--> Entity

    Where the nodes have the following properties:
    Relation
        * node_prop: 'RELATION'
        * id: relation['id']
        * type: relation['type']

    Entity
        * node_prop: 'ENTITY'
        * id: entity['id']
        * type: entity['type']
        * boundaries: entity['boundaries']
        * text: entity['text']
        * is_discontinuous: entity['is_discontinuous']

    Arguments:
        samples: Samples
            A list of text and annotation samples.

        feature_graph: nx.DiGraph
            A graph of features to populate.
    """
    for sample in samples:
        for entity_dict in get_sample_entity_param_dicts(sample):
            insert_entity_features(entity_dict, feature_graph)

        for relation_dict in get_sample_relation_param_dicts(sample):
            insert_relation_features(relation_dict, feature_graph)
