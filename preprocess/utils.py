#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provide utility functions and type hints for the `parsing` module.

The contents of this module shouldn't be imported directly, instead, you should
use the functions from the `preprocessing.parsing` module (the parsing.py file)
instead.
"""
from collections import namedtuple
from enum import Enum
import os
import re

from typing import NoReturn
from typing import Optional
from typing import Union

import networkx as nx
from stanza.models.common.doc import Document
from stanza.models.common.doc import Sentence
from stanza.models.common.doc import Word


# Data structures and Type Hints, hopefully to help you understanding the
# mapping of inputs to outputs
Sample = namedtuple('Sample', ['fname', 'txt', 'a1', 'a2'])
Samples = list[Sample]
AnnotationA1 = namedtuple('AnnotationA1', ['id', 'type', 'idxs', 'text'])
AnnotationsA1 = list[AnnotationA1]
AnnotationA2 = namedtuple('AnnotationA2', ['id', 'type', 'e1_id', 'e2_id'])
AnnotationsA2 = Optional[list[AnnotationA2]]
Annotations = Union[AnnotationsA1, AnnotationsA2]
Documents = list[Document]
Entity = namedtuple('Entity',
                    ['fname', 'id', 'type', 'text', 'sent_idx', 'words',
                     'is_discontinuous'])
Boundary = namedtuple('Boundary', ['start', 'end'])
Boundaries = list[Boundary]
Relation = namedtuple('Relation', ['fname', 'type', 'id', 'e1', 'e2'])

## Graph nodes
DocumentNodeAttrs = dict[str, str]
SentenceNodeAttrs = dict[str, Union[str, int]]
WordNodeAttrs = dict[str, Union[str, int]]
EntityNodeAttrs = dict[str, str]
RelationNodeAttrs = dict[str, str]

## Entity and relation property
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


class GraphEdgeColor(Enum):
    DOC_SENT = '#B80C09'
    SENT_ROOT = '#0B4F6C'
    SENT_TOKEN = '#21D19F'
    TOKEN_HEAD = '#474973'
    ENT_TOKEN = '#01BAEF'
    REL_ENT = '#FCAB10'

class GraphNodeColor(Enum):
    DOCUMENT = '#DC122A'
    SENTENCE = '#6666FF'
    TOKEN = '#66FF66'
    HEAD = '#7323cf'
    ENTITY = '#51E5FF'
    RELATION = '#474973'


def get_document_node_key(fname: str) -> str:
    """Get a key for a document's node in the feature graph"""
    return fname


def get_sentence_node_key(fname: str, sent_idx: int) -> str:
    """Get a key for a sentence's node in the feature graph"""
    return f'{fname}:SENT-{sent_idx}'


def get_word_node_key(word: Word) -> str:
    """Get a key for a word's node in the feature graph"""
    return f'{word.sent.doc.fname}:WORD-{word.end_char}'


def get_entity_node_key(entity: Entity) -> str:
    """Get a key for an entity's node in the feature graph"""
    return f'{entity.fname}:ENTITY-{entity.id}'


def get_relation_node_key(relation: Relation) -> str:
    """Get a key for a relation's node in the feature graph"""
    return f'{relation.fname}:RELATION-{relation.id}'


def is_valid_entity(line: str) -> bool:
    """Check if the line is a valid entity annotation."""
    regex = r'^T\d+\t\w+ \d+ \d+(;\d+ \d+)*\t.+$'
    return re.search(regex, line) is not None


def is_valid_relation(line: str) -> bool:
    """Check if the line is a valid binary relation annotation."""
    regex = r'^R\d+\t\w+ \w+:T\d+ \w+:T\d+$'
    return re.fullmatch(regex, line) is not None


# TODO: Write a propper docstring for the entity-word overlap
def check_boundaries_overlap(
    x_start: int, x_end: int, y_start: int, y_end: int) -> bool:
    """Check if the boundaries of both elements overlap.

    Given the starting and end characters of two text spans, check if the X's
    span overlaps with Y's span. This function is used to decide if a given
    token belongs to an entity's text span. To handle issues with unexpected
    tokenization, the definition of "overlap" used here.

    if (
        token.start <= entity.start < entity.end and (
            entity.start < token.end <= entity.end or
            entity.start < entity.end < token.end
        )
    ) elif (
        entity.start < token.start < entity.end and (
            entity.start < token.end <= entity.end or
            entity.start < entity.end < token.end
        )
    )

    Case 1:
    X's span       |----------|
    Y's span 1  |----------|        or
    Y's span 2  |-------------|     or
    Y's span 3  |----------------|

    Case 2
    X's span       |----------|
    Y's span 1     |-------|           or
    Y's span 2     |----------|        or
    Y's span 3     |-------------|

    Case 3
    X's span       |----------|
    Y's span 1        |----|           or
    Y's span 2        |-------|        or
    Y's span 3        |----------|

    Arguments:
        x_start: int
            The position of X's starting character.

        x_start: int
            The position of X's starting character.

        x_end: int
            The position of X's starting character.

        y_start: int
            The position of X's starting character.

        y_end: int
            The position of X's starting character.

    Return: True | False
        True if both spans overlap, False otherwise.
    """
    if ((y_start <= x_start < x_end) and
        ((x_start < y_end <= x_end) or (x_start < x_end < y_end))):
        return True
    elif ((x_start < y_start < x_end) and
          ((x_start < y_end <= x_end) or (x_start < x_end < y_end))):
        return True
    else:
        return False


def __replace_nbsps(rawtext: str) -> str:
    """Replace nbsps with whitespaces"""
    return rawtext.replace('\xa0', ' ')

def __read_annotations(filepath: str) -> Optional[Annotations]:
    """Get the list of lines from an annotation file (either A1 or A2)."""
    try:
        with open(filepath) as fd:
            text = fd.read().strip()
            text = __replace_nbsps(text)
            annotations = text.splitlines()
    except FileNotFoundError:
        annotations = None

    return annotations


def read_rawtext(filepath: str) -> str:
    """Read the contents of either a document or annotation file."""
    text = 'load_dataset_files:read_rawtext:DEFAULT-TEXT'
    with open(filepath) as fd:
        text = fd.read().strip()
        text = __replace_nbsps(text)
        text = text.replace(os.linesep, ' ')

    return text


def __get_annotation_a1_from(line: str) -> AnnotationA1:
    """Create an annotation namedtuple from an annotation line.

    Build an A1 annotation from a single line from the `.a1` annotation file.

    Arguments:
        line: str
            A single line from an (.a1) annotation file.

    Return: AnnotationA1
        A namedtuple with the data from the annotation line.
    """
    [ent_id, data, text] = line.split('\t')
    ent_type = data.split(' ')[0]
    idxs = re.search(r'\d+ \d+(;\d+ \d+)*', data).group()
    idxs = idxs.split(';')
    idxs = [tuple(e.split(' ')) for e in idxs]
    idxs = [Boundary(start=int(s), end=int(e)) for (s, e) in idxs]

    return AnnotationA1(id=ent_id, type=ent_type, idxs=idxs, text=text)


def get_annotations_a1(fpath: str) -> list[AnnotationA1]:
    """Create a list of A1 annotations from the annotation file's lines.

    Given a list of text lines read from an annotation file with the '.a1'
    extension, return a list of annotation namedtuples.

    Arguments:
        fpath: str
            The path to an A1 annotation file (with extension '.a1').

    Return: list[AnnotationA1]
        A list of annotation structures, matching the lines from the a1
        annotations file.
    """
    a1_lines = __read_annotations(fpath)
    annotations = [__get_annotation_a1_from(l) for l in a1_lines]
    return [e for e in annotations if e.type not in {'Title', 'Paragraph'}]


def __get_annotation_a2_from(line: str) -> AnnotationA2:
    """Create an annotation namedtuple from an annotation line.

    Build an A2 annotation from a single line from the `.a2` annotation file.

    Arguments:
        line: str
            A single line from an (.a2) annotation file.

    Return: AnnotationA2
        A namedtuple with the data from the annotation line.
    """
    [rel_id, data] = line.split('\t')
    [rel_type, e1, e2] = data.split(' ')
    e1_id = e1.split(':')[-1]
    e2_id = e2.split(':')[-1]

    return AnnotationA2(id=rel_id, type=rel_type, e1_id=e1_id, e2_id=e2_id)


def get_annotations_a2(fpath: str) -> list[AnnotationA2]:
    """Create a list of A2 annotations from the annotation file's lines.

    Given a list of text lines read from an annotation file with the '.a2'
    extension, return a list of annotation namedtuples.

    Arguments:
        fpath: str
            The path to an A2 annotation file (with extension '.a2').

    Return: list[AnnotationA2]
        A list of annotation structures, matching the lines from the A2
        annotations file.
    """
    a2_lines = __read_annotations(fpath)
    return [__get_annotation_a2_from(l) for l in a2_lines
            if is_valid_relation(l)]


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
        'color': GraphNodeColor.DOCUMENT.value # for pyvis
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
        'color': GraphNodeColor.SENTENCE.value # for pyvis
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
    # Changing the color for the sentence's head token
    if word.head == 0:
        color = GraphNodeColor.HEAD.value
    else:
        color = GraphNodeColor.TOKEN.value

    return {
        'fname': word.sent.doc.fname,
        'start_idx': word.parent.start_char,
        'end_idx': word.parent.end_char,
        'text': word.text,
        'upos': word.upos,
        'lemma': word.lemma,
        'label': word.text,                 # for PyVis
        'color': color                      # for pyvis
    }


def get_entity_node_attrs(entity: Entity) -> EntityNodeAttrs:
    """Create the graph's node attribute for an `Entity`.

    Given an entity annotation, build a property dict for an entity node in
    the feature graph.

    Arguments:
        entity: Entity
            An entity's namedtuple instance.

    Return: EntityNodeAttrs
        A dictionary containing the entity's features to be used by networkx's
        feature graph.
    """
    return {
        'id': entity.id,
        'type': entity.type,
        'text': entity.text,
        'is_discontinuous': entity.is_discontinuous,
        'label': entity.id # for PyVis
    }


def get_relation_node_attrs(relation: Relation) -> RelationNodeAttrs:
    """Create the graph's node attribute for a `Relation`.

    Given a relation annotation, build a property dict for an relation node in
    the feature graph.

    Arguments:
        relation: Relation
            A relation's nameduple instance.

    Return: RelationNodeAttrs
        A dictionary containing the relation's features to be used by
        networkx's feature graph.
    """
    return {
        'id': relation.id,
        'type': relation.type,
        'label': relation.id # for PyVis
    }


def get_entity_param_dict(entity: Entity) -> EntityParams:
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
            'text':             'patients with low-grade MALT lymphoma',
            'is_discontinuous': True
        }

    Arguments:
        entity: Entity
            A single entity namedtuple instance.

    Return: EntityParams
        A dictionary containing the parameters from the entity annotation line.
    """
    fname = entity.words[0].sent.doc.fname
    return {
        'fname': fname,
        'id': entity.id,
        'type': entity.type,
        'text': entity.text,
        'is_discontinuous': entity.is_discontinuous
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
    return [get_entity_param_dict(e) for e in sample.a1]


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
            The name of the document (without the extension) where this
            relation is from.

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
    doc_node_key = get_document_node_key(document.fname)
    doc_node_attrs = get_document_node_attrs(document)
    feature_graph.add_node(doc_node_key,
                           **doc_node_attrs,
                           node_prop=GraphNodeType.DOCUMENT.value)


def insert_sentence_features(
        sentence: Sentence, feature_graph: nx.DiGraph) -> NoReturn:
    """Insert a sentence node into the features graph.

    Arguments:
        sentence: Sentence
            A stanza-annotated sentence.

        feature_graph: nx.DiGraph
            A graph of features to populate.
    """
    fname = sentence.doc.fname
    sent_node_key = get_sentence_node_key(fname, sentence.idx)
    sent_node_attrs = get_sentence_node_attrs(sentence)
    feature_graph.add_node(sent_node_key,
                           **sent_node_attrs,
                           node_prop=GraphNodeType.SENTENCE.value)

    doc_node_key = get_document_node_key(fname)
    feature_graph.add_edge(sent_node_key,
                           doc_node_key,
                           edge_prop=GraphEdgeType.DOC_TO_SENT.value,
                           title=GraphEdgeType.DOC_TO_SENT.value,
                           color=GraphEdgeColor.DOC_SENT.value)


def insert_word_features(word: Word, feature_graph: nx.DiGraph) -> NoReturn:
    """Insert a token node into the features graph.

    Given a `Word`, this fuction will add a `Token` node and its edges into the
    features graph. Although the stanza implementation differentiates between
    `Token` and `Word` objects, for the purpose of simplicity, the feature
    graph will only use the term `Token` as if we were talking about stanza's
    `Word` object. Note that this assumption could be problematic with
    languages that have multi-word tokens. For more information:

        https://stanfordnlp.github.io/stanza/data_objects.html#word

    Arguments:
        word: Word
            A stanza-annotated word.

        feature_graph: nx.DiGraph
            A graph of features to populate.
    """
    fname = word.sent.doc.fname
    word_node_key = get_word_node_key(word)
    word_node_attrs = get_word_node_attrs(word)

    feature_graph.add_node(word_node_key,
                           **word_node_attrs,
                           node_prop=GraphNodeType.TOKEN.value)

    sent_node_key = get_sentence_node_key(fname, word.sent.idx)
    feature_graph.add_edge(word_node_key,
                           sent_node_key,
                           edge_prop=GraphEdgeType.SENT_TO_TOKEN.value,
                           title=GraphEdgeType.SENT_TO_TOKEN.value,
                           color=GraphEdgeColor.SENT_TOKEN.value)


def insert_deprel_features(
        word: Word, head: Word, feature_graph: nx.DiGraph) -> NoReturn:
    """Insert dependency relation edge into the features graph.

    Arguments:
        word: Word
            A stanza-annotated word

        head: Word
            The syntactic head of the `word` argument.
    """
    fname = word.sent.doc.fname
    word_node_key = get_word_node_key(word)
    head_node_key = get_word_node_key(head)
    feature_graph.add_edge(word_node_key,
                           head_node_key,
                           deprel=word.deprel,
                           edge_prop=GraphEdgeType.DEPREL.value,
                           title=word.deprel,
                           color=GraphEdgeColor.TOKEN_HEAD.value)


def insert_annotated_document_features(documents: Documents,
                                       feature_graph: nx.DiGraph,
                                       ignore_pos: list[str]=None,
                                       ignore_deprel: list[str]=None
                                       ) -> NoReturn:
    """Populate the feature graph with document, sentence and word features.

    For each document in the list, add its features, as well as its respective
    sentences and words features.

    Arguments:
        documents: Documents
            A list of stanza-annotated documents.

        feature_graph: nx.DiGraph
            A graph of features to populate.
    """
    ignore_pos = set() if ignore_pos is None else set(ignore_pos)
    ignore_deprel = set() if ignore_deprel is None else set(ignore_deprel)

    for doc in documents:
        insert_document_features(doc, feature_graph)

        for sent in doc.sentences:
            insert_sentence_features(sent, feature_graph)
            for word in sent.words:
                if word.upos in ignore_pos or word.deprel in ignore_deprel:
                    continue

                insert_word_features(word, feature_graph)
                if (head_id := word.head) != 0:
                    head_word = sent.words[head_id-1]
                    insert_word_features(head_word, feature_graph)
                    insert_deprel_features(word, head_word, feature_graph)


def insert_entity_features(
        entity: Entity, feature_graph: nx.DiGraph) -> NoReturn:
    """Insert the entity node into the features graph. 

    Insert into the graph a node and an edge connecting the entity to the
    rightmost token of its sequence of tokens. For example, given the entity
    annotation

        'T9\tHabitat 306 324\tmonoclonal B cells'

    Add a node to the graph with an edge connected to the token `cells`, out of
    the sequence [`monoclonal`, `B`, `cells`].

    Arguments:
        entity: Entity
            An entity namedtuple instance containing its parameters.

        feature_graph: nx.DiGraph
            A graph of features to populate.
    """
    fname = entity.words[0].sent.doc.fname
    entity_node_key = get_entity_node_key(entity)
    entity_node_attrs = get_entity_node_attrs(entity)
    feature_graph.add_node(entity_node_key,
                           **entity_node_attrs,
                           node_prop=GraphNodeType.ENTITY.value,
                           color=GraphNodeColor.ENTITY.value)

    rightmost_word = entity.words[-1]
    word_key = get_word_node_key(rightmost_word)
    feature_graph.add_edge(entity_node_key,
                           word_key,
                           edge_prop=GraphEdgeType.ENTITY.value,
                           title=GraphEdgeType.ENTITY.value,
                           color=GraphEdgeColor.ENT_TOKEN.value)


def insert_relation_features(
        relation: Relation, feature_graph: nx.DiGraph) -> NoReturn:
    """Insert the relation node into the features graph. 

    Insert into the graph a node and edges representing the relation and its
    two entities. For example, given the relation annotation

        'R1\tLives_In Microorganism:T19 Location:T15'


    add anode to the graph with edges connected to the nodes of entities `T19`
    and `T15`.

    Arguments:
        relation: Relation
            A relation's namedtuple instance with its parameters.

        feature_graph: nx.DiGraph
            A graph of features to populate.
    """
    rel_node_key = get_relation_node_key(relation)
    rel_node_attrs = get_relation_node_attrs(relation)
    feature_graph.add_node(rel_node_key,
                           **rel_node_attrs,
                           node_prop=GraphNodeType.RELATION.value,
                           color=GraphNodeColor.RELATION.value)

    entity1_key = get_entity_node_key(relation.e1)
    feature_graph.add_edge(rel_node_key,
                           entity1_key,
                           edge_prop=GraphEdgeType.REL_TO_E1.value,
                           title=GraphEdgeType.REL_TO_E1.value,
                           color=GraphEdgeColor.REL_ENT.value)

    entity2_key = get_entity_node_key(relation.e2)
    feature_graph.add_edge(rel_node_key,
                           entity2_key,
                           edge_prop=GraphEdgeType.REL_TO_E2.value,
                           title=GraphEdgeType.REL_TO_E2.value,
                           color=GraphEdgeColor.REL_ENT.value)


def insert_entity_relation_features(documents: Documents,
                                    feature_graph: nx.DiGraph,
                                    ignore_intra_sent_relations: bool = False
                                    ) -> NoReturn:
    """Insert (inplace) the entity relation features into a graph.

    Populate the feature graph with entity relation features, extracted from
    the list of annotated samples. The entity and relation edges are as
    follows:

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
        documents: util.Documents
            A list of stanza-annotated documents, including entity-relation
            annotations.

        feature_graph: nx.DiGraph
            A graph of features to populate.
    """
    for doc in documents:
        for ent in doc.entities.values():
            insert_entity_features(ent, feature_graph)

        for rel in doc.relations.values():
            insert_relation_features(rel, feature_graph)
