#! /usr/bin/env python3

import os
import sys

import nltk

from itertools import dropwhile

from logger import logger
from tagger import POSTagger


TAGGER = None


class PassiveDetector:

    @staticmethod
    def tag(sent):
        """
        Take a sentence a return a list of (word, tag) tuples
        """
        tokens = nltk.word_tokenize(sent)
        return TAGGER.tag(tokens)
        
    @staticmethod
    def checker(tags):
        """
        Returns true if the sentence is a passive sentence
        
        - if there is a "be" verb followed by another verb not including gerunds
        - ignoring gerunds (*ing)
        """
        
        afterBe = list(
            dropwhile( lambda tag : not tag.startswith("BE"), tags
            )
        )
        
        logger.info("After 'to be': ")
        
        print(afterBe)
        
        nongerunds = lambda tag : tag.startswith("V") and not tag.startswith("VBG")
        
        filtered = filter(nongerunds, afterBe)
        
        checked_tags = any(filtered)
        
        print(checked_tags)
        
        return checked_tags
        
        

def find_passives(sentence):
    """
    Given a sentence, tag the sentence and print if it's passive
    """
    
    tagged_sent = PassiveDetector.tag(sentence)
    
    tags = map(lambda tup: tup[1], tagged_sent)
    
    if PassiveDetector.checker(tags):
        print("Passive located")
        print(f"Passive: {sentence}")
    else:
        print(f"Not passive: {sentence}")
        

def extract_text(arg, punkt):
    """
    Extract sentences from argument text and find the passive sentences
    """

    with open(arg) as f:
        logger.info("Reading Text file")
        text = f.read()
        sentences = punkt.tokenize(text)
        
        logger.info(f"{len(sentences)} sentences detected")
        
        for sent in sentences:
            find_passives(sent)
            print("-"*60)


if __name__ == "__main__":

    TAGGER = POSTagger().get()
    
    if len(sys.argv) > 1:
    
        # pre-trained version of PunktSentenceTokenizer
        punkt = nltk.tokenize.punkt.PunktSentenceTokenizer()
        
        for arg in sys.argv[1:]:
            extract_text(arg, punkt)
    else:
        print("No sentences")
