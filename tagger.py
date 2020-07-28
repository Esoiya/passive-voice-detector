#!/usr/bin/env python3

import os
import sys

import nltk
from nltk.corpus import brown
from pickle import dump, load


from logger import logger


class POSTagger:

    """
    POS tagger using the Brown Corpus
    """
    
    def __create(self):
        """
        Trains tagger using Brown Corpus
        """
        logger.info("Building Tagger")
        
        train_sents = brown.tagged_sents()
        
        # NLTK book chapter
        
        tag0 = nltk.RegexpTagger(
            [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'), # cardinal numbers
             (r'(The|the|A|a|An|an)$', 'AT'), # articles
             (r'.*able$', 'JJ'),              # adjectives
             (r'.*ness$', 'NN'),              # nouns formed from adjectives
             (r'.*ly$', 'RB'),                # adverbs
             (r'.*s$', 'NNS'),                # plural nouns
             (r'.*ing$', 'VBG'),              # gerunds
             (r'.*ed$', 'VBD'),               # past tense verbs
             (r'.*', 'NN')                    # nouns (default)
            ])
            
        logger.debug("Tag 0 completed")
        
        tag1 = nltk.UnigramTagger(train_sents, backoff=tag0)
        logger.debug("Tag 1 completed")
        
        tag2 = nltk.BigramTagger(train_sents, backoff=tag1)
        logger.debug("Tag 2 completed")
        
        tag3 = nltk.TrigramTagger(train_sents, backoff=tag2)
        logger.info("Built Tagger")
        
        return tag3
        
    
    def load_tagger(self):
    
        input = open("tagger.pkl", "rb")
            
        tagger = load(input)
        
        input.close()
            
        logger.info("The tagger has been loaded.")
        
        
        return tagger
        
    
    def get(self):
        if os.path.exists("tagger.pkl"):
            return self.load_tagger()
        
        with open("tagger.pkl", "wb") as tag:
        
            new_tagger = self.__create()
            
            dump(new_tagger, tag, -1)
            
        logger.info(f"Tagger : {new_tagger}")
        return new_tagger
