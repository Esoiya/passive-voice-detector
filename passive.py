#! /usr/bin/env python3
"""
"""

import os
import sys

import nltk

from tagger import POSTagger


TAGGER = None


if __name__ == "__main__":

    TAGGER = POSTagger().get()
    
    print(f"Tagger type: {type(TAGGER)}")
    print(f"Tagger: {TAGGER}")
    