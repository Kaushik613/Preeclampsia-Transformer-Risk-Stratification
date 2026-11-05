"""
Standard model interface for FTTransformer.
Exposes FTTransformer and tokenizers.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
_model_mod = importlib.import_module("src.04_model")

FTTransformer = _model_mod.FTTransformer
NumericalFeatureTokenizer = _model_mod.NumericalFeatureTokenizer
CategoricalFeatureTokenizer = _model_mod.CategoricalFeatureTokenizer
