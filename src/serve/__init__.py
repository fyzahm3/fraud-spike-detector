"""Live model serving for the hosted site.

Real XGBoost inference in the web process — the trained booster loaded once and
asked for a prediction per request. See src/serve/scorer.py.
"""
