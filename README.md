# edge-adapt
Adaptive link prediction 

- EdgeBank: https://github.com/fpour/DGB
    - Data: https://zenodo.org/records/7213796#.Y1cO6y8r30o
- PopTrack: https://github.com/temporal-graphs-negative-sampling/TGB/tree/main

Mixing: 
- (start with) simple interpolation / rule-based mixing
- log-reg based weighting
- bayesian weighting


Add THAS for multi-hub reasoning (the main problem right now is that even if this is guaranteed to outdo each individually, lacks multi-hub & temporality). 
- Might experiment with other simple heuristics suhc as recent interaction frequency.

### Key Components
Data Loader 
-> Interaction History
-> Feature Extractors (THAS, EdgeBank, PopTrack, Recency & Common Neighbours?)
-> Interpolation model (experiment with weighting)
-> Link Scores
-> Evaluation Metrics 
