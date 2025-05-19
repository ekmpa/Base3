# Base3: Simple Models for Robust Dynamic Link Prediction

**Base3** is a lightweight, training-free framework for dynamic link prediction on temporal graphs. It supplements the strong recurrence-based EdgeBank baseline with **inductive capabilities**, combining three complementary, non-learnable signals:

- **Edge recurrence** via [EdgeBank](https://github.com/fpour/DGB) 
- **Node popularity** via [PopTrack](https://github.com/temporal-graphs-negative-sampling/TGB/tree/main)  
- **Temporal co-occurrence** via our proposed module, **t-CoMem**

Base3 fuses these signals through a modular interpolation strategy and achieves **state-of-the-art performance** on multiple datasets from the [Temporal Graph Benchmark (TGB)](https://tgb.complexdatalab.com), even outperforming complex deep learning models in challenging settings like **inductive** and **historical** negative sampling.

## 🔍 Highlights
- No training or backprop required
- Strong generalization to unseen nodes (inductive setting)
- Modular, interpretable design
- Outperforms deep models on several TGB datasets


Built on insights from EdgeBank and PopTrack, Base3 shows that **simple models can be competitive—even superior—in real-world temporal graph learning.** Importantly, **Base3 stays strong in inductive and historical sampling settings, with the potential to act as an unprecendtly strong baseline.**
