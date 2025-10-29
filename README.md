# ReviewGen‑BENCH

**Can LLM‑Generated Reviews Substitute Human Reviews in Text‑Aware Recommender Systems?**
*A preliminary study on Amazon categories*

> Status: research code and artifacts for a preliminary study. Results may evolve.

---

## TL;DR

Many items have few or no human reviews, which limits text‑aware recommenders. ReviewGen‑BENCH is a metadata‑conditioned pipeline that synthesizes item reviews with an LLM, filters them with a multi‑objective critic (fluency, toxicity, semantic relevance), and integrates the filtered text into multiple recommendation models. Across four Amazon categories and seven architectures, synthetic reviews are often competitive with human text for ranking accuracy and can improve cold‑start coverage and novelty at the same time.

---

## Highlights

* **End‑to‑end pipeline** for generating, filtering, and integrating synthetic item reviews
* **Unified evaluation** across 4 datasets, 7 models, and multiple augmentation settings
* **Quality critics** for fluency, toxicity, and relevance
* **Cold‑start friendly**: synthetic text can expand tail coverage without hurting novelty

---

## Contribution summary

* **C1. Metadata‑conditioned generative pipeline.** Convert item metadata into diverse prompts, generate candidate reviews with an LLM, and filter via a multi‑objective critic.
* **C2. Unified evaluation protocol.** Compare human, synthetic, and hybrid text across datasets, models, and prompt styles/lengths.
* **C3. Critic ablations and bias analysis.** Quantify how critic thresholds and architectures influence accuracy, coverage, and novelty.
* **C4. Open artifacts.** Plan to release synthetic corpora, critic scores, configs, and scripts for exact reproducibility.

---

## Datasets

Four Amazon product categories from the public Amazon Reviews data:

* **All Beauty**
* **Amazon Fashion**
* **Health and Personal Care**
* **Magazine Subscriptions**

After 4‑core filtering, the datasets span ~400–1,400 users, ~185–1,468 items, and remain sparse enough to stress cold‑start behavior.

> Place raw data under `data/raw/<dataset_name>/` and processed splits under `data/processed/<dataset_name>/`.

---

## Models

The benchmark covers representative families:

* **Interaction only**: MF, BPR, VAECF
* **Text feature encoders**: ConvMF, CDL
* **Text regularization**: CTR, HFT

A unified interface allows consistent training and evaluation with or without text.

---

## Method overview

1. **Prompt diversification**
   Two styles (dense narrative, aspect list) × three lengths (short ≈50 tokens, medium ≈120, long ≈250) → six prompt variants per item.
2. **LLM generation**
   Generate synthetic reviews from item metadata for items with scarce or missing reviews.
3. **Multi‑objective critic**
   Filter candidates by: low perplexity (fluency), low toxicity, high cosine similarity to metadata embeddings (relevance).
4. **Integration**
   Train recommenders with one of: Editorial only (metadata), Human only, Human+Editorial, or **Synthesized+Editorial**.

A schematic appears below. Put an image file at `assets/reviewgen-bench-architecture.png` and update the path if needed.

![ReviewGen‑BENCH architecture](assets/reviewgen-bench-architecture.png)

---

## Key results at a glance

* Synthetic reviews are **competitive with human text** on most categories for Recall@10.
* On Fashion, synthetic dense‑long text improved **cold‑item coverage** and **novelty** together relative to metadata baselines.
* Gains vary by **dataset**, **prompt style/length**, and **model family**; dense narrative prompts tend to work best.

> Exact numbers depend on data preprocessing and random seeds. See the `reports/` folder for tables and plots.

---

## Repository structure

```
.
├── assets/                          # Figures for README and docs
├── configs/                         # YAML configs for datasets, models, prompts
├── data/
│   ├── raw/                         # Original data dumps
│   └── processed/                   # Tokenized text, splits, vocab, caches
├── scripts/
│   ├── prepare_data.py              # Download, filter, and split data
│   ├── generate_reviews.py          # LLM synthesis with prompt variants
│   ├── run_critic.py                # Fluency, toxicity, relevance filtering
│   ├── train.py                     # Train a given model with a config
│   ├── evaluate.py                  # Compute Recall@K, Cold‑Rate@K, Novelty
│   └── aggregate_results.py         # Tables and plots for reports
├── src/
│   ├── dataio/                      # Loaders, preprocessors, tokenizers
│   ├── models/                      # MF, BPR, VAECF, CTR, HFT, ConvMF, CDL
│   ├── prompts/                     # Prompt templates and LLM utils
│   ├── critics/                     # Perplexity, toxicity, relevance modules
│   └── utils/                       # Training loops, metrics, logging
├── reports/                         # Generated tables and figures
├── requirements.txt
└── README.md
```

> File names and folders are suggestions. Align them with your actual implementation.

---

## Installation

```bash
# Clone
git clone https://github.com/<user>/<repo>.git
cd <repo>

# Python environment (example with venv)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Suggested `requirements.txt`

```
numpy
scipy
pandas
scikit-learn
torch
transformers
sentence-transformers
detoxify
matplotlib
pyyaml
rich
```

> Add your preferred LLM client if remote generation is used.

---

## Quick start

1. **Prepare data**

   ```bash
   python scripts/prepare_data.py \
     --dataset fashion \
     --kcore 4
   ```
2. **Generate synthetic reviews**

   ```bash
   python scripts/generate_reviews.py \
     --dataset fashion \
     --prompts dense,aspect \
     --lengths S,M,L \
     --out data/processed/fashion/synth_reviews.jsonl
   ```
3. **Run the critic filter**

   ```bash
   python scripts/run_critic.py \
     --in data/processed/fashion/synth_reviews.jsonl \
     --ppl_thresh 150 --tox_thresh 0.20 --rel_thresh 0.15 \
     --out data/processed/fashion/synth_reviews.filtered.jsonl
   ```
4. **Train a model**

   ```bash
   # CTR with synthesized+editorial text
   python scripts/train.py \
     --model CTR \
     --dataset fashion \
     --text_mode synthetic_dense_L \
     --config configs/ctr_fashion.yaml
   ```
5. **Evaluate**

   ```bash
   python scripts/evaluate.py \
     --runs runs/ctr_fashion_synth_dense_L/ \
     --metrics recall@10,coldrate@10,novelty
   ```

---

## Configuration

Example `configs/ctr_fashion.yaml`:

```yaml
dataset: fashion
text_mode: synthetic_dense_L
splits: {train: 0.8, val: 0.1, test: 0.1}
model:
  name: CTR
  factors: 128
  l2: 1e-4
  beta_text: 1.0
train:
  batch_size: 2048
  epochs: 100
  lr: 5e-3
  seed: 42
critic:
  ppl_thresh: 150
  tox_thresh: 0.20
  rel_thresh: 0.15
```

---

## Metrics

* **Recall@K** for ranking accuracy
* **Cold‑Rate@K**: fraction of recommended items with few historical interactions
* **Novelty**: average information content based on item popularity

---

## Reproducing reported tables and figures

* Place generated CSV files under `reports/metrics/`
* Run `python scripts/aggregate_results.py` to build summary tables
* Figures such as cold‑start versus novelty can be regenerated from the aggregated CSV

Add your architecture and cold‑start figures to `assets/` and reference them in the README and paper:

![Cold‑start vs novelty](assets/coldstart_novelty.png)

---

## Theory of change and risks

Synthetic reviews can expand representation of tail items where organic text is missing or noisy. The critic reduces risks by filtering low‑quality or toxic generations and by enforcing semantic alignment with metadata. Remaining risks include bias, domain shift, and hallucinated attributes. Use transparent thresholds, audits, and careful human review for sensitive categories.

---

## Roadmap

* More prompt variants and reward‑model scoring
* Hybrid human+synthetic strategies
* User studies and fairness audits
* Larger datasets and temporal splits

---

## How to cite

If you use ReviewGen‑BENCH, please cite:

```
@inproceedings{reviewgenbench2025,
  title   = {Can LLM-Generated Reviews Substitute Human Reviews in Text-Aware Recommender Systems? A Preliminary Study on Amazon},
  author  = {Anonymous Authors},
  year    = {2025},
  booktitle = {Under review},
  note    = {Anonymous code and data repository}
}
```

Update author and venue fields when the paper is finalized.

---

## License

Choose a license for this repository (for example MIT) and update `LICENSE` accordingly.

---

## Contact

Questions and feedback: Please open an issue on this repository.
