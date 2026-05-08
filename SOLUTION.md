# SMILES-2026 Hallucination Detection — Solution Report

## Reproducibility

### Option 1 — Google Colab (recommended)

1. Open `solution.ipynb` in Google Colab.
2. **Runtime → Change runtime type → T4 GPU**.
3. Run all cells top-to-bottom. Cell 1 clones the upstream repo and installs deps; cells 2–4 overwrite the upstream placeholder files (`aggregation.py`, `probe.py`, `splitting.py`) with this submission's implementations via `%%writefile`.
4. Feature extraction (~25 min on T4) is cached to `/content/X_train.npy` and `/content/X_test.npy`, so re-runs after a disconnect are instant.
5. The final cell auto-downloads `predictions.csv` and `results.json`.

### Option 2 — Local

```bash
git clone <this repo> && cd SMILES-2026-Hallucination-Detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python solution.py
```

`solution.py` is the official entrypoint and produces the same `predictions.csv` as the notebook (both consume the same three modified modules).

### Environment

- Python 3.10+
- PyTorch ≥ 2.0 with CUDA, transformers ≥ 4.40, scikit-learn ≥ 1.3
- Qwen2.5-0.5B (auto-downloaded on first run, ~1 GB)
- Random seed: 42 (set in `splitting.py` and `probe.py`)

---

## Final solution

### What was modified

Three student files were rewritten; fixed infrastructure (`model.py`, `evaluate.py`) was left untouched. `solution.py` was changed only to flip `USE_GEOMETRIC = True` (a constant the skeleton intends students to toggle).

#### `aggregation.py` — multi-layer last-token concat + geometric features

Default skeleton uses last token of the **final** layer only (896 dims). Submission uses:

- **Layer selection:** last real (non-padding) token from layers at depth `{50%, 75%, 100%}` (i.e. layers ~12, ~18, ~24 of Qwen2.5-0.5B's 24 transformer blocks). Concatenated → **3 × 896 = 2688 dims**.
- **Geometric features (50 dims):**
  - Per-layer L2 norm of the last-token vector across all 25 hidden states (25 dims).
  - Inter-layer cosine similarity between consecutive layers' last-token vectors — a measure of representation drift (24 dims).
  - log(1 + sequence length) (1 dim).
- **Final feature vector: 2738 dims.**

#### `probe.py` — regularized MLP with early stopping

Default skeleton is `[in → 256 → 1]`, 200 epochs, no regularization, hard-coded CPU. Submission uses:

- **Architecture:** `[in → 512 → ReLU → Dropout(0.3) → 128 → ReLU → Dropout(0.3) → 1]`.
- **Optimizer:** Adam, `lr=1e-3`, `weight_decay=1e-4`.
- **Loss:** `BCEWithLogitsLoss(pos_weight=neg/pos)` (kept from skeleton).
- **Early stopping:** carves an internal 10% val slice from the training data (stratified, seed-42), patience 20 on internal val loss, max 300 epochs. The best-val-loss state dict is restored before returning.
- **Threshold tuning:** `fit_hyperparameters` sweeps unique probabilities + a 101-point grid on the external val split, keeps the threshold that maximizes F1 (kept from skeleton).
- **Device:** auto-detects CUDA / MPS / CPU.

#### `splitting.py` — stratified 5-fold

Default skeleton is a single 70/15/15 stratified split (one fold). Submission uses:

- **Stratified 5-fold** on `y` (seed 42).
- Per fold: the held-out fold is `idx_test`; the remaining 80% is split 80/20 (stratified) into `idx_train` and `idx_val`.
- `evaluate.run_evaluation` already iterates folds and averages metrics, so this drops in cleanly.

### Why these choices

- **Mid-to-late layers:** prior work on hallucination probing (SAPLMA, INSIDE, TruthX) consistently shows the truthfulness/hallucination signal lives in mid-to-late layers, not the final layer alone. Concatenating three depths gives the probe richer signal at modest cost (3× parameter count in the input layer of the MLP, no extra forward passes).
- **Geometric features:** per-layer activation norms and inter-layer cosine drift are well-established uncertainty proxies — they capture *how* the representation evolves through the model, not just where it ends up. Cheap to compute (no extra forward passes).
- **Regularization + early stopping:** with ~13k samples and ~2700-dim features, a vanilla MLP trained for a fixed 200 epochs overfits. Dropout + weight decay + early-stopping the best-val state controls this.
- **5-fold:** gives a more stable test-metric estimate than a single split, and (combined with refitting the final probe on all train+val) uses the labeled data more efficiently.

### What contributed most

The biggest single lever is **multi-layer aggregation**: switching from final-layer-only to a 3-layer concat is the main source of feature richness. The geometric features and probe regularization are smaller individual gains stacked on top.

---

## Experiments and design decisions not adopted

These were considered during design but not adopted in the final submission:

- **Mean-pool over response tokens (whole-sequence pool).** Rejected as the primary aggregation: causal models concentrate "what was just said / understood" into the last token's representation, and pooling tends to dilute that signal with prompt context. A useful future ablation.
- **Token-attention pooling with a learned attention block.** Rejected for risk: more parameters to fit on a small dataset, more hyperparameters to tune, marginal expected gain over the simpler concat.
- **Linear probe (logistic regression) instead of MLP.** Rejected because the geometric features interact non-linearly with the dense layer activations; the MLP can model that, a linear probe cannot. A linear probe is a sensible sanity-check baseline if the MLP underperforms.
- **PCA / dimensionality reduction on the 2738-dim feature vector before the MLP.** Rejected for the first submission because the dataset is small enough (~13k) and the MLP is small enough (~1.4M params) that the input dimension is not the bottleneck; the regularization already addresses the variance.
- **Group-aware splits** (e.g., grouping by prompt template). Not adopted because no obvious group key exists in the schema (`prompt`, `response`, `label`).

The natural next steps if iterating: ensemble the 5 fold probes for the final prediction (average their probabilities) instead of refitting on train+val — typically a small but reliable gain.
