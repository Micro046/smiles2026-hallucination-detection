# SMILES-2026 Hallucination Detection — Solution

## How to reproduce

The easiest way to run this is through `solution.ipynb` on Google Colab with a T4 GPU (Runtime → Change runtime type → GPU).

The notebook is self-contained: the first cell clones the upstream repo and installs dependencies, and the following cells write my modified `aggregation.py`, `probe.py`, and `splitting.py` into the cloned repo before any imports happen, so you don't need to upload anything manually. After that it's just run-all — feature extraction takes around 25 minutes the first time but caches to `/content/X_train.npy` and `/content/X_test.npy`, so a Colab disconnect doesn't mean starting over. The last cell auto-downloads `predictions.csv` and `results.json`.

To run locally instead:

```bash
git clone https://github.com/Micro046/smiles2026-hallucination-detection
cd smiles2026-hallucination-detection
pip install -r requirements.txt
python solution.py
```

`solution.py` imports from the same three modules as the notebook, so both paths produce identical output. Seed is fixed at 42 throughout.

---

## Approach

I only touched the three student files (`aggregation.py`, `probe.py`, `splitting.py`) and flipped the `USE_GEOMETRIC` flag to `True` in `solution.py`. Everything else is the original infrastructure.

### Feature extraction (`aggregation.py`)

The default implementation takes the last token of the final transformer layer (896 dims). My intuition was that the hallucination signal probably doesn't live only in the very last layer — research on probing transformers generally finds that mid-to-late layers carry more factual/truthfulness information than the final one. So I concatenated the last real token's hidden state from three layers spaced at roughly 50%, 75%, and 100% of the model depth (layers ~12, ~18, 24 for Qwen2.5-0.5B's 24 blocks). That gives 3 × 896 = 2688 dimensions.

On top of that I added what the skeleton calls geometric features. These are cheap to compute and don't require any extra forward passes: the L2 norm of the last token at each of the 25 hidden states (25 values), the cosine similarity between consecutive layers' last-token vectors (24 values — this captures how much the representation is changing layer by layer), and a log-scaled sequence length. The final feature vector is 2738 dimensions.

### Probe (`probe.py`)

The skeleton MLP was `[in → 256 → 1]` trained for a fixed 200 epochs with no regularization. With ~13k samples and a 2738-dim input that's a recipe for overfitting, so I made a few changes. The architecture is now `[in → 512 → Dropout(0.3) → 128 → Dropout(0.3) → 1]` with weight decay 1e-4 on the optimizer. I also added early stopping — I hold out 10% of the training data internally, track validation loss, and restore the best checkpoint if it doesn't improve for 20 epochs (up to 300 max). The public `fit()` signature is unchanged, this all happens internally. Threshold tuning on the external val split was already in the skeleton and I kept it as is.

The probe also auto-detects whether CUDA/MPS is available and moves computation there, which matters for running efficiently in Colab.

### Splitting (`splitting.py`)

I replaced the single 70/15/15 split with stratified 5-fold cross-validation. Each fold uses the held-out fifth as the test split, and the remaining 80% is further split 80/20 into train and val (stratified). `evaluate.py` already handles iterating over multiple folds and averaging, so no infrastructure changes were needed. For the final `predictions.csv` I refit on all non-test indices, same as `solution.py` already does.

### What helped most

The multi-layer aggregation made the biggest difference — going from the final layer alone to a 3-layer concat gives the probe much richer signal to work with. The geometric features and regularization are meaningful but smaller gains on top of that.

---

## Things I tried that didn't make it in

**Mean-pooling over all response tokens.** My first instinct was to pool over the whole response rather than just the last token, but in a causal decoder the last token attends to everything before it, so its representation already summarizes the full sequence. Pooling across all tokens mixes in the prompt context and dilutes the response-specific signal.

**Logistic regression instead of the MLP.** I considered a simple linear probe since it generalises better with fewer samples, but the geometric features interact non-linearly with the dense activations and the MLP can capture that while logistic regression can't.

**PCA before the MLP.** I looked at whether compressing 2738 dims down before the MLP would help, but with the dataset size and the regularization already in place the input dimension isn't the bottleneck, so it wasn't worth the extra complexity.

**Attention-pooling with a learned query.** A small attention block over the response tokens would let the model learn which positions are most informative for hallucination detection, which is appealing in principle. I dropped it because it adds parameters on a small dataset, has extra hyperparameters to tune, and the simple last-token approach is already competitive.

**Group-aware splits.** I considered splitting by prompt group to avoid data leakage between similar prompts, but there's no obvious group key in the dataset schema, so stratified random splitting was the best available option.

One thing I'd try with more time is ensembling the 5 fold probes at inference time (averaging their probabilities) rather than refitting on train+val — that usually gives a consistent small boost without any extra training.
