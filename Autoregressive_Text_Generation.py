#!/usr/bin/env python
# coding: utf-8

# # Autoregressive Text Generation on Tiny Shakespeare
# ### SimpleRNN vs Bidirectional RNN vs LSTM vs GRU  |  Foundations of Generative AI (ICT4403) - Assignment 2
# 
# **Objective.** Build a word-level autoregressive language model that predicts the next word given the previous `SEQ_LEN` words, train four recurrent architectures under identical conditions, compare them quantitatively (loss, perplexity, accuracy, training time) and qualitatively (coherence, diversity, contextual consistency, repetition), then study the effect of sampling temperature.
# 
# **Dataset.** *Tiny Shakespeare* (Karpathy, `char-rnn`) - https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# 
# **Notebook layout**
# 1. Setup & configuration
# 2. **Task (i)** - dataset characteristics, preprocessing, input-target sequence construction, model definition and training
# 3. **Task (ii)** - quantitative & qualitative evaluation / comparison
# 4. **Task (iii)** - temperature study and selection of the best model

# ## 1. Setup and configuration

# In[1]:


import os, re, time, json, random, platform
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from numpy.lib.stride_tricks import sliding_window_view

# ---------- Experimental configuration (IDENTICAL for all four models) ----------
SEED        = 42
SEQ_LEN     = 10      # context window (words) -> predict word 11
MAX_VOCAB   = 3000    # most frequent word types kept (incl. <pad>, <unk>)
EMB_DIM     = 64
UNITS       = 128     # recurrent width (Bi-RNN uses 64 per direction = 128 total)
DROPOUT     = 0.3
BATCH       = 256
EPOCHS      = 8
LR          = 2e-3
PATIENCE    = 2       # early stopping on validation loss
TEMPS       = [0.5, 0.8, 1.0, 1.2]
GEN_LEN     = 50      # words generated per sample

DATA_PATH   = 'data/tinyshakespeare.txt'
DATA_URL    = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
os.makedirs('data', exist_ok=True); os.makedirs('results', exist_ok=True); os.makedirs('models', exist_ok=True)

def set_seeds(s=SEED):
    random.seed(s); np.random.seed(s); tf.keras.utils.set_random_seed(s)
set_seeds()
print('TensorFlow', tf.__version__, '| Python', platform.python_version(), '| devices:', [d.device_type for d in tf.config.list_physical_devices()])


# ## 2. Task (i) - Dataset, preprocessing and sequence construction
# 
# ### 2.1 Dataset characteristics

# In[2]:


if not os.path.exists(DATA_PATH):
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
raw = open(DATA_PATH, encoding='utf-8').read().replace('\r', '')

lines = raw.split('\n')
speaker_lines = [l for l in lines if l.strip().endswith(':') and l.strip()]
raw_words = re.findall(r"[A-Za-z']+", raw)
stats = {
    'characters': len(raw),
    'lines (incl. blank)': len(lines),
    'non-empty lines': sum(1 for l in lines if l.strip()),
    'speaker-label lines': len(speaker_lines),
    'distinct speakers': len(set(speaker_lines)),
    'word tokens (raw)': len(raw_words),
    'distinct words (case-sensitive)': len(set(raw_words)),
    'distinct words (lower-cased)': len(set(w.lower() for w in raw_words)),
    'avg words / non-empty line': round(len(raw_words) / sum(1 for l in lines if l.strip()), 2),
}
print(pd.Series(stats).to_string())
print('\n--- first 400 characters ---\n' + raw[:400])


# In[3]:


top = Counter(w.lower() for w in raw_words).most_common(20)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].bar([w for w, _ in top], [c for _, c in top], color='#4C72B0'); ax[0].set_title('20 most frequent words'); ax[0].tick_params(axis='x', rotation=60)
freqs = np.array(sorted(Counter(w.lower() for w in raw_words).values(), reverse=True))
ax[1].loglog(np.arange(1, len(freqs)+1), freqs, color='#C44E52'); ax[1].set_title("Zipf plot (rank vs frequency, log-log)"); ax[1].set_xlabel('rank'); ax[1].set_ylabel('frequency')
plt.tight_layout(); plt.savefig('results/dataset_characteristics.png', dpi=140); plt.show()


# **Observations.** Tiny Shakespeare is ~1.1 MB of dialogue-form text from several Shakespeare plays. It has a heavy-tailed (Zipfian) vocabulary: a few function words (*the, and, to, i, of*) dominate while thousands of words appear only once or twice. It also has strong *structure* - speaker labels (`NAME:`) followed by newline-separated verse lines, archaic vocabulary (*thou, thy, hath*) and frequent punctuation. These properties make it a good test-bed: local grammar and style can be learned by a small model, but long-range plot coherence cannot.

# ### 2.2 Pre-processing and tokenisation

# In[4]:


def tokenize(text):
    text = text.lower().replace('\r', '')
    text = re.sub(r'\n+', '\n', text)                        # collapse blank lines -> single newline token
    return re.findall(r"[a-z]+(?:'[a-z]+)*|[.,!?;:\-]|\n", text)   # words (with inner apostrophes), punctuation, newline

tokens = tokenize(raw)
counter = Counter(tokens)
PAD, UNK = 0, 1
vocab = ['<pad>', '<unk>'] + [w for w, _ in counter.most_common(MAX_VOCAB - 2)]
word2idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
ids = np.array([word2idx.get(t, UNK) for t in tokens], dtype=np.int32)

print(f'Total tokens           : {len(tokens):,}')
print(f'Distinct token types   : {len(counter):,}')
print(f'Vocabulary kept (V)    : {V:,}')
print(f'Token coverage by vocab: {(ids != UNK).mean()*100:.2f}%   (<unk> rate {(ids == UNK).mean()*100:.2f}%)')
print('First 25 tokens        :', tokens[:25])


# ### 2.3 Input-target sequence construction (autoregressive framing)
# A sliding window of length `SEQ_LEN + 1` (stride 1) is moved over the token-id stream. The first `SEQ_LEN` ids are the **input** and the last id is the **target** (the next word). Maximising the likelihood of every target given its history is exactly the autoregressive factorisation  P(w1..wn) = prod P(wt | w<t).
# 
# Data are split **chronologically** (90 % train / 5 % validation / 5 % test) so the test set contains text the model has never seen.

# In[5]:


windows = sliding_window_view(ids, SEQ_LEN + 1)
X_all, y_all = windows[:, :-1].copy(), windows[:, -1].copy()
n = len(X_all); n_tr, n_va = int(0.90 * n), int(0.05 * n)
X_tr, y_tr = X_all[:n_tr], y_all[:n_tr]
X_va, y_va = X_all[n_tr:n_tr+n_va], y_all[n_tr:n_tr+n_va]
X_te, y_te = X_all[n_tr+n_va:], y_all[n_tr+n_va:]
print(f'Total sequences {n:,} | train {len(X_tr):,} | val {len(X_va):,} | test {len(X_te):,} | X shape {X_all.shape}')

def decode(seq): return ' '.join(vocab[i] if vocab[i] != '\n' else '\\n' for i in seq)
print('\nExample input -> target pairs:')
for k in (0, 1, 2, 5000):
    print(f'  [{decode(X_all[k])}]  ->  {decode([y_all[k]])}')

def make_ds(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle: ds = ds.shuffle(50_000, seed=SEED)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
ds_tr, ds_va, ds_te = make_ds(X_tr, y_tr, True), make_ds(X_va, y_va), make_ds(X_te, y_te)


# ### 2.4 Model architectures
# All four models share the same pipeline - only the recurrent layer changes:
# 
# `Embedding(V, 64)  ->  [recurrent layer]  ->  Dropout(0.3)  ->  Dense(V, softmax)`
# 
# | Model | Recurrent layer | Notes |
# |---|---|---|
# | SimpleRNN | `SimpleRNN(128)` | plain tanh recurrence, suffers from vanishing gradients |
# | BiRNN | `Bidirectional(SimpleRNN(64))` | 64 units per direction = 128 output dims, reads the *input window* forwards and backwards (generation is still left-to-right: bidirectionality is used only over the already-known context) |
# | LSTM | `LSTM(128)` | input/forget/output gates + cell state |
# | GRU | `GRU(128)` | reset/update gates, fewer parameters than LSTM |
# 
# Same optimiser (Adam, lr 2e-3, grad-clip 1.0), batch size (256), max epochs (8), early stopping (patience 2 on val-loss, best weights restored), seed, and data splits.

# In[6]:


def build_model(kind):
    rnn = {
        'SimpleRNN': lambda: layers.SimpleRNN(UNITS),
        'BiRNN':     lambda: layers.Bidirectional(layers.SimpleRNN(UNITS // 2)),
        'LSTM':      lambda: layers.LSTM(UNITS),
        'GRU':       lambda: layers.GRU(UNITS),
    }[kind]()
    m = models.Sequential([
        layers.Input((SEQ_LEN,), dtype='int32'),
        layers.Embedding(V, EMB_DIM, name='embedding'),
        rnn,
        layers.Dropout(DROPOUT),
        layers.Dense(V, activation='softmax', name='next_word'),
    ], name=kind)
    m.compile(optimizer=tf.keras.optimizers.Adam(LR, clipnorm=1.0),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name='top5')])
    return m

class EpochTimer(callbacks.Callback):
    def on_train_begin(self, logs=None): self.times = []
    def on_epoch_begin(self, epoch, logs=None): self._t = time.perf_counter()
    def on_epoch_end(self, epoch, logs=None): self.times.append(time.perf_counter() - self._t)

MODEL_NAMES = ['SimpleRNN', 'BiRNN', 'LSTM', 'GRU']
for k in MODEL_NAMES:
    print(f'{k:10s} trainable parameters: {build_model(k).count_params():,}')


# ### 2.5 Training all four models

# In[7]:


histories, timers, trained = {}, {}, {}
for name in MODEL_NAMES:
    ck = f'results/ckpt_{name}.json'
    if os.path.exists(ck) and os.path.exists(f'models/{name}.keras'):     # resume: skip already-trained models
        d = json.load(open(ck)); histories[name], timers[name] = d['history'], d['epoch_times']
        trained[name] = tf.keras.models.load_model(f'models/{name}.keras'); print(f'Loaded finished model {name}'); continue
    print(f'\n================ Training {name} ================')
    set_seeds()                                     # identical init seed / shuffling for every model
    model = build_model(name)
    timer = EpochTimer()
    es = callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    h = model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS, callbacks=[es, timer], verbose=2)
    histories[name], timers[name], trained[name] = h.history, timer.times, model
    model.save(f'models/{name}.keras')
    json.dump({'history': h.history, 'epoch_times': timer.times}, open(ck, 'w'))

json.dump({k: {'history': histories[k], 'epoch_times': timers[k]} for k in MODEL_NAMES},
          open('results/training_histories.json', 'w'), indent=1)


# ## 3. Task (ii) - Evaluation and comparison
# ### 3.1 Quantitative metrics

# In[8]:


rows = []
for name in MODEL_NAMES:
    m, h = trained[name], histories[name]
    te_loss, te_acc, te_top5 = m.evaluate(ds_te, verbose=0)
    va_loss = min(h['val_loss'])
    rows.append({
        'Model': name,
        'Params': m.count_params(),
        'Epochs run': len(h['loss']),
        'Best epoch': int(np.argmin(h['val_loss'])) + 1,
        'Train loss (final)': h['loss'][-1],
        'Val loss (best)': va_loss,
        'Test loss': te_loss,
        'Val perplexity': np.exp(va_loss),
        'Test perplexity': np.exp(te_loss),
        'Test top-1 acc': te_acc,
        'Test top-5 acc': te_top5,
        'Train time (s)': sum(timers[name]),
        'Sec / epoch': float(np.mean(timers[name])),
    })
results = pd.DataFrame(rows).set_index('Model')
results.round(4).to_csv('results/quantitative_results.csv')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
results.round(3)


# In[9]:


colors = {'SimpleRNN': '#8172B2', 'BiRNN': '#CCB974', 'LSTM': '#4C72B0', 'GRU': '#55A868'}
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
for name in MODEL_NAMES:
    h = histories[name]; ep = range(1, len(h['loss']) + 1)
    ax[0].plot(ep, h['loss'], '--', color=colors[name]); ax[0].plot(ep, h['val_loss'], '-o', color=colors[name], label=name)
    ax[1].plot(ep, h['val_accuracy'], '-o', color=colors[name], label=name)
ax[0].set_title('Loss  (dashed = train, solid = validation)'); ax[0].set_xlabel('epoch'); ax[0].legend()
ax[1].set_title('Validation top-1 accuracy'); ax[1].set_xlabel('epoch'); ax[1].legend()
ax[2].bar(MODEL_NAMES, results['Train time (s)'], color=[colors[k] for k in MODEL_NAMES]); ax[2].set_title('Total training time (s)')
plt.tight_layout(); plt.savefig('results/training_curves.png', dpi=140); plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for a, col in zip(ax, ['Test perplexity', 'Test top-1 acc', 'Test top-5 acc']):
    a.bar(MODEL_NAMES, results[col], color=[colors[k] for k in MODEL_NAMES]); a.set_title(col)
    for i, v in enumerate(results[col]): a.text(i, v, f'{v:.3f}' if v < 5 else f'{v:.1f}', ha='center', va='bottom')
plt.tight_layout(); plt.savefig('results/test_metrics.png', dpi=140); plt.show()


# ### 3.2 Qualitative evaluation - generation machinery
# Text is generated autoregressively: feed the last `SEQ_LEN` tokens, obtain P(next word), apply **temperature** T (p_i proportional to p_i^(1/T)), sample, append, slide the window, repeat. `<pad>`/`<unk>` are masked out.
# 
# Because "coherence / diversity / repetition" are partly subjective, we complement reading the text with automatic **proxies** (computed on 5 seeds x 50 words per model/temperature):
# 
# | Proxy | Meaning | Better |
# |---|---|---|
# | **Distinct-1 / Distinct-2** | unique unigrams / bigrams / total (diversity) | higher |
# | **Repetition (rep-3)** | share of trigram occurrences that already appeared in the same sample | lower |
# | **Stutter rate** | share of positions where a token equals the previous token | lower |
# | **Bigram validity** | share of generated bigrams that occur in the training corpus (local coherence / grammaticality) | higher |
# | **Trigram validity** | share of generated trigrams that occur in the training corpus (longer contextual consistency) | higher* |
# 
# \*Very high trigram validity at low T can also mean the model is *copying* memorised text, so it is read together with diversity.

# In[10]:


SEEDS = [
    "i will not do it , my lord , for the",
    "what is the matter , good sir ? tell me",
    "the king is dead , and now the people must",
    "romeo :\nmy love is gone , and i must",
    "thou art a villain , and i will have thy",
]
def encode_seed(s):
    t = [word2idx.get(w, UNK) for w in tokenize(s)][-SEQ_LEN:]
    return [PAD] * (SEQ_LEN - len(t)) + t
seed_ids = np.array([encode_seed(s) for s in SEEDS], dtype=np.int32)
print('Seed <unk> counts:', [int((r == UNK).sum()) for r in seed_ids])

def sample_next(probs, T, rng):
    logits = np.log(np.clip(probs.astype('float64'), 1e-12, 1.0)) / T
    logits[:, [PAD, UNK]] = -1e9
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits); p /= p.sum(1, keepdims=True)
    return np.array([rng.choice(len(pi), p=pi) for pi in p])

def generate(model, seeds, T, n_words=GEN_LEN, rng=None):
    ctx, out = seeds.copy(), []
    for _ in range(n_words):
        probs = model(ctx, training=False).numpy()
        nxt = sample_next(probs, T, rng)
        out.append(nxt); ctx = np.concatenate([ctx[:, 1:], nxt[:, None]], axis=1)
    return np.stack(out, 1)

def detok(words):
    s = ' '.join(words)
    s = re.sub(r' ([.,!?;:])', r'\1', s)
    return s.replace(' \n ', '\n').replace(' \n', '\n').replace('\n ', '\n')

# n-gram sets from the TRAINING portion only (reference for validity proxies)
train_ids = ids[: n_tr + SEQ_LEN]
bi_ref = set(zip(train_ids[:-1], train_ids[1:]))
tri_ref = set(zip(train_ids[:-2], train_ids[1:-1], train_ids[2:]))

def text_metrics(seed_row, gen_row):
    full = list(seed_row[-2:]) + list(gen_row)        # allow seed->generation boundary n-grams
    g = list(gen_row)
    bi_all  = list(zip(full[:-1], full[1:]));  tri_all = list(zip(full[:-2], full[1:-1], full[2:]))
    bi_g    = list(zip(g[:-1], g[1:]));        tri_g   = list(zip(g[:-2], g[1:-1], g[2:]))
    return {
        'distinct1': len(set(g)) / len(g),
        'distinct2': len(set(bi_g)) / len(bi_g),
        'rep3': 1 - len(set(tri_g)) / len(tri_g),
        'stutter': float(np.mean([a == b for a, b in zip(g[:-1], g[1:])])),
        'bigram_validity': float(np.mean([b in bi_ref for b in bi_all[-len(g):]])),
        'trigram_validity': float(np.mean([t in tri_ref for t in tri_all[-len(g):]])),
    }


# ## 4. Task (iii) - Same seeds, four sampling temperatures
# Every model receives the **same 5 seeds** at **T = 0.5, 0.8, 1.0, 1.2** (the same RNG seed is used for each model/temperature so differences are due to the model, not luck).

# In[11]:


metric_rows, samples = [], {}
for name in MODEL_NAMES:
    for T in TEMPS:
        rng = np.random.default_rng(SEED)
        gen = generate(trained[name], seed_ids, T, rng=rng)
        for si in range(len(SEEDS)):
            words = [vocab[i] for i in gen[si]]
            samples[(name, T, si)] = detok(words)
            metric_rows.append({'Model': name, 'T': T, 'seed': si, **text_metrics(seed_ids[si], gen[si])})
gen_df = pd.DataFrame(metric_rows)
gen_df.to_csv('results/generation_metrics_raw.csv', index=False)
summary = gen_df.groupby(['Model', 'T']).mean(numeric_only=True).drop(columns='seed').round(3)
summary.to_csv('results/generation_metrics_summary.csv')

# save ALL generated samples (4 models x 4 temps x 5 seeds = 80 texts)
with open('results/generated_samples.md', 'w', encoding='utf-8') as f:
    f.write('# Generated text samples\n\nSeed -> 50 generated words. Same seeds for every model/temperature.\n')
    for T in TEMPS:
        f.write(f'\n## Temperature {T}\n')
        for si, s in enumerate(SEEDS):
            f.write(f'\n### Seed {si+1}: `{s!r}`\n')
            for name in MODEL_NAMES:
                txt = samples[(name, T, si)].replace('\n', ' / ')
                f.write(f'- **{name}**: {txt}\n')
summary


# ### 4.1 Generated text (seeds 1 and 2 shown here; all 80 samples are in `results/generated_samples.md`)

# In[12]:


for T in TEMPS:
    print('\n' + '#' * 90 + f'\n#  TEMPERATURE = {T}\n' + '#' * 90)
    for si in (0, 1):
        print(f'\n--- SEED {si+1}: "{SEEDS[si]}" ---')
        for name in MODEL_NAMES:
            print(f'[{name:9s}] ' + samples[(name, T, si)].replace('\n', ' / '))


# ### 4.2 Effect of temperature on coherence, diversity and repetition

# In[13]:


fig, ax = plt.subplots(2, 3, figsize=(17, 8))
panels = [('distinct2', 'Diversity - Distinct-2 (higher = more varied)'), ('rep3', 'Repetition - rep-3 (lower = better)'),
          ('stutter', 'Stutter rate (lower = better)'), ('bigram_validity', 'Local coherence - bigram validity'),
          ('trigram_validity', 'Contextual consistency - trigram validity'), ('distinct1', 'Distinct-1')]
for a, (col, title) in zip(ax.ravel(), panels):
    for name in MODEL_NAMES:
        s = summary.loc[name][col]; a.plot(s.index, s.values, '-o', color=colors[name], label=name)
    a.set_title(title); a.set_xlabel('temperature'); a.set_xticks(TEMPS)
ax[0, 0].legend(); plt.tight_layout(); plt.savefig('results/temperature_analysis.png', dpi=140); plt.show()


# In[14]:


for col in ['distinct2', 'rep3', 'bigram_validity', 'trigram_validity']:
    print(f'\n=== {col} (rows: model, columns: temperature) ===')
    print(summary[col].unstack().round(3).to_string())


# ## 5. Selecting the best autoregressive generator

# In[15]:


# Transparent composite: average RANK (1 = best) over 7 criteria. Generation criteria use T = 0.8 (balanced setting).
T_SEL = 0.8
sel = summary.xs(T_SEL, level='T')
crit = pd.DataFrame({
    'Test perplexity (low)':  results['Test perplexity'],
    'Test top-5 acc (high)':  results['Test top-5 acc'],
    'Train time (low)':       results['Train time (s)'],
    'Bigram validity (high)': sel['bigram_validity'],
    'Trigram validity (high)':sel['trigram_validity'],
    'Distinct-2 (high)':      sel['distinct2'],
    'Rep-3 (low)':            sel['rep3'],
})
lower_better = [c for c in crit.columns if '(low)' in c]
ranks = crit.rank(ascending=True)
for c in crit.columns:
    if c not in lower_better: ranks[c] = crit[c].rank(ascending=False)
ranks['Mean rank'] = ranks.mean(axis=1)
print(ranks.sort_values('Mean rank').round(2).to_string())
best = ranks['Mean rank'].idxmin()
print(f'\n>>> Best overall model by composite rank: {best}')
ranks.round(3).to_csv('results/model_ranking.csv')


# ## 6. Analysis and conclusions
# 
# ### 6.1 Model comparison (Task ii)
# All four models were trained with identical data, seed, optimiser, batch size, epoch budget and early-stopping rule (single run each, seed 42).
# 
# | Model | Test perplexity | Test top-1 | Test top-5 | Train time (s) |
# |---|---|---|---|---|
# | SimpleRNN | 91.9 | **26.5 %** | 45.2 % | 298 |
# | BiRNN | 91.4 | 26.4 % | 45.0 % | **258** |
# | LSTM | 89.2 | 25.2 % | 45.8 % | 451 |
# | GRU | **79.1** | 26.0 % | **46.5 %** | 580 |
# 
# * **Loss / perplexity.** GRU is clearly best (test loss 4.371, perplexity 79.1, about 11 % lower than LSTM's 89.2). LSTM, BiRNN and SimpleRNN are close to each other (89-92). Test loss is higher than validation loss for every model because the chronologically last 5 % of the corpus comes from different plays than the text seen in training (distribution shift), not because of a bug.
# * **Accuracy.** Top-1 accuracy is nearly flat (25-26.5 %) and SimpleRNN is nominally highest, so top-1 does not separate the models; top-5 accuracy and perplexity, which score the whole predicted distribution, do favour GRU. With one run per model, gaps of ~1 point in accuracy should not be over-interpreted.
# * **Training time.** The gated models cost more: GRU 580 s and LSTM 451 s versus 258-298 s for the plain RNNs. GRU being slower than LSTM is probably an artefact of Keras' CPU implementation on a shared single core (SimpleRNN epochs alone ranged 25-41 s), so timings are indicative only. BiRNN stopped after 7 epochs (early stopping); the others ran all 8.
# * **Bidirectional RNN.** It gave no real benefit over SimpleRNN. This is expected: generation is strictly left-to-right, and reading the already-known 10-word window backwards adds little.
# * **Qualitative.** At T = 0.5 all models fall into stock loops: repeated speaker labels (`duke vincentio:` several times in one sample), filler like "i have heard" / "i know not", and occasional stutters such as "i know not not" (LSTM). At T = 0.8 outputs look like Shakespeare-shaped text - speaker labels, line breaks, archaic phrasing - and are mostly locally grammatical, but none sustains a topic or an argument, and none stays on the seed's subject. This is the expected ceiling of a 10-word context window and a ~1 MB corpus.
# 
# ### 6.2 Influence of sampling temperature (Task iii)
# Means over 5 seeds x 50 words per model and temperature (so each cell is only 250 generated words; small differences are noise).
# 
# | Trend as T rises 0.5 -> 1.2 | Observation |
# |---|---|
# | **Coherence** (bigram validity) | falls for every model, from about 0.98 to 0.60-0.68; trigram validity falls from 0.71-0.79 to 0.20-0.25 |
# | **Diversity** (distinct-2) | rises from 0.84-0.94 to 0.96-0.996; distinct-1 from 0.55-0.64 to 0.77-0.83 |
# | **Repetition** (rep-3, stutter) | highest at T = 0.5 (LSTM 0.062, others <= 0.029); essentially 0 for T >= 1.0 |
# 
# * **T = 0.5 - safe but dull.** Almost every bigram is real Shakespeare-corpus text (about 0.98), but output is repetitive and generic, and trigram validity (0.71-0.79) shows it often reproduces common corpus phrases.
# * **T = 0.8 - best trade-off.** Repetition is nearly gone and diversity is high (distinct-2 about 0.92-0.95), while coherence is still reasonable (bigram validity 0.88-0.94).
# * **T = 1.0 - transition.** Bigram validity drops to 0.75-0.82; rare words and broken grammar start to appear.
# * **T = 1.2 - incoherent.** Highest diversity but text degrades into word salad ("party image the good souls worthy are he fond of wrongs ..."), with bigram validity of only 0.60-0.68. The diversity is partly the model sampling its low-probability tail, not creativity.
# 
# ### 6.3 Best model
# **GRU is the most effective autoregressive generator on this dataset**, with the caveats below:
# * lowest test loss and perplexity (79.1 vs 89-92) and the best top-5 accuracy;
# * at the balanced T = 0.8 it has the highest bigram validity (0.940 vs 0.884-0.904) and trigram validity (0.644 vs 0.500-0.584) with repetition near zero (rep-3 0.004). Its distinct-2 (0.922) is slightly lower than the others (0.939-0.947) - the flip side of staying closer to valid corpus text;
# * it has the best mean rank in the composite ranking (2.00), narrowly ahead of LSTM (2.07), BiRNN (2.93) and SimpleRNN (3.00).
# 
# **Caveats.** The composite lead over LSTM is small and depends on the chosen criteria and equal weights; GRU is also the slowest to train here. Results come from a single training run and 5 seeds, the coherence/consistency measures are proxies (n-gram validity against the training corpus), and the generated text, while stylistically Shakespearean, is not semantically coherent for any model. If training cost were the main concern, SimpleRNN/BiRNN give about 15 % higher perplexity at roughly half the training time. Natural next steps would be a longer context, several random seeds with confidence intervals, and a larger corpus or a Transformer baseline.
