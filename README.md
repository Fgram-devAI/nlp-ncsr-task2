# NLP NCSR — Assignment 2 (RNNs & Word Embeddings)

Implementation of Assignment 2 for the NCSR "Επεξεργασία Φυσικής Γλώσσας" (NLP) course.

The assignment has three parts:

| Part | Topic                                       | Status      |
|------|---------------------------------------------|-------------|
| A    | Word embeddings (word2vec / GloVe)          | done        |
| B    | Traditional text classification (NB / SVM)  | done        |
| C    | Text classification with RNNs / LSTMs       | not started |

## Setup

```bash
# 1. Create the conda env (Python 3.11)
conda create -n nlp-ncsr-task2 python=3.11 -y
conda activate nlp-ncsr-task2

# 2. Install Python dependencies
pip install -r requirements.txt
```

The first run that loads the pre-trained word embeddings will download
~2 GB to `~/gensim-data/` (cached for subsequent runs):

- `word2vec-google-news-300` — 1.66 GB
- `glove-wiki-gigaword-300`  — 376 MB

## Project structure

```
nlp-ncsr-task2/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/                                  # one notebook per part
│   ├── part_a_embeddings.ipynb
│   └── part_b_traditional.ipynb
├── part_a_embeddings/                          # Part A scripts
│   ├── embeddings_utils.py
│   ├── a1_given_words.py … a6_tsne_glove.py
│   └── figures/tsne_glove_a6.png
└── part_b_traditional_txt_classification/      # Part B scripts
    ├── data_utils.py                           # AG News loader (kagglehub)
    ├── b1_train_models.py                      # train + report 4 models
    └── b2_error_analysis.py                    # docs all 4 models miss
```

## Running Part A

Each exercise is a standalone script. Run from inside `part_a_embeddings/`:

```bash
cd part_a_embeddings
python a1_given_words.py
python a2_own_words.py
python a3_student.py
python a4_given_analogies.py
python a5_own_analogies.py
python a6_tsne_glove.py     # also opens a matplotlib window
```

Or run everything in one place via the notebook:

```bash
jupyter lab notebooks/part_a_embeddings.ipynb
```

## Running Part B

The dataset auto-downloads via `kagglehub` on first run (requires Kaggle
credentials in `~/.kaggle/kaggle.json`). Cached under `~/.cache/kagglehub/`.

```bash
cd part_b_traditional_txt_classification
python b1_train_models.py        # accuracy / dim / time table for all 4 models
python b2_error_analysis.py      # docs misclassified by all 4
```

Or in one place:

```bash
jupyter lab notebooks/part_b_traditional.ipynb
```

