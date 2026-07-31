## Towards robust databases: an ensemble-based workflow for error detection in scalar chemical data

_by Mikhail Khrisanfov, Dmitriy Matyushin, Anastasia Sholokhova, Andrey Samokhin_

Full text of the preprint is available at [ChemRxiv](https://doi.org/10.26434/chemrxiv-2025-6bh22).

At the moment the repository contains code for model training, dataset generation, data cleaning, data analysis and plotting figures for the manuscript.

The IPython Notebook file for the **preprint** with all the examples, figures, and code needed to reproduce the findings is available in [`notebooks/data-analysis-article.ipynb`](./notebooks/data-analysis-article.ipynb).

The IPython Notebook file for the **article** with all the examples, figures, and code needed to reproduce the findings is available in [`notebooks/data-analysis-article-nmi.ipynb`](./notebooks/data-analysis-article-nmi.ipynb).

### Abstract

This study presents a validation and refinement of the generalizable deep-learning based error detection workflow that may be applicable to multiple databases of structure-dependent molecular properties paving the way for more reliable data cleaning in chemistry. The workflow employs N (usually, 5) machine learning predictive models with each assigning a “yellow card” to *t%* of the entries with worst prediction accuracy. The entries with N “yellow cards” are considered erroneous. In our tests the “yellow cards” workflow outperformed filtering based on absolute error and percentile thresholds. Moreover, an early peak in the relationship between *t%* and the number of entries simultaneously classified as erroneous by all models provides a visual criterion for assessing the applicability of the workflow and selecting the filtering parameters.

We provide a step-by-step actionable plan for applying the method to datasets of properties connected to molecular structure, emphasizing model diversity, hyperparameter optimization, threshold selection, and iterative refinement using diagnostic plots. This work establishes a validated, generalizable framework for quality control in chemical data curation, extending to a broad range of structure-dependent molecular properties, and paves the way for more reliable data cleaning in chemistry.

### Installation

The project requires **Python 3.12+**
Clone the repository:

```
git clone https://github.com/mkhrisanfov/yellow-cards-workflow
```

Change folder to `yellow-cards-workflow`:

```
cd yellow-cards-workflow
```

Create a virtual environment (or install globally, for advanced users, skip to installing dependencies):

```
python -m venv .venv
```

**Activate virtual environment** for Linux:

```bash
source .venv/bin/activate
```

or **activate virtual environment** for Windows (cmd):

```cmd
.venv\Scripts\activate
```

or **activate virtual environment** for Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and `yellow_cards_workflow` package from [pyproject.toml](./pyproject.toml):

```
pip install -e .
```
