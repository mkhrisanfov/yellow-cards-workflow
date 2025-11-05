## Towards robust databases: an ensemble-based workflow for error detection applied to chemical data

_by Mikhail Khrisanfov, Dmitriy Matyushin, Andrey Samokhin_

Full text of the preprint is available at [ChemRxiv](https://doi.org/10.26434/chemrxiv-2025-6bh22).

At the moment the repository contains code for model training, dataset generation, data cleaning, data analysis and plotting figures for the manuscript.

The file with all the examples, figures, and code needed to reproduce the findings in the preprint is available in [`notebooks/data-analysis-article.ipynb`](./notebooks/data-analysis-article.ipynb).

### Abstract

This study presents a validation and refinement of the “yellow cards” error detection workflow that can be applied to any property connected to molecular structure. In our implementation the workflow employed 5 predictive models with each assigning a “yellow card” to 5% of the entries with worst prediction accuracy. The entries with 5 “yellow cards” were considered erroneous. All the base hypotheses from previous studies were tested and verified using computational datasets with controlled number and type of errors added.
Results confirm the five key hypotheses. Firstly, (i) predictive models generalize effectively and ignore errors during training, and (ii) exhibit weakly correlated prediction errors across architectures, supporting the efficacy of ensemble voting using diverse models. Furthermore, (iii) the group containing maximum number of “yellow cards” is dominated by the erroneous entries, which is indicated by (iv) the U-shaped distribution of entries across groups and the inverted-U pattern in standard deviations of predictions. These visual cues serve as robust, model-agnostic indicators for assessing workflow performance. Finally, (v) the "yellow cards" workflow outperforms simpler methods (absolute error or percentile) in precision-recall metrics.
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
