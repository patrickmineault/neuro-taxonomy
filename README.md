# Brain-taxonomy explorer

Categorize different neuroscience abstracts according to L1/L2/L3 taxonomy from [Liu et al. (2025)](https://arxiv.org/abs/2504.01990). In broad strokes, the taxonomy in Liu et al. defines cognitive capabilities that L1) are already mastered by AI L2) are starting to get explored and L3) are broadly unexplored. But those categories that Liu et al. found, how are they represented in the neuroscience literature?

We tried to look at this from several angles, and ended up with a simple two-tier method:

1. For intensive neuroimaging experiments, we manually labelled the tasks covered by the experiments
2. For the broader literature, we let Claude assign each study in different databases the category with the highest load, and a secondary category.

We also tried to use existing Cognitive Atlas categories in existing fMRI meta-analysis datasets, but the results were not great, mostly because the Cognitive Atlas taxonomy is fragmented and the automatic tagging in datasets like NeuroQuery and NeuroSynth is quite noisy (e.g. visual is different than vision). To "resolve" this, we tried to add numerous epicycles to attempt to clean up the categorization, but this quickly became unwieldy; it transpired that it was easier and cleaner (though at the cost of ~$200 in Claude credits) to simply let an LLM decide the right buckets to put the papers in.

## Usage

To download abstracts from psyarxiv and biorxiv:

```
python download_neuroscience_abstracts.py
```

To categorize different datasets into the L1/L2/L3 categorization using Claude:

```
python batch_categorize_papers.py --source data/biorxiv_neuroscience_abstracts_2025.csv
python batch_categorize_papers.py --source data/psyarxiv_neuroscience_abstracts_2025.csv

# For data from Neuroquery
python batch_categorize_papers.py 
```

(note this last one uses datasets.py). This uses the `CLAUDE_API_KEY` in `.env`.

To categorize instead using the built-in categories in NeuroQuery and NeuroSynth leveraging cognitive atlas terms, use:

```
python count_brains_vs_ai_papers.py
```

This uses `cogat_paper_count.py`.

Then to plot the results, run `plot_paper_counts.R`.

Separately, to plot paper counts for *intensive* neuroimaging, use `plot_intensive_paper_counts.R`.



