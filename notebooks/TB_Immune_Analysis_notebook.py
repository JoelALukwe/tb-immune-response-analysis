# TB Immune Response Analysis (script form)
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).parents[1]
cyto_path = root.parent / "cytokine-signature-analysis" / "data" / "synthetic_cytokine.csv"

# Load the synthetic cytokine data
df = pd.read_csv(cyto_path, index_col=0)
cytokines = [c for c in df.columns if c!='group']

# Basic group split
groupA = df[df['group']=='Latent'][cytokines]
groupB = df[df['group']=='Active'][cytokines]

# Differential stats
pvals = []
folds = []
for col in cytokines:
    t, p = ttest_ind(groupB[col], groupA[col], equal_var=False)
    pvals.append(p)
    folds.append((groupB[col].mean()+1e-9)/(groupA[col].mean()+1e-9))

stat_df = pd.DataFrame({
    'cytokine': cytokines,
    'pvalue': pvals,
    'fold_change': folds
}).set_index('cytokine')
stat_df['log2FC'] = np.log2(stat_df['fold_change'])
stat_df['-log10p'] = -np.log10(stat_df['pvalue'])
stat_df.to_csv(root / "results" / "differential_stats.csv")

# Volcano plot
plt.figure(figsize=(6,5))
plt.scatter(stat_df['log2FC'], stat_df['-log10p'])
for i, txt in enumerate(stat_df.index):
    plt.annotate(txt, (stat_df['log2FC'].iloc[i], stat_df['-log10p'].iloc[i]))
plt.xlabel("log2(FC) Active/Latent")
plt.ylabel("-log10(p-value)")
plt.title("Volcano plot (synthetic data)")
plt.savefig(root / "results" / "volcano.png", bbox_inches='tight')
plt.close()
