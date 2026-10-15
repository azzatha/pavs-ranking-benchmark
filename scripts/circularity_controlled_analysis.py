import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_hpoa_pmids(hpoa_path):
    """Return set of PMIDs (strings) referenced in phenotype.hpoa."""
    hpoa = pd.read_csv(
        hpoa_path, sep='\t', comment='#', low_memory=False,
        names=['DatabaseID','DiseaseName','Qualifier','HPO_ID',
               'Reference','Evidence','Onset','Frequency',
               'Sex','Modifier','Aspect','Biocuration']
    )
    pmids = set()
    for val in hpoa['Reference'].dropna():
        pmids.update(re.findall(r'PMID:(\d+)', str(val), re.IGNORECASE))
    return pmids


def load_overlapping_pmids_from_file(txt_path):
    """Load pre-computed overlapping PMIDs from lit_overlapping_pmids.txt."""
    pmids = set()
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            m = re.search(r'(\d+)', line)
            if m:
                pmids.add(m.group(1))
    return pmids


def get_pmid_from_pavs_id(pavs_id):
    """Extract PMID from a pavs_id that contains PMID_<number>.
    Works for IDs like:
      PMID_9398836_Family_J2_20_III_3
      lit_PMID_28513613_family_1
      PMID:28513613
    Returns None for non-literature IDs.
    """
    m = re.search(r'PMID[_:](\d+)', str(pavs_id))
    return m.group(1) if m else None


def tag_overlap(df_all, hpoa_pmids):
    """Add 'pmid' and 'in_hpoa' columns based on pavs_id pattern."""
    df_all = df_all.copy()
    df_all['pmid'] = df_all['pavs_id'].apply(get_pmid_from_pavs_id)
    df_all['in_hpoa'] = df_all['pmid'].apply(
        lambda p: p in hpoa_pmids if p is not None else False
    )
    return df_all


def compute_metrics(g_df):
    if g_df.empty:
        return None
    ranks    = g_df['rank'].values
    num_cand = g_df['num_candidates'].values
    return {
        'n':        len(g_df),
        'Hits@1':   np.mean(ranks <= 1),
        'Hits@10':  np.mean(ranks <= 10),
        'Mean AUC': np.mean((num_cand - ranks) / (num_cand - 1)),
        'MRR':      np.mean(1.0 / ranks),
    }


def roc_points(g_df):
    ranks    = g_df['rank'].values
    num_cand = g_df['num_candidates'].values
    sorted_ranks = np.sort(ranks)
    unique_ranks, counts = np.unique(sorted_ranks, return_counts=True)
    tpr = np.cumsum(counts) / len(ranks)
    fpr = (unique_ranks - 1) / (num_cand[0] - 1)
    fpr = np.insert(fpr, 0, 0)
    tpr = np.insert(tpr, 0, 0)
    return fpr, tpr


# ── Main analysis ─────────────────────────────────────────────────────────────

def circularity_controlled_analysis(
    results_path,
    output_dir,
    hpoa_path='data/phenotype.hpoa',
    overlapping_pmids_path='results/lit_overlapping_pmids.txt',
):
    print(f"Loading results from {results_path}...")
    df_all = pd.read_csv(results_path)
    os.makedirs(output_dir, exist_ok=True)

    # ── Load HPOA overlapping PMIDs ───────────────────────────────────────────
    if os.path.exists(overlapping_pmids_path):
        print(f"Loading overlapping PMIDs from {overlapping_pmids_path}...")
        hpoa_pmids = load_overlapping_pmids_from_file(overlapping_pmids_path)
        print(f"  PMIDs in overlap set: {len(hpoa_pmids)}")
    else:
        print(f"  {overlapping_pmids_path} not found — loading directly from HPOA...")
        hpoa_pmids = load_hpoa_pmids(hpoa_path)
        print(f"  Unique PMIDs in HPOA: {len(hpoa_pmids)}")

    # ── Tag overlap ───────────────────────────────────────────────────────────
    df_all = tag_overlap(df_all, hpoa_pmids)

    # ── Debug: show ID format to verify matching ──────────────────────────────
    lit_rows = df_all[df_all['group'] == 'Literature']
    print(f"\nSample pavs_id values (Literature group):")
    for v in lit_rows['pavs_id'].head(3):
        pmid = get_pmid_from_pavs_id(v)
        print(f"  {repr(v)}  →  pmid={pmid}")

    # ── Overlap summary ───────────────────────────────────────────────────────
    ic0, sim0 = 'extrinsic', 'resnik'
    lit_sample = df_all[
        (df_all['group'] == 'Literature') &
        (df_all['ic_type'] == ic0) &
        (df_all['sim_method'] == sim0)
    ]
    n_total   = len(lit_sample)
    n_overlap = int(lit_sample['in_hpoa'].sum())
    n_clean   = n_total - n_overlap

    print(f"\nLiterature cohort HPOA overlap (Extrinsic Resnik):")
    print(f"  Total:        {n_total}")
    if n_total > 0:
        print(f"  In HPOA:      {n_overlap} ({n_overlap/n_total*100:.1f}%)")
        print(f"  NOT in HPOA:  {n_clean}  ({n_clean/n_total*100:.1f}%)")
    else:
        print("  No Literature rows found — check 'group' column values:")
        print(f"  {df_all['group'].value_counts().to_dict()}")

    # ── Cohort definitions ────────────────────────────────────────────────────
    cohort_defs = {
        'Saudi':                     lambda d: d[d['group'] == 'Saudi'],
        'DDD':                       lambda d: d[d['group'] == 'DDD'],
        'Literature (full)':         lambda d: d[d['group'] == 'Literature'],
        'Literature (non-HPOA)':     lambda d: d[(d['group'] == 'Literature') & (~d['in_hpoa'])],
        'Literature (HPOA-overlap)': lambda d: d[(d['group'] == 'Literature') & (d['in_hpoa'])],
    }

    combinations = [
        ('intrinsic',  'lin'),
        ('intrinsic',  'resnik'),
        ('extrinsic',  'lin'),
        ('extrinsic',  'resnik'),
    ]

    full_stats = []

    for ic_type, sim_method in combinations:
        print(f"\nAnalyzing {ic_type}-{sim_method}...")
        df = df_all[
            (df_all['ic_type']    == ic_type) &
            (df_all['sim_method'] == sim_method)
        ].copy()

        plt.figure(figsize=(10, 7))
        plt.title(
            f'Circularity-Controlled ROC — Literature\n'
            f'({ic_type.capitalize()} {sim_method.capitalize()})',
            fontsize=13
        )

        for label, selector in cohort_defs.items():
            g_df = selector(df)
            m = compute_metrics(g_df)
            if m is None:
                print(f"  No data for: {label}")
                continue

            full_stats.append({
                'IC Type':    ic_type,
                'Similarity': sim_method,
                'Cohort':     label,
                **m,
            })

            if 'Literature' in label:
                fpr, tpr = roc_points(g_df)
                plt.plot(
                    fpr, tpr,
                    label=f'{label} (AUC={m["Mean AUC"]:.4f}, n={m["n"]})',
                    linewidth=2
                )

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, 'figures', f'circularity_roc_{ic_type}_{sim_method}.png'),
            dpi=300
        )
        plt.close()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    stats_df = pd.DataFrame(full_stats)
    csv_path = os.path.join(output_dir, 'circularity_controlled_stats.csv')
    stats_df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n### Circularity-Controlled Results")
    print("| Cohort | Method | n | AUC | MRR | Hits@1 (%) | Hits@10 (%) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for _, s in stats_df.sort_values(['Cohort', 'IC Type', 'Similarity']).iterrows():
        method = f"{s['IC Type'].capitalize()} {s['Similarity'].capitalize()}"
        print(f"| {s['Cohort']} | {method} | "
              f"{s['n']} | {s['Mean AUC']:.4f} | {s['MRR']:.4f} | "
              f"{s['Hits@1']*100:.2f} | {s['Hits@10']*100:.2f} |")

    # ── Save markdown ─────────────────────────────────────────────────────────
    lines = []
    lines.append("## Circularity-Controlled Evaluation\n\n")
    if n_total > 0:
        lines.append(
            f"> Literature cohort HPOA overlap: "
            f"{n_overlap}/{n_total} cases ({n_overlap/n_total*100:.1f}%) "
            f"have a PMID referenced in phenotype.hpoa\n\n"
        )

    lines.append("### All Combinations × All Cohorts\n")
    lines.append("| Cohort | Method | n | AUC | MRR | Hits@1 (%) | Hits@10 (%) |\n")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for _, s in stats_df.sort_values(['Cohort', 'IC Type', 'Similarity']).iterrows():
        method = f"{s['IC Type'].capitalize()} {s['Similarity'].capitalize()}"
        lines.append(
            f"| {s['Cohort']} | {method} | "
            f"{s['n']} | {s['Mean AUC']:.4f} | {s['MRR']:.4f} | "
            f"{s['Hits@1']*100:.2f} | {s['Hits@10']*100:.2f} |\n"
        )

    lines.append("\n### Best Method per Cohort (by MRR)\n")
    lines.append("| Cohort | Method | n | AUC | MRR | Hits@1 (%) | Hits@10 (%) |\n")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    best = stats_df.loc[stats_df.groupby('Cohort')['MRR'].idxmax()]
    for _, s in best.iterrows():
        method = f"{s['IC Type'].capitalize()} {s['Similarity'].capitalize()}"
        lines.append(
            f"| {s['Cohort']} | {method} | "
            f"{s['n']} | {s['Mean AUC']:.4f} | {s['MRR']:.4f} | "
            f"{s['Hits@1']*100:.2f} | {s['Hits@10']*100:.2f} |\n"
        )

    md_path = os.path.join(output_dir, 'circularity_controlled_summary.md')
    with open(md_path, 'w') as f:
        f.writelines(lines)
    print(f"Markdown summary saved to {md_path}")


if __name__ == "__main__":
    circularity_controlled_analysis(
        results_path='data/phenotype_similarity_comprehensive.csv',
        output_dir='results/',
        hpoa_path='data/phenotype.hpoa',
        overlapping_pmids_path='results/lit_overlapping_pmids.txt',
    )