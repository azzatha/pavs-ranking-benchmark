# PAVS Gene Prioritization Benchmark

Circularity-controlled benchmark for phenotype-driven gene prioritization 
using the PAVS dataset.

## Paper
> Disentangling Phenotype Source from Population in Gene Prioritization:
> A Benchmark Study Using PAVS

## Repository structure

| Folder | Contents |
|---|---|
| `scripts/` | Analysis scripts |
| `data/` | Instructions to obtain input data |
| `results/` | Output tables and figures |

## Requirements

```bash
pip install pandas numpy matplotlib
```

## Data

Download from Zenodo ([PAVS v1.1.0](https://doi.org/10.5281/zenodo.19311539)):
- `phenotype_similarity_comprehensive.csv` → place in `data/`
- `literature.ttl` → place in `data/`

Download from HPO:
- `phenotype.hpoa` → https://hpo.jax.org/data/annotations → place in `data/`

## Running the analysis

### 1. Check HPOA overlap
```bash
python scripts/hpoa_overlap_check.py
```

### 2. Circularity-controlled evaluation
```bash
python scripts/circularity_controlled_analysis.py
```

Results are saved to `results/`.

## Citation
If you use this code, please cite the PAVS paper:
> Abdelhakim, M., Althagafi, A., Schofield, P. N., & Hoehndorf, R. (2026). PAVS: A Standardized Database of Phenotype-Associated Variants from Saudi Arabian Rare Disease Patients. medRxiv, 2026-04.