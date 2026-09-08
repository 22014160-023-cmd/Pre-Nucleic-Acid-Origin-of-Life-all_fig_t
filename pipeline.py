# ============================================================
# PRE-NUCLEIC ACID ORIGIN OF LIFE
# REPRODUCIBILITY PIPELINE
# ============================================================
#
# Manuscript:
# Mapping Chemical Continuity Before Nucleic Acids:
# A Computational Analysis of Prebiotic Molecular Chemical
# Space and Functional Emergence
#
# PART 1 — SETUP + FINAL DATASET + STRUCTURE QC
#
# Current analysis only.
#
# Final input:
# 193 molecular records
# -> 5 duplicate structures removed
# -> 188 unique molecular structures
#
# ============================================================

import os
import re
import sys
import json
import math
import random
import warnings
import hashlib
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 20260908

random.seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT = Path(
    "/content/drive/MyDrive/Pre_Nucleic_Acid_Origin_of_Life"
)

FINAL_ANALYSIS = PROJECT / "final_structure_analysis"

CSV_DIR = FINAL_ANALYSIS / "csv"
TABLE_DIR = FINAL_ANALYSIS / "tables"
STAT_DIR = FINAL_ANALYSIS / "statistics"
FIG_DIR = FINAL_ANALYSIS / "figures"
REPORT_DIR = FINAL_ANALYSIS / "reports"

for directory in [
    CSV_DIR,
    TABLE_DIR,
    STAT_DIR,
    FIG_DIR,
    REPORT_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

DATASET = (
    CSV_DIR /
    "24_FINAL_STRUCTURE_BASED_ANALYSIS_DATASET.csv"
)

# ------------------------------------------------------------
# Required packages
# ------------------------------------------------------------

try:
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import (
        Descriptors,
        rdFingerprintGenerator
    )
except ImportError:
    raise ImportError(
        "RDKit is required. In Google Colab run:\n"
        "!pip install rdkit"
    )

try:
    import scipy
    from scipy import stats
except ImportError:
    raise ImportError(
        "SciPy is required."
    )

try:
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except ImportError:
    raise ImportError(
        "scikit-learn is required."
    )

try:
    import matplotlib.pyplot as plt
    import networkx as nx
except ImportError:
    raise ImportError(
        "matplotlib and networkx are required."
    )

print("=" * 72)
print("PRE-NUCLEIC ACID ORIGIN OF LIFE")
print("REPRODUCIBILITY PIPELINE — PART 1")
print("=" * 72)

print("\nPython:", sys.version.split()[0])
print("Platform:", platform.platform())
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("SciPy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)

try:
    import rdkit
    print("RDKit:", rdkit.__version__)
except:
    pass

print("NetworkX:", nx.__version__)

# ------------------------------------------------------------
# Load final dataset
# ------------------------------------------------------------

if not DATASET.exists():
    raise FileNotFoundError(
        f"\nRequired dataset not found:\n{DATASET}\n\n"
        "Place the final structure-based dataset at this path "
        "before running the reproducibility pipeline."
    )

df = pd.read_csv(DATASET)

print("\nInput dataset:")
print("Shape:", df.shape)

# ------------------------------------------------------------
# Column detection
# ------------------------------------------------------------

def detect_column(
    columns,
    preferred,
    keyword=None
):

    for col in preferred:
        if col in columns:
            return col

    if keyword is not None:
        for col in columns:
            if keyword.lower() in str(col).lower():
                return col

    return None


SMILES_COL = detect_column(
    df.columns,
    [
        "canonical_smiles",
        "Canonical_SMILES",
        "CanonicalSMILES",
        "SMILES",
        "smiles"
    ],
    "smile"
)

LAYER_COL = detect_column(
    df.columns,
    [
        "Layer",
        "layer",
        "LAYER",
        "analytical_layer",
        "Analytical_Layer"
    ],
    "layer"
)

NAME_COL = detect_column(
    df.columns,
    [
        "source_name",
        "compound_name",
        "Compound_Name",
        "name",
        "Name",
        "compound"
    ],
    None
)

ID_COL = detect_column(
    df.columns,
    [
        "ID",
        "id",
        "compound_id",
        "Compound_ID"
    ],
    None
)

if SMILES_COL is None:
    raise ValueError(
        "SMILES column could not be identified.\n"
        f"Available columns: {list(df.columns)}"
    )

if LAYER_COL is None:
    raise ValueError(
        "Layer column could not be identified.\n"
        f"Available columns: {list(df.columns)}"
    )

print("\nDetected columns:")
print("SMILES:", SMILES_COL)
print("Layer :", LAYER_COL)
print("Name  :", NAME_COL)
print("ID    :", ID_COL)

# ------------------------------------------------------------
# Normalize analytical layer labels
# ------------------------------------------------------------

def normalize_layer(value):

    if pd.isna(value):
        return None

    s = str(value).strip()

    # L0 ... L5
    match = re.search(
        r"\bL\s*([0-5])\b",
        s,
        flags=re.IGNORECASE
    )

    if match:
        return f"L{match.group(1)}"

    # Layer 0 ... Layer 5
    match = re.search(
        r"layer[\s_\-]*([0-5])",
        s,
        flags=re.IGNORECASE
    )

    if match:
        return f"L{match.group(1)}"

    # Numeric layer
    try:

        value_num = int(float(s))

        if 0 <= value_num <= 5:
            return f"L{value_num}"

    except:
        pass

    # Textual aliases
    mapping = {

        "l0": "L0",
        "inorganic": "L0",
        "mineral": "L0",

        "l1": "L1",
        "simple organic": "L1",
        "simple_organic": "L1",

        "l2": "L2",
        "prebiotic building block": "L2",
        "prebiotic_building_block": "L2",

        "l3": "L3",
        "pre-nucleic": "L3",
        "pre_nucleic": "L3",
        "pre-nucleic functional system":
            "L3",
        "pre_nucleic_functional_system":
            "L3",

        "l4": "L4",
        "rna": "L4",
        "ribonucleotide": "L4",

        "l5": "L5",
        "dna": "L5",
        "deoxyribonucleotide": "L5"
    }

    return mapping.get(
        s.lower(),
        None
    )


df["Layer_Normalized"] = (
    df[LAYER_COL]
    .apply(normalize_layer)
)

if df["Layer_Normalized"].isna().any():

    bad_values = (
        df.loc[
            df["Layer_Normalized"].isna(),
            LAYER_COL
        ]
        .unique()
    )

    raise ValueError(
        f"Unrecognized layer values: {bad_values}"
    )

# ------------------------------------------------------------
# Structure validation
# ------------------------------------------------------------

valid_rows = []
molecules = []

invalid_smiles = []

for idx, smiles in enumerate(
    df[SMILES_COL].astype(str)
):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        invalid_smiles.append(
            {
                "row": idx,
                "smiles": smiles
            }
        )

    else:

        valid_rows.append(idx)
        molecules.append(mol)

if invalid_smiles:

    print(
        "\nWARNING:",
        len(invalid_smiles),
        "invalid SMILES removed."
    )

df = (
    df.iloc[valid_rows]
    .reset_index(drop=True)
)

print(
    "\nValid molecular structures:",
    len(df)
)

# ------------------------------------------------------------
# Verify final expected structure count
# ------------------------------------------------------------

EXPECTED_STRUCTURES = 188

if len(df) != EXPECTED_STRUCTURES:

    print(
        "\nWARNING:"
        f" Expected {EXPECTED_STRUCTURES} structures "
        f"but found {len(df)}."
    )

# ------------------------------------------------------------
# Verify unique structures
# ------------------------------------------------------------

canonical_smiles = []

for mol in molecules:

    canonical_smiles.append(
        Chem.MolToSmiles(
            mol,
            canonical=True
        )
    )

df["Canonical_SMILES_Verified"] = (
    canonical_smiles
)

duplicate_count = (
    df["Canonical_SMILES_Verified"]
    .duplicated()
    .sum()
)

print(
    "\nDuplicate canonical structures:",
    duplicate_count
)

if duplicate_count != 0:

    raise ValueError(
        "Final dataset contains duplicate molecular "
        "structures. The final analysis requires 188 "
        "unique structures."
    )

# ------------------------------------------------------------
# Layer counts
# ------------------------------------------------------------

layer_order = [
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5"
]

layer_counts = (
    df["Layer_Normalized"]
    .value_counts()
    .reindex(
        layer_order,
        fill_value=0
    )
)

print("\nFinal layer composition:")

for layer, count in layer_counts.items():

    print(
        f"{layer}: {count}"
    )

# ------------------------------------------------------------
# Save cleaned reproducibility dataset
# ------------------------------------------------------------

CLEAN_DATASET = (
    CSV_DIR /
    "final_188_unique_structures.csv"
)

df.to_csv(
    CLEAN_DATASET,
    index=False
)

# ------------------------------------------------------------
# Dataset checksum
# ------------------------------------------------------------

sha256 = hashlib.sha256()

with open(
    CLEAN_DATASET,
    "rb"
) as handle:

    for chunk in iter(
        lambda: handle.read(1024 * 1024),
        b""
    ):

        sha256.update(chunk)

dataset_hash = sha256.hexdigest()

with open(
    STAT_DIR /
    "dataset_sha256.txt",
    "w"
) as handle:

    handle.write(
        dataset_hash + "\n"
    )

print(
    "\nDataset SHA256:",
    dataset_hash
)

# ------------------------------------------------------------
# Basic QC table
# ------------------------------------------------------------

qc = pd.DataFrame(
    {
        "metric": [
            "input_records",
            "valid_structures",
            "unique_structures",
            "duplicate_structures",
            "invalid_smiles",
            "L0_count",
            "L1_count",
            "L2_count",
            "L3_count",
            "L4_count",
            "L5_count"
        ],
        "value": [
            len(pd.read_csv(DATASET)),
            len(df),
            df[
                "Canonical_SMILES_Verified"
            ].nunique(),
            duplicate_count,
            len(invalid_smiles),
            layer_counts["L0"],
            layer_counts["L1"],
            layer_counts["L2"],
            layer_counts["L3"],
            layer_counts["L4"],
            layer_counts["L5"]
        ]
    }
)

qc.to_csv(
    TABLE_DIR /
    "Table_Dataset_QC.csv",
    index=False
)

print("\nPART 1 COMPLETE.")

# ============================================================
# PART 2 — RDKit DESCRIPTORS + PCA + CHEMICAL DISTANCE
# ============================================================

print("=" * 72)
print("REPRODUCIBILITY PIPELINE — PART 2")
print("=" * 72)

# ------------------------------------------------------------
# Reconstruct molecules
# ------------------------------------------------------------

molecules = [
    Chem.MolFromSmiles(smiles)
    for smiles in df[SMILES_COL].astype(str)
]

if any(
    mol is None
    for mol in molecules
):

    raise ValueError(
        "Failed to reconstruct one or more molecules."
    )

# ------------------------------------------------------------
# RDKit descriptor panel
# ------------------------------------------------------------

descriptor_functions = {

    "MolWt":
        Descriptors.MolWt,

    "LogP":
        Descriptors.MolLogP,

    "TPSA":
        Descriptors.TPSA,

    "NumHDonors":
        Descriptors.NumHDonors,

    "NumHAcceptors":
        Descriptors.NumHAcceptors,

    "NumRotatableBonds":
        Descriptors.NumRotatableBonds,

    "RingCount":
        Descriptors.RingCount,

    "HeavyAtomCount":
        Descriptors.HeavyAtomCount,

    "FractionCSP3":
        Descriptors.FractionCSP3,

    "NumAromaticRings":
        Descriptors.NumAromaticRings,

    "NumAliphaticRings":
        Descriptors.NumAliphaticRings,

    "NumSaturatedRings":
        Descriptors.NumSaturatedRings,

    "NumHeterocycles":
        Descriptors.NumHeterocycles,

    "NumValenceElectrons":
        Descriptors.NumValenceElectrons,

    "MaxPartialCharge":
        Descriptors.MaxPartialCharge,

    "MinPartialCharge":
        Descriptors.MinPartialCharge,

    "MaxAbsPartialCharge":
        Descriptors.MaxAbsPartialCharge,

    "MinAbsPartialCharge":
        Descriptors.MinAbsPartialCharge,

    "LabuteASA":
        Descriptors.LabuteASA,

    "BalabanJ":
        Descriptors.BalabanJ
}

descriptor_names = list(
    descriptor_functions.keys()
)

descriptor_matrix = []

for mol in molecules:

    values = []

    for name, func in descriptor_functions.items():

        try:
            value = float(
                func(mol)
            )

        except:
            value = np.nan

        values.append(value)

    descriptor_matrix.append(
        values
    )

descriptor_df = pd.DataFrame(
    descriptor_matrix,
    columns=descriptor_names
)

# ------------------------------------------------------------
# Handle descriptor missing/non-finite values
# ------------------------------------------------------------

descriptor_df = (
    descriptor_df
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

if descriptor_df.isna().any().any():

    for col in descriptor_df.columns:

        median = descriptor_df[col].median()

        descriptor_df[col] = (
            descriptor_df[col]
            .fillna(median)
        )

# ------------------------------------------------------------
# Verify 20 descriptors
# ------------------------------------------------------------

if descriptor_df.shape[1] != 20:

    raise ValueError(
        f"Expected 20 descriptors, "
        f"found {descriptor_df.shape[1]}"
    )

print(
    "\nDescriptor matrix:",
    descriptor_df.shape
)

# ------------------------------------------------------------
# Save descriptor dataset
# ------------------------------------------------------------

descriptor_output = pd.concat(
    [
        df.reset_index(drop=True),
        descriptor_df
    ],
    axis=1
)

descriptor_output.to_csv(
    CSV_DIR /
    "final_188_RDKit_descriptors.csv",
    index=False
)

# ------------------------------------------------------------
# Standardize descriptors
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    descriptor_df.values
)

# ------------------------------------------------------------
# PCA
# ------------------------------------------------------------

pca_full = PCA()

X_pca = pca_full.fit_transform(
    X_scaled
)

explained = (
    pca_full
    .explained_variance_ratio_
)

print("\nPCA explained variance:")

for i, value in enumerate(
    explained[:10],
    start=1
):

    print(
        f"PC{i}: "
        f"{100 * value:.4f}%"
    )

# ------------------------------------------------------------
# PCA table
# ------------------------------------------------------------

pca_columns = [
    f"PC{i+1}"
    for i in range(
        X_pca.shape[1]
    )
]

pca_df = pd.DataFrame(
    X_pca,
    columns=pca_columns
)

pca_df.insert(
    0,
    "Layer",
    df["Layer_Normalized"].values
)

if NAME_COL is not None:

    pca_df.insert(
        0,
        "Name",
        df[NAME_COL].astype(str).values
    )

pca_df.to_csv(
    TABLE_DIR /
    "PCA_coordinates.csv",
    index=False
)

# ------------------------------------------------------------
# PCA statistics
# ------------------------------------------------------------

pca_statistics = pd.DataFrame(
    {
        "PC": [
            f"PC{i+1}"
            for i in range(
                len(explained)
            )
        ],
        "Explained_Variance": explained,
        "Explained_Variance_Percent":
            explained * 100
    }
)

pca_statistics[
    "Cumulative_Explained_Variance_Percent"
] = (
    pca_statistics[
        "Explained_Variance_Percent"
    ]
    .cumsum()
)

pca_statistics.to_csv(
    STAT_DIR /
    "PCA_explained_variance.csv",
    index=False
)

# ------------------------------------------------------------
# Expected key PCA values
# ------------------------------------------------------------

pc1 = explained[0] * 100
pc2 = explained[1] * 100
pc3 = explained[2] * 100

pc12 = (
    explained[:2].sum()
    * 100
)

pc123 = (
    explained[:3].sum()
    * 100
)

print(
    "\nKey PCA results:"
)

print(
    f"PC1 = {pc1:.4f}%"
)

print(
    f"PC2 = {pc2:.4f}%"
)

print(
    f"PC1 + PC2 = {pc12:.4f}%"
)

print(
    f"PC3 = {pc3:.4f}%"
)

print(
    f"PC1 + PC2 + PC3 = {pc123:.4f}%"
)

# ------------------------------------------------------------
# Molecular chemical distance
# ------------------------------------------------------------

# Euclidean distance in standardized
# descriptor space.

from scipy.spatial.distance import (
    pdist,
    squareform
)

chemical_distance_matrix = squareform(
    pdist(
        X_scaled,
        metric="euclidean"
    )
)

np.save(
    STAT_DIR /
    "chemical_distance_matrix.npy",
    chemical_distance_matrix
)

# ------------------------------------------------------------
# Layer gap matrix
# ------------------------------------------------------------

layer_numeric = (
    df["Layer_Normalized"]
    .str.extract(
        r"(\d)"
    )[0]
    .astype(int)
    .values
)

layer_gap_matrix = np.abs(
    layer_numeric[:, None]
    -
    layer_numeric[None, :]
)

np.save(
    STAT_DIR /
    "layer_gap_matrix.npy",
    layer_gap_matrix
)

# ------------------------------------------------------------
# Pairwise Spearman analysis
# ------------------------------------------------------------

upper_triangle = np.triu_indices(
    len(df),
    k=1
)

distance_values = (
    chemical_distance_matrix[
        upper_triangle
    ]
)

gap_values = (
    layer_gap_matrix[
        upper_triangle
    ]
)

rho, p_value = stats.spearmanr(
    distance_values,
    gap_values
)

distance_statistics = pd.DataFrame(
    {
        "metric": [
            "Spearman_rho",
            "P_value",
            "N_pairs"
        ],
        "value": [
            rho,
            p_value,
            len(distance_values)
        ]
    }
)

distance_statistics.to_csv(
    STAT_DIR /
    "Chemical_Distance_vs_Layer_Gap.csv",
    index=False
)

print(
    "\nChemical distance vs layer separation:"
)

print(
    f"Spearman rho = {rho:.8f}"
)

if p_value == 0:

    print(
        "P < 0.001"
    )

else:

    print(
        f"P = {p_value:.6g}"
    )

print(
    "\nPART 2 COMPLETE."
)




# ============================================================
# PART 3 — MOLECULAR SIMILARITY NETWORK
# + LAYER CONNECTIVITY
# + PERMUTATION VALIDATION
# ============================================================

print("=" * 72)
print("REPRODUCIBILITY PIPELINE — PART 3")
print("=" * 72)

# ------------------------------------------------------------
# Morgan fingerprints
# ------------------------------------------------------------

MORGAN_RADIUS = 2
MORGAN_SIZE = 2048
TANIMOTO_THRESHOLD = 0.20

morgan_generator = (
    rdFingerprintGenerator
    .GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_SIZE
    )
)

fingerprints = [
    morgan_generator.GetFingerprint(
        mol
    )
    for mol in molecules
]

print(
    "\nMorgan fingerprints generated:"
)

print(
    f"Radius = {MORGAN_RADIUS}"
)

print(
    f"Fingerprint size = {MORGAN_SIZE}"
)

# ------------------------------------------------------------
# Tanimoto similarity matrix
# ------------------------------------------------------------

n = len(fingerprints)

tanimoto_matrix = np.eye(
    n,
    dtype=float
)

for i in range(n):

    similarities = (
        DataStructs
        .BulkTanimotoSimilarity(
            fingerprints[i],
            fingerprints[i + 1:]
        )
    )

    for offset, similarity in enumerate(
        similarities
    ):

        j = i + 1 + offset

        tanimoto_matrix[
            i,
            j
        ] = similarity

        tanimoto_matrix[
            j,
            i
        ] = similarity

np.save(
    STAT_DIR /
    "tanimoto_similarity_matrix.npy",
    tanimoto_matrix
)

# ------------------------------------------------------------
# Threshold network
# ------------------------------------------------------------

G = nx.Graph()

for i in range(n):

    G.add_node(
        i,
        layer=df.loc[
            i,
            "Layer_Normalized"
        ],
        name=(
            str(
                df.loc[
                    i,
                    NAME_COL
                ]
            )
            if NAME_COL is not None
            else f"Compound_{i+1}"
        )
    )

for i in range(n):

    for j in range(
        i + 1,
        n
    ):

        similarity = (
            tanimoto_matrix[
                i,
                j
            ]
        )

        if (
            similarity
            >=
            TANIMOTO_THRESHOLD
        ):

            G.add_edge(
                i,
                j,
                similarity=float(
                    similarity
                ),
                distance=max(
                    1e-9,
                    1.0 - float(
                        similarity
                    )
                )
            )

# ------------------------------------------------------------
# Network statistics
# ------------------------------------------------------------

total_edges = G.number_of_edges()

within_edges = 0
cross_edges = 0

for u, v in G.edges():

    if (
        G.nodes[u]["layer"]
        ==
        G.nodes[v]["layer"]
    ):

        within_edges += 1

    else:

        cross_edges += 1

cross_fraction = (
    cross_edges
    /
    total_edges
)

components = (
    nx.number_connected_components(G)
)

largest_component = max(
    (
        len(component)
        for component in
        nx.connected_components(G)
    ),
    default=0
)

print("\nNetwork statistics:")
print(
    "Nodes:",
    G.number_of_nodes()
)
print(
    "Edges:",
    total_edges
)
print(
    "Within-layer edges:",
    within_edges
)
print(
    "Cross-layer edges:",
    cross_edges
)
print(
    f"Cross-layer fraction: "
    f"{100 * cross_fraction:.4f}%"
)
print(
    "Components:",
    components
)
print(
    "Largest component:",
    largest_component
)

network_stats = pd.DataFrame(
    {
        "metric": [
            "nodes",
            "edges",
            "within_layer_edges",
            "cross_layer_edges",
            "cross_layer_fraction",
            "connected_components",
            "largest_component",
            "tanimoto_threshold",
            "morgan_radius",
            "morgan_fingerprint_size"
        ],
        "value": [
            n,
            total_edges,
            within_edges,
            cross_edges,
            cross_fraction,
            components,
            largest_component,
            TANIMOTO_THRESHOLD,
            MORGAN_RADIUS,
            MORGAN_SIZE
        ]
    }
)

network_stats.to_csv(
    STAT_DIR /
    "network_statistics.csv",
    index=False
)

# ------------------------------------------------------------
# Layer-pair connectivity matrix
# ------------------------------------------------------------

observed_matrix = pd.DataFrame(
    0.0,
    index=layer_order,
    columns=layer_order
)

for u, v in G.edges():

    lu = G.nodes[u]["layer"]
    lv = G.nodes[v]["layer"]

    observed_matrix.loc[
        lu,
        lv
    ] += 1

    observed_matrix.loc[
        lv,
        lu
    ] += 1

# Correct diagonal because each within-layer edge
# was counted twice.
for layer in layer_order:

    observed_matrix.loc[
        layer,
        layer
    ] /= 2

observed_matrix.to_csv(
    TABLE_DIR /
    "layer_pair_observed_edges.csv"
)

# ------------------------------------------------------------
# Expected layer-pair connectivity
# ------------------------------------------------------------

layer_sizes = {
    layer: int(
        (
            df["Layer_Normalized"]
            ==
            layer
        ).sum()
    )
    for layer in layer_order
}

expected_matrix = pd.DataFrame(
    0.0,
    index=layer_order,
    columns=layer_order
)

N = n

for a in layer_order:

    for b in layer_order:

        if a == b:

            possible = (
                layer_sizes[a]
                *
                (
                    layer_sizes[a] - 1
                )
                /
                2
            )

        else:

            possible = (
                layer_sizes[a]
                *
                layer_sizes[b]
            )

        total_possible = (
            N * (N - 1) / 2
        )

        expected_matrix.loc[
            a,
            b
        ] = (
            possible
            /
            total_possible
            *
            total_edges
        )

expected_matrix.to_csv(
    TABLE_DIR /
    "layer_pair_expected_edges.csv"
)

# ------------------------------------------------------------
# Observed / expected
# ------------------------------------------------------------

oe_matrix = (
    observed_matrix
    /
    expected_matrix.replace(
        0,
        np.nan
    )
)

oe_matrix.to_csv(
    TABLE_DIR /
    "layer_pair_observed_expected.csv"
)

# ------------------------------------------------------------
# Permutation validation
#
# Fixed molecular network.
# Randomly permute layer labels.
# ------------------------------------------------------------

PERMUTATIONS = 10000

observed_labels = (
    df["Layer_Normalized"]
    .values
)

observed_cross_fraction = (
    np.mean(
        [
            observed_labels[u]
            !=
            observed_labels[v]
            for u, v in G.edges()
        ]
    )
)

rng = np.random.default_rng(
    SEED
)

permuted_cross_fractions = (
    np.empty(
        PERMUTATIONS,
        dtype=float
    )
)

edge_pairs = list(
    G.edges()
)

for permutation in range(
    PERMUTATIONS
):

    shuffled_labels = (
        rng.permutation(
            observed_labels
        )
    )

    cross_count = 0

    for u, v in edge_pairs:

        if (
            shuffled_labels[u]
            !=
            shuffled_labels[v]
        ):

            cross_count += 1

    permuted_cross_fractions[
        permutation
    ] = (
        cross_count
        /
        total_edges
    )

null_mean = (
    permuted_cross_fractions.mean()
)

null_sd = (
    permuted_cross_fractions.std(
        ddof=1
    )
)

z_score = (
    (
        observed_cross_fraction
        -
        null_mean
    )
    /
    null_sd
)

# One-sided enrichment test:
# H1 = observed cross-layer connectivity
# greater than randomized expectation.

p_greater = (
    np.sum(
        permuted_cross_fractions
        >=
        observed_cross_fraction
    )
    /
    PERMUTATIONS
)

# Empirical depletion test
p_less = (
    (
        np.sum(
            permuted_cross_fractions
            <=
            observed_cross_fraction
        )
        + 1
    )
    /
    (
        PERMUTATIONS
        + 1
    )
)

expected_exact = (
    1
    -
    (
        sum(
            count * (count - 1)
            for count in
            layer_sizes.values()
        )
        /
        (
            N * (N - 1)
        )
    )
)

enrichment_ratio = (
    observed_cross_fraction
    /
    null_mean
)

enrichment_percent = (
    enrichment_ratio - 1
) * 100

permutation_summary = pd.DataFrame(
    {
        "metric": [
            "permutations",
            "seed",
            "observed_cross_layer_fraction",
            "exact_random_expected_fraction",
            "null_mean",
            "null_sd",
            "enrichment_ratio",
            "enrichment_percent",
            "z_score",
            "p_greater",
            "p_less"
        ],
        "value": [
            PERMUTATIONS,
            SEED,
            observed_cross_fraction,
            expected_exact,
            null_mean,
            null_sd,
            enrichment_ratio,
            enrichment_percent,
            z_score,
            p_greater,
            p_less
        ]
    }
)

permutation_summary.to_csv(
    STAT_DIR /
    "cross_layer_permutation_summary.csv",
    index=False
)

np.save(
    STAT_DIR /
    "cross_layer_permutation_null.npy",
    permuted_cross_fractions
)

print("\nPermutation validation:")
print(
    f"Observed = "
    f"{100 * observed_cross_fraction:.2f}%"
)
print(
    f"Exact random expectation = "
    f"{100 * expected_exact:.2f}%"
)
print(
    f"Null mean = "
    f"{100 * null_mean:.2f}%"
)
print(
    f"Null SD = "
    f"{100 * null_sd:.2f}%"
)
print(
    f"Enrichment ratio = "
    f"{enrichment_ratio:.4f}x"
)
print(
    f"Z = {z_score:.4f}"
)
print(
    f"P(enrichment) = {p_greater:.6g}"
)
print(
    f"P(depletion) = {p_less:.6g}"
)

# ------------------------------------------------------------
# Save edge list
# ------------------------------------------------------------

edge_records = []

for u, v, data in G.edges(
    data=True
):

    edge_records.append(
        {
            "node_i": u,
            "node_j": v,
            "name_i":
                G.nodes[u]["name"],
            "name_j":
                G.nodes[v]["name"],
            "layer_i":
                G.nodes[u]["layer"],
            "layer_j":
                G.nodes[v]["layer"],
            "tanimoto_similarity":
                data["similarity"],
            "chemical_distance":
                data["distance"],
            "cross_layer":
                (
                    G.nodes[u]["layer"]
                    !=
                    G.nodes[v]["layer"]
                )
        }
    )

edge_df = pd.DataFrame(
    edge_records
)

edge_df.to_csv(
    CSV_DIR /
    "molecular_similarity_network_edges.csv",
    index=False
)

print("\nPART 3 COMPLETE.")



# ============================================================
# PART 4 — NETWORK ROBUSTNESS + BRIDGE CENTRALITY
# + 13 PUBLICATION FIGURES
# ============================================================

print("=" * 72)
print("REPRODUCIBILITY PIPELINE — PART 4")
print("=" * 72)

# ------------------------------------------------------------
# Network robustness
# ------------------------------------------------------------

robustness_multipliers = [
    0.80,
    0.90,
    1.00,
    1.10,
    1.20
]

robustness_records = []

for multiplier in robustness_multipliers:

    threshold = (
        TANIMOTO_THRESHOLD
        *
        multiplier
    )

    R = nx.Graph()

    R.add_nodes_from(
        G.nodes(
            data=True
        )
    )

    for u, v in G.edges(
        data=True
    ):

        if (
            data := G[u][v]
        ):

            similarity = (
                data["similarity"]
            )

            if similarity >= threshold:

                R.add_edge(
                    u,
                    v,
                    similarity=similarity
                )

    edges = R.number_of_edges()

    if edges > 0:

        cross = sum(
            R.nodes[u]["layer"]
            !=
            R.nodes[v]["layer"]
            for u, v in R.edges()
        )

        cross_fraction_R = (
            cross / edges
        )

    else:

        cross = 0
        cross_fraction_R = np.nan

    component_sizes = sorted(
        [
            len(c)
            for c in
            nx.connected_components(R)
        ],
        reverse=True
    )

    robustness_records.append(
        {
            "multiplier": multiplier,
            "threshold": threshold,
            "nodes": R.number_of_nodes(),
            "edges": edges,
            "cross_layer_edges": cross,
            "cross_layer_fraction":
                cross_fraction_R,
            "components":
                nx.number_connected_components(R),
            "largest_component":
                (
                    component_sizes[0]
                    if component_sizes
                    else 0
                )
        }
    )

robustness_df = pd.DataFrame(
    robustness_records
)

robustness_df.to_csv(
    STAT_DIR /
    "network_robustness.csv",
    index=False
)

print("\nNetwork robustness:")
print(
    robustness_df.to_string(
        index=False
    )
)

# ------------------------------------------------------------
# Weighted betweenness centrality
#
# distance = 1 - similarity
# because high similarity should mean short distance.
# ------------------------------------------------------------

betweenness = (
    nx.betweenness_centrality(
        G,
        weight="distance",
        normalized=True
    )
)

cross_degree = {}

for node in G.nodes():

    count = 0

    for neighbor in G.neighbors(
        node
    ):

        if (
            G.nodes[node]["layer"]
            !=
            G.nodes[neighbor]["layer"]
        ):

            count += 1

    cross_degree[node] = count

bridge_records = []

for node in G.nodes():

    bridge_records.append(
        {
            "node": node,
            "name":
                G.nodes[node]["name"],
            "layer":
                G.nodes[node]["layer"],
            "betweenness":
                betweenness[node],
            "cross_layer_degree":
                cross_degree[node],
            "degree":
                G.degree(node)
        }
    )

bridge_df = pd.DataFrame(
    bridge_records
)

bridge_df = (
    bridge_df
    .sort_values(
        [
            "cross_layer_degree",
            "betweenness"
        ],
        ascending=False
    )
)

bridge_df.to_csv(
    STAT_DIR /
    "bridge_centrality.csv",
    index=False
)

top_bridges = (
    bridge_df[
        bridge_df[
            "cross_layer_degree"
        ] > 0
    ]
    .sort_values(
        "betweenness",
        ascending=False
    )
    .head(12)
)

print("\nTop bridge molecules:")

print(
    top_bridges[
        [
            "name",
            "layer",
            "betweenness",
            "cross_layer_degree"
        ]
    ]
    .to_string(index=False)
)

# ------------------------------------------------------------
# Layer statistics
# ------------------------------------------------------------

layer_statistics = []

for layer in layer_order:

    nodes = [
        node
        for node in G.nodes()
        if G.nodes[node]["layer"]
        == layer
    ]

    degrees = [
        G.degree(node)
        for node in nodes
    ]

    layer_statistics.append(
        {
            "layer": layer,
            "nodes": len(nodes),
            "mean_degree":
                np.mean(degrees)
                if degrees
                else 0,
            "median_degree":
                np.median(degrees)
                if degrees
                else 0,
            "total_degree":
                np.sum(degrees)
                if degrees
                else 0
        }
    )

layer_statistics_df = pd.DataFrame(
    layer_statistics
)

layer_statistics_df.to_csv(
    STAT_DIR /
    "layer_network_statistics.csv",
    index=False
)

# ------------------------------------------------------------
# Common figure settings
# ------------------------------------------------------------

layer_labels = {
    "L0":
        "L0 — Inorganic / mineral chemistry",
    "L1":
        "L1 — Simple organic chemistry",
    "L2":
        "L2 — Prebiotic building blocks",
    "L3":
        "L3 — Pre-nucleic functional systems",
    "L4":
        "L4 — RNA / ribonucleotide systems",
    "L5":
        "L5 — DNA / deoxyribonucleotide systems"
}

layer_colors = {
    "L0": "#4C78A8",
    "L1": "#F58518",
    "L2": "#54A24B",
    "L3": "#E45756",
    "L4": "#72B7B2",
    "L5": "#B279A2"
}

# ------------------------------------------------------------
# FIGURE 1 — Conceptual framework
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(15, 9)
)

ax.axis("off")

positions = {
    "L0": (0.08, 0.50),
    "L1": (0.25, 0.50),
    "L2": (0.42, 0.50),
    "L3": (0.59, 0.50),
    "L4": (0.76, 0.67),
    "L5": (0.76, 0.33)
}

for layer, position in positions.items():

    ax.text(
        position[0],
        position[1],
        layer_labels[layer],
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.7",
            facecolor=layer_colors[layer],
            edgecolor="black",
            linewidth=1.2
        ),
        transform=ax.transAxes
    )

# Analytical arrows
for a, b in [
    ("L0", "L1"),
    ("L1", "L2"),
    ("L2", "L3"),
    ("L3", "L4"),
    ("L3", "L5")
]:

    x1, y1 = positions[a]
    x2, y2 = positions[b]

    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=2,
            color="#555555"
        )
    )

ax.text(
    0.5,
    0.90,
    "Analytical framework for mapping molecular chemical continuity",
    ha="center",
    fontsize=19,
    fontweight="bold",
    transform=ax.transAxes
)

ax.text(
    0.5,
    0.08,
    "L0–L5 are analytical categories rather than established historical chronology.",
    ha="center",
    fontsize=10.5,
    color="#555555",
    transform=ax.transAxes
)

fig.savefig(
    FIG_DIR /
    "Figure_01_Conceptual_Framework.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# FIGURE 2 — PCA
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12, 9)
)

for layer in layer_order:

    mask = (
        df["Layer_Normalized"]
        == layer
    ).values

    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=55,
        alpha=0.82,
        color=layer_colors[layer],
        edgecolor="white",
        linewidth=0.5,
        label=layer
    )

ax.set_xlabel(
    f"PC1 ({pc1:.2f}% variance)",
    fontsize=12
)

ax.set_ylabel(
    f"PC2 ({pc2:.2f}% variance)",
    fontsize=12
)

ax.set_title(
    "RDKit Descriptor Chemical Space",
    fontsize=18,
    fontweight="bold"
)

ax.legend(
    title="Analytical layer",
    frameon=True
)

ax.grid(
    alpha=0.15
)

fig.savefig(
    FIG_DIR /
    "Figure_02_RDKit_PCA_Chemical_Space.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# FIGURE 3 — Chemical-space density
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12, 9)
)

hb = ax.hexbin(
    X_pca[:, 0],
    X_pca[:, 1],
    gridsize=35,
    mincnt=1,
    cmap="viridis"
)

ax.set_xlabel(
    f"PC1 ({pc1:.2f}%)"
)

ax.set_ylabel(
    f"PC2 ({pc2:.2f}%)"
)

ax.set_title(
    "Density of Molecular Chemical Space",
    fontsize=18,
    fontweight="bold"
)

fig.colorbar(
    hb,
    ax=ax,
    label="Molecular density"
)

fig.savefig(
    FIG_DIR /
    "Figure_03_Chemical_Space_Density.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# FIGURE 4 — Chemical distance vs layer gap
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 8)
)

plot_df = pd.DataFrame(
    {
        "chemical_distance":
            distance_values,
        "layer_gap":
            gap_values
    }
)

for gap in sorted(
    plot_df["layer_gap"].unique()
):

    values = plot_df.loc[
        plot_df["layer_gap"] == gap,
        "chemical_distance"
    ]

    x = np.full(
        len(values),
        gap,
        dtype=float
    )

    jitter = (
        np.random.default_rng(
            SEED + int(gap)
        )
        .normal(
            0,
            0.055,
            len(values)
        )
    )

    ax.scatter(
        x + jitter,
        values,
        s=12,
        alpha=0.15
    )

medians = (
    plot_df
    .groupby("layer_gap")
    ["chemical_distance"]
    .median()
)

ax.plot(
    medians.index,
    medians.values,
    marker="o",
    linewidth=2.5,
    markersize=7,
    label="Median chemical distance"
)

ax.set_xlabel(
    "Absolute analytical layer separation",
    fontsize=12
)

ax.set_ylabel(
    "Chemical distance",
    fontsize=12
)

ax.set_title(
    "Chemical Distance Increases with Layer Separation",
    fontsize=17,
    fontweight="bold"
)

ax.text(
    0.98,
    0.96,
    f"Spearman ρ = {rho:.3f}\nP < 0.001",
    transform=ax.transAxes,
    ha="right",
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.45",
        facecolor="white",
        edgecolor="#BBBBBB"
    )
)

ax.legend()

ax.grid(
    alpha=0.15
)

fig.savefig(
    FIG_DIR /
    "Figure_04_Chemical_Distance_vs_Layer_Gap.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# FIGURE 5 — Full molecular similarity network
# ------------------------------------------------------------

rng = np.random.default_rng(
    SEED
)

angles = np.linspace(
    0,
    2 * np.pi,
    6,
    endpoint=False
)

layer_centers = {}

for i, layer in enumerate(
    layer_order
):

    layer_centers[layer] = np.array(
        [
            3.8 * np.cos(angles[i]),
            3.8 * np.sin(angles[i])
        ]
    )

initial_pos = {}

for layer in layer_order:

    nodes = [
        node
        for node in G.nodes()
        if G.nodes[node]["layer"]
        == layer
    ]

    for node in nodes:

        initial_pos[node] = (
            layer_centers[layer]
            +
            rng.normal(
                0,
                0.85,
                size=2
            )
        )

pos = nx.spring_layout(
    G,
    pos=initial_pos,
    seed=SEED,
    k=0.65,
    iterations=600,
    weight="similarity",
    scale=10
)

fig = plt.figure(
    figsize=(17, 15),
    facecolor="white"
)

ax = fig.add_axes(
    [0.04, 0.06, 0.72, 0.82]
)

# Background layer regions
for layer in layer_order:

    nodes = [
        node
        for node in G.nodes()
        if G.nodes[node]["layer"]
        == layer
    ]

    if len(nodes) > 2:

        coords = np.array(
            [
                pos[node]
                for node in nodes
            ]
        )

        center = coords.mean(
            axis=0
        )

        radius = max(
            1.25,
            np.percentile(
                np.linalg.norm(
                    coords - center,
                    axis=1
                ),
                90
            ) * 1.35
        )

        ax.add_patch(
            plt.Circle(
                center,
                radius,
                facecolor=layer_colors[layer],
                edgecolor=layer_colors[layer],
                alpha=0.035,
                linewidth=2,
                zorder=0
            )
        )

within_edge_list = []
cross_edge_list = []

for u, v in G.edges():

    if (
        G.nodes[u]["layer"]
        ==
        G.nodes[v]["layer"]
    ):

        within_edge_list.append(
            (u, v)
        )

    else:

        cross_edge_list.append(
            (u, v)
        )

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=within_edge_list,
    ax=ax,
    edge_color="#AAB0B6",
    alpha=0.065,
    width=0.45
)

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=cross_edge_list,
    ax=ax,
    edge_color="#343A40",
    alpha=0.13,
    width=0.65
)

bc_values = np.array(
    [
        betweenness[node]
        for node in G.nodes()
    ]
)

if bc_values.max() > 0:

    node_sizes = (
        35
        +
        850
        *
        (
            bc_values
            /
            bc_values.max()
        ) ** 0.55
    )

else:

    node_sizes = np.full(
        n,
        55
    )

for layer in layer_order:

    nodes = [
        node
        for node in G.nodes()
        if G.nodes[node]["layer"]
        == layer
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_color=layer_colors[layer],
        node_size=[
            node_sizes[node]
            for node in nodes
        ],
        alpha=0.92,
        linewidths=0.7,
        edgecolors="white",
        ax=ax
    )

top_bridge_nodes = (
    top_bridges["node"]
    .tolist()
)

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=top_bridge_nodes,
    node_color="none",
    node_size=[
        node_sizes[node] * 1.75
        for node in top_bridge_nodes
    ],
    linewidths=2.4,
    edgecolors="#111111",
    ax=ax
)

for node in top_bridge_nodes:

    x, y = pos[node]

    label = str(
        G.nodes[node]["name"]
    )

    if len(label) > 30:
        label = (
            label[:27]
            + "..."
        )

    label = (
        f"{label}\n"
        f"{G.nodes[node]['layer']} "
        f"BC={betweenness[node]:.3f}"
    )

    ax.annotate(
        label,
        xy=(x, y),
        xytext=(
            x + rng.uniform(
                -0.35,
                0.35
            ),
            y + rng.uniform(
                0.35,
                0.75
            )
        ),
        fontsize=7.2,
        ha="center",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="#777777",
            linewidth=0.7,
            alpha=0.88
        ),
        arrowprops=dict(
            arrowstyle="-",
            color="#666666",
            linewidth=0.65
        ),
        zorder=10
    )

legend_handles = []

for layer in layer_order:

    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=layer_colors[layer],
            markeredgecolor="white",
            markersize=9,
            label=layer_labels[layer]
        )
    )

ax.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(1.015, 0.68),
    frameon=True,
    title="Analytical layer",
    fontsize=9
)

stats_text = (
    "NETWORK SUMMARY\n"
    "────────────────────────\n"
    f"Structures       {n}\n"
    f"Similarity edges {total_edges:,}\n"
    f"Within-layer     {within_edges:,}\n"
    f"Cross-layer      {cross_edges:,}\n"
    f"Cross-layer      {100 * cross_fraction:.1f}%\n"
    f"Tanimoto         ≥ {TANIMOTO_THRESHOLD:.2f}\n"
    f"Morgan            r=2, 2048-bit"
)

ax.text(
    1.015,
    0.37,
    stats_text,
    transform=ax.transAxes,
    fontsize=9,
    family="DejaVu Sans Mono",
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.65",
        facecolor="#F8F9FA",
        edgecolor="#D0D0D0"
    )
)

fig.text(
    0.04,
    0.94,
    "Molecular Similarity Network Across Pre-Nucleic Acid Chemical Space",
    fontsize=21,
    fontweight="bold",
    ha="left"
)

fig.text(
    0.04,
    0.912,
    "188 unique structures • Morgan radius 2 • Tanimoto similarity ≥ 0.20",
    fontsize=11.5,
    color="#555555",
    ha="left"
)

fig.text(
    0.04,
    0.018,
    "Edges represent molecular fingerprint similarity, not historical ancestry. "
    "Layer labels denote analytical categories rather than established evolutionary chronology.",
    fontsize=8.8,
    color="#555555"
)

ax.set_axis_off()

fig.savefig(
    FIG_DIR /
    "Figure_05_Full_Molecular_Similarity_Network.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

# ------------------------------------------------------------
# FIGURE 6 — Observed / expected connectivity
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 8)
)

im = ax.imshow(
    oe_matrix.values,
    cmap="RdBu_r",
    vmin=0,
    vmax=max(
        2,
        np.nanmax(
            oe_matrix.values
        )
    )
)

ax.set_xticks(
    range(6)
)

ax.set_yticks(
    range(6)
)

ax.set_xticklabels(
    layer_order
)

ax.set_yticklabels(
    layer_order
)

for i in range(6):

    for j in range(6):

        value = (
            oe_matrix.iloc[
                i,
                j
            ]
        )

        text = (
            "—"
            if np.isnan(value)
            else f"{value:.2f}"
        )

        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

ax.set_xlabel(
    "Layer"
)

ax.set_ylabel(
    "Layer"
)

ax.set_title(
    "Observed / Expected Molecular Network Connectivity",
    fontsize=17,
    fontweight="bold"
)

fig.colorbar(
    im,
    ax=ax,
    label="Observed / expected edge ratio"
)

fig.savefig(
    FIG_DIR /
    "Figure_06_Layer_Connectivity_Heatmap.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# FIGURE 7 — Cross-layer permutation
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 8)
)

ax.hist(
    permuted_cross_fractions,
    bins=60,
    alpha=0.75,
    edgecolor="white"
)

ax.axvline(
    null_mean,
    linewidth=2.5,
    label="Null mean"
)

ax.axvline(
    observed_cross_fraction,
    linewidth=3,
    linestyle="--",
    label="Observed"
)

ax.axvline(
    expected_exact,
    linewidth=1.8,
    linestyle=":",
    label="Exact random expectation"
)

ax.set_xlabel(
    "Cross-layer edge fraction"
)

ax.set_ylabel(
    "Permutation count"
)

ax.set_title(
    "Permutation Validation of Cross-Layer Connectivity",
    fontsize=17,
    fontweight="bold"
)

ax.text(
    0.98,
    0.95,
    f"10,000 permutations\n"
    f"Observed = {100 * observed_cross_fraction:.2f}%\n"
    f"Null mean = {100 * null_mean:.2f}%\n"
    f"Z = {z_score:.2f}",
    transform=ax.transAxes,
    ha="right",
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="#BBBBBB"
    )
)

ax.legend()

ax.grid(
    alpha=0.15
)

fig.savefig(
    FIG_DIR /
    "Figure_07_Cross_Layer_Permutation_Test.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# FIGURE 8 — Network robustness
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 8)
)

ax.plot(
    robustness_df["multiplier"],
    robustness_df["edges"],
    marker="o",
    linewidth=2.5,
    label="Edges"
)

ax2 = ax.twinx()

ax2.plot(
    robustness_df["multiplier"],
    robustness_df["largest_component"],
    marker="s",
    linewidth=2.5,
    label="Largest component"
)

ax.set_xlabel(
    "Threshold multiplier"
)

ax.set_ylabel(
    "Network edges"
)

ax2.set_ylabel(
    "Largest component size"
)

ax.set_title(
    "Network Robustness to Similarity Threshold",
    fontsize=17,
    fontweight="bold"
)

fig.savefig(
    FIG_DIR /
    "Figure_08_Network_Robustness.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# FIGURE 9 — Bridge centrality
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12, 8)
)

bridge_plot = (
    bridge_df[
        bridge_df[
            "cross_layer_degree"
        ] > 0
    ]
    .sort_values(
        "betweenness",
        ascending=False
    )
    .head(15)
    .sort_values(
        "betweenness"
    )
)

labels = [
    str(x)
    for x in bridge_plot["name"]
]

ax.barh(
    range(
        len(bridge_plot)
    ),
    bridge_plot[
        "betweenness"
    ]
)

ax.set_yticks(
    range(
        len(bridge_plot)
    )
)

ax.set_yticklabels(
    labels,
    fontsize=8.5
)

ax.set_xlabel(
    "Weighted betweenness centrality"
)

ax.set_title(
    "Cross-Layer Bridge Centrality",
    fontsize=17,
    fontweight="bold"
)

ax.grid(
    axis="x",
    alpha=0.15
)

fig.savefig(
    FIG_DIR /
    "Figure_09_Bridge_Centrality.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# FIGURE 10 — Dataset layer composition
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 7)
)

counts = [
    layer_counts[layer]
    for layer in layer_order
]

ax.bar(
    layer_order,
    counts,
    color=[
        layer_colors[layer]
        for layer in layer_order
    ],
    edgecolor="white"
)

for i, value in enumerate(
    counts
):

    ax.text(
        i,
        value + max(counts) * 0.015,
        str(value),
        ha="center",
        fontweight="bold"
    )

ax.set_xlabel(
    "Analytical layer"
)

ax.set_ylabel(
    "Number of unique structures"
)

ax.set_title(
    "Final Dataset Composition by Analytical Layer",
    fontsize=17,
    fontweight="bold"
)

fig.savefig(
    FIG_DIR /
    "Figure_10_Dataset_Layer_Composition.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# FIGURE 11 — Layer network statistics
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 8)
)

x = np.arange(
    len(layer_order)
)

width = 0.38

ax.bar(
    x - width / 2,
    layer_statistics_df[
        "nodes"
    ],
    width,
    label="Nodes"
)

ax.bar(
    x + width / 2,
    layer_statistics_df[
        "mean_degree"
    ],
    width,
    label="Mean degree"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    layer_order
)

ax.set_ylabel(
    "Value"
)

ax.set_title(
    "Layer-Level Network Statistics",
    fontsize=17,
    fontweight="bold"
)

ax.legend()

fig.savefig(
    FIG_DIR /
    "Figure_11_Layer_Network_Statistics.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# FIGURE 12 — RDKit descriptor correlation
# ------------------------------------------------------------

correlation_matrix = (
    descriptor_df
    .corr(
        method="spearman"
    )
)

fig, ax = plt.subplots(
    figsize=(14, 12)
)

im = ax.imshow(
    correlation_matrix.values,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

ax.set_xticks(
    range(
        len(descriptor_names)
    )
)

ax.set_yticks(
    range(
        len(descriptor_names)
    )
)

ax.set_xticklabels(
    descriptor_names,
    rotation=90,
    fontsize=7
)

ax.set_yticklabels(
    descriptor_names,
    fontsize=7
)

ax.set_title(
    "Spearman Correlation Among RDKit Molecular Descriptors",
    fontsize=17,
    fontweight="bold"
)

fig.colorbar(
    im,
    ax=ax,
    label="Spearman correlation"
)

fig.savefig(
    FIG_DIR /
    "Figure_12_RDKit_Descriptor_Correlation.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# FIGURE 13 — Network topology robustness
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 8)
)

ax.plot(
    robustness_df["threshold"],
    robustness_df["components"],
    marker="o",
    linewidth=2.5,
    label="Connected components"
)

ax2 = ax.twinx()

ax2.plot(
    robustness_df["threshold"],
    robustness_df["cross_layer_fraction"],
    marker="s",
    linewidth=2.5,
    label="Cross-layer fraction"
)

ax.set_xlabel(
    "Tanimoto threshold"
)

ax.set_ylabel(
    "Connected components"
)

ax2.set_ylabel(
    "Cross-layer edge fraction"
)

ax.set_title(
    "Network Topology Across Similarity Thresholds",
    fontsize=17,
    fontweight="bold"
)

fig.savefig(
    FIG_DIR /
    "Figure_13_Network_Topology_Robustness.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# Verify figures
# ------------------------------------------------------------

figure_files = sorted(
    FIG_DIR.glob("Figure_*.png")
)




print(
    "\nFigures generated:",
    len(figure_files)
)

for figure in figure_files:

    print(
        "✓",
        figure.name
    )

print("\nPART 4 COMPLETE.")


# ============================================================
# PART 5 — FINAL TABLES + REPORT + REPRODUCIBILITY PACKAGE
# + GITHUB
# ============================================================

print("=" * 72)
print("REPRODUCIBILITY PIPELINE — PART 5")
print("=" * 72)

# ------------------------------------------------------------
# Final layer composition table
# ------------------------------------------------------------

layer_table = pd.DataFrame(
    {
        "Layer": layer_order,
        "Description": [
            layer_labels[x]
            for x in layer_order
        ],
        "N": [
            layer_counts[x]
            for x in layer_order
        ]
    }
)

layer_table[
    "Percent"
] = (
    layer_table["N"]
    /
    n
    *
    100
)

layer_table.to_csv(
    TABLE_DIR /
    "Table_Layer_Composition.csv",
    index=False
)

# ------------------------------------------------------------
# Final network summary table
# ------------------------------------------------------------

network_summary = pd.DataFrame(
    {
        "Metric": [
            "Unique structures",
            "Morgan fingerprint radius",
            "Morgan fingerprint bits",
            "Tanimoto threshold",
            "Network edges",
            "Within-layer edges",
            "Cross-layer edges",
            "Cross-layer fraction",
            "Connected components",
            "Largest component",
            "Spearman rho: chemical distance vs layer gap",
            "Permutation count",
            "Permutation seed",
            "Observed cross-layer fraction",
            "Randomized null mean",
            "Randomized null SD",
            "Permutation Z score",
            "Permutation P enrichment"
        ],
        "Value": [
            n,
            MORGAN_RADIUS,
            MORGAN_SIZE,
            TANIMOTO_THRESHOLD,
            total_edges,
            within_edges,
            cross_edges,
            cross_fraction,
            components,
            largest_component,
            rho,
            PERMUTATIONS,
            SEED,
            observed_cross_fraction,
            null_mean,
            null_sd,
            z_score,
            p_greater
        ]
    }
)

network_summary.to_csv(
    TABLE_DIR /
    "Table_Final_Network_Summary.csv",
    index=False
)

# ------------------------------------------------------------
# Top bridge table
# ------------------------------------------------------------

top_bridges[
    [
        "name",
        "layer",
        "betweenness",
        "cross_layer_degree",
        "degree"
    ]
].to_csv(
    TABLE_DIR /
    "Table_Top_Bridge_Molecules.csv",
    index=False
)

# ------------------------------------------------------------
# Figure 5 statistics
# ------------------------------------------------------------

fig5_statistics = pd.DataFrame(
    {
        "metric": [
            "structures",
            "edges",
            "within_layer_edges",
            "cross_layer_edges",
            "cross_layer_fraction",
            "threshold",
            "Morgan_radius",
            "Morgan_bits"
        ],
        "value": [
            n,
            total_edges,
            within_edges,
            cross_edges,
            cross_fraction,
            TANIMOTO_THRESHOLD,
            MORGAN_RADIUS,
            MORGAN_SIZE
        ]
    }
)

fig5_statistics.to_csv(
    STAT_DIR /
    "Figure_05_Network_Statistics.csv",
    index=False
)

# ------------------------------------------------------------
# Master results JSON
# ------------------------------------------------------------

master_results = {

    "project":
        "Pre-Nucleic Acid Origin of Life",

    "manuscript":
        "Mapping Chemical Continuity Before Nucleic Acids: "
        "A Computational Analysis of Prebiotic Molecular "
        "Chemical Space and Functional Emergence",

    "analysis_note":
        "L0-L5 are analytical categories, not established "
        "historical chronology.",

    "dataset": {
        "unique_structures": int(n),
        "layers": {
            layer:
                int(layer_counts[layer])
            for layer in layer_order
        }
    },

    "rdkit": {
        "descriptor_count":
            int(
                descriptor_df.shape[1]
            ),
        "morgan_radius":
            MORGAN_RADIUS,
        "morgan_bits":
            MORGAN_SIZE
    },

    "pca": {
        "PC1_percent":
            float(pc1),
        "PC2_percent":
            float(pc2),
        "PC1_PC2_percent":
            float(pc12),
        "PC3_percent":
            float(pc3),
        "PC1_PC2_PC3_percent":
            float(pc123)
    },

    "chemical_distance": {
        "spearman_rho":
            float(rho),
        "p_value":
            float(p_value)
    },

    "network": {
        "threshold":
            TANIMOTO_THRESHOLD,
        "nodes":
            int(n),
        "edges":
            int(total_edges),
        "within_layer_edges":
            int(within_edges),
        "cross_layer_edges":
            int(cross_edges),
        "cross_layer_fraction":
            float(cross_fraction),
        "components":
            int(components),
        "largest_component":
            int(largest_component)
    },

    "permutation_validation": {
        "permutations":
            PERMUTATIONS,
        "seed":
            SEED,
        "observed_cross_layer_fraction":
            float(observed_cross_fraction),
        "exact_random_expected":
            float(expected_exact),
        "null_mean":
            float(null_mean),
        "null_sd":
            float(null_sd),
        "enrichment_ratio":
            float(enrichment_ratio),
        "enrichment_percent":
            float(enrichment_percent),
        "z_score":
            float(z_score),
        "p_greater":
            float(p_greater),
        "p_less":
            float(p_less)
    },

    "robustness": (
        robustness_df
        .to_dict(
            orient="records"
        )
    ),

    "top_bridge_molecules": (
        top_bridges[
            [
                "name",
                "layer",
                "betweenness",
                "cross_layer_degree"
            ]
        ]
        .to_dict(
            orient="records"
        )
    )
}

with open(
    STAT_DIR /
    "master_results.json",
    "w"
) as handle:

    json.dump(
        master_results,
        handle,
        indent=2
    )

# ------------------------------------------------------------
# Reproducibility report
# ------------------------------------------------------------

report = f"""
PRE-NUCLEIC ACID ORIGIN OF LIFE
REPRODUCIBILITY REPORT
===============================================

Manuscript
----------
Mapping Chemical Continuity Before Nucleic Acids:
A Computational Analysis of Prebiotic Molecular
Chemical Space and Functional Emergence

ANALYTICAL NOTE
---------------
L0-L5 are analytical categories rather than
established historical chronology.

DATASET
-------
Final unique structures: {n}

Layer composition:
L0 = {layer_counts["L0"]}
L1 = {layer_counts["L1"]}
L2 = {layer_counts["L2"]}
L3 = {layer_counts["L3"]}
L4 = {layer_counts["L4"]}
L5 = {layer_counts["L5"]}

RDKit
-----
Descriptors: {descriptor_df.shape[1]}

Morgan fingerprint:
radius = {MORGAN_RADIUS}
bits = {MORGAN_SIZE}

PCA
---
PC1 = {pc1:.4f}%
PC2 = {pc2:.4f}%
PC1 + PC2 = {pc12:.4f}%
PC3 = {pc3:.4f}%
PC1 + PC2 + PC3 = {pc123:.4f}%

CHEMICAL DISTANCE
-----------------
Spearman rho = {rho:.8f}

P-value:
{"< 0.001" if p_value == 0 or p_value < 0.001 else f"{p_value:.6g}"}

TANIMOTO NETWORK
----------------
Threshold = {TANIMOTO_THRESHOLD}

Nodes = {n}
Edges = {total_edges}
Within-layer edges = {within_edges}
Cross-layer edges = {cross_edges}
Cross-layer fraction = {100 * cross_fraction:.4f}%

Connected components = {components}
Largest component = {largest_component}

PERMUTATION VALIDATION
----------------------
Permutations = {PERMUTATIONS}
Seed = {SEED}

Observed cross-layer fraction =
{100 * observed_cross_fraction:.4f}%

Exact random expectation =
{100 * expected_exact:.4f}%

Null mean =
{100 * null_mean:.4f}%

Null SD =
{100 * null_sd:.4f}%

Enrichment ratio =
{enrichment_ratio:.6f}x

Enrichment percentage =
{enrichment_percent:.4f}%

Z =
{z_score:.6f}

One-sided enrichment P =
{p_greater:.6g}

Interpretation
--------------
The observed cross-layer connectivity is lower
than expected under random reassignment of the
analytical layer labels.

Therefore the current permutation test does NOT
support a claim of statistically enriched
cross-layer connectivity.

The result is consistent with stronger within-layer
organization than expected under random labels.

ROBUSTNESS
----------
Threshold multipliers:
0.80, 0.90, 1.00, 1.10, 1.20

BRIDGE ANALYSIS
---------------
Weighted betweenness centrality was calculated using
chemical distance = 1 - Tanimoto similarity.

Top cross-layer bridge candidates are stored in:
Table_Top_Bridge_Molecules.csv

FIGURES
-------
13 publication figures were generated.

IMPORTANT INTERPRETATION
------------------------
Chemical similarity and network connectivity do not
demonstrate historical ancestry, abiogenesis, or a
unique prebiotic pathway.

The analysis maps present-day molecular chemical
relationships among the defined analytical categories.
"""

with open(
    REPORT_DIR /
    "FINAL_REPRODUCIBILITY_REPORT.txt",
    "w"
) as handle:

    handle.write(
        report.strip()
        + "\n"
    )

# ------------------------------------------------------------
# Requirements file
# ------------------------------------------------------------

requirements = """numpy
pandas
scipy
scikit-learn
matplotlib
networkx
rdkit
"""

with open(
    FINAL_ANALYSIS /
    "requirements.txt",
    "w"
) as handle:

    handle.write(
        requirements
    )

# ------------------------------------------------------------
# Create standalone reproducibility README
# ------------------------------------------------------------

readme = f"""# Pre-Nucleic Acid Origin of Life

## Reproducibility Pipeline

**Manuscript**

> Mapping Chemical Continuity Before Nucleic Acids:
> A Computational Analysis of Prebiotic Molecular
> Chemical Space and Functional Emergence

## Final analysis

The current pipeline analyzes the finalized molecular
structure dataset and reproduces the complete
structure-based analysis.

Final dataset:

- 188 unique molecular structures
- 20 RDKit molecular descriptors
- Standardized descriptor PCA
- Morgan fingerprints, radius 2
- 2048-bit fingerprints
- Tanimoto similarity network
- Similarity threshold: 0.20
- Layer connectivity
- 10,000-label permutation validation
- Network threshold robustness
- Weighted bridge centrality
- 13 publication figures

## Analytical layers

L0–L5 are analytical categories and should not be
interpreted as established historical chronology.

## Key PCA results

PC1: {pc1:.4f}%

PC2: {pc2:.4f}%

PC1 + PC2: {pc12:.4f}%

PC3: {pc3:.4f}%

PC1 + PC2 + PC3: {pc123:.4f}%

## Molecular similarity network

Nodes: {n}

Edges: {total_edges}

Within-layer edges: {within_edges}

Cross-layer edges: {cross_edges}

Cross-layer fraction: {100 * cross_fraction:.2f}%

Connected components: {components}

Largest component: {largest_component}

## Chemical distance

Spearman rho between chemical distance and
absolute analytical layer separation:

{rho:.8f}

## Permutation validation

10,000 permutations were performed with fixed
molecular network topology and randomly permuted
layer labels.

Randomization seed:

`{SEED}`

Observed cross-layer fraction:

`{100 * observed_cross_fraction:.2f}%`

Exact random expectation:

`{100 * expected_exact:.2f}%`

Null mean:

`{100 * null_mean:.2f}%`

Null SD:

`{100 * null_sd:.2f}%`

Z:

`{z_score:.4f}`

The observed cross-layer connectivity is depleted
relative to random layer assignment.

Therefore this test does not support a claim of
cross-layer enrichment.

## Robustness

Similarity thresholds were evaluated at:

- 0.80 × baseline
- 0.90 × baseline
- 1.00 × baseline
- 1.10 × baseline
- 1.20 × baseline

## Interpretation

This computational analysis describes molecular
similarity and organization within a defined chemical
dataset.

Network connectivity cannot by itself demonstrate
historical ancestry, abiogenesis, or a unique
prebiotic pathway.

## Reproduction

Run:

```text
reproducibility_pipeline.py





