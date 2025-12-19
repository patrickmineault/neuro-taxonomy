"""
Count relevant papers for each term in brains_vs_ai.csv using different methods.

This script iterates through the brains_vs_ai.csv file and counts papers in
Neurosynth and NeuroQuery databases using:
1. Annotation-based keyword search (use_annotations=True)
2. Cognitive Atlas extraction from abstracts (for ca_concept only)
3. Embedding-based semantic similarity search (use_embeddings=True)

For each row, it counts papers using both the 'function' column and 'ca_concept' column.
Rows where ca_concept is 'NA' are skipped for ca_concept counting.

Note: Cognitive Atlas (cogat) method only applies to ca_concept, not function,
since cogat terms are specifically from the Cognitive Atlas ontology.
"""

import pandas as pd
from neuroquery import NeuroQueryModel, encoding, fetch_neuroquery_model
from tqdm import tqdm

from cogat_paper_count import (
    count_papers_by_embedding_similarity,
    get_or_create_cogat_counts,
    get_or_create_embeddings,
)
from datasets import (
    load_cognitive_atlas,
    load_neuroquery_dataset,
    load_neurosynth_dataset,
)

encoding._MAX_SIMILAR_DOCS_RETURNED = 10000  # Increase max similar docs returned


def count_papers_for_term_from_annotations_exact(dset, term):
    """
    Count papers containing a term using exact match on the dataset's existing annotations.

    Unlike count_papers_for_term_from_annotations which uses substring matching,
    this function requires an exact match on the term portion of the column name.
    For example, searching for "inhibition" will NOT match "response inhibition".

    Parameters
    ----------
    dset : nimare.dataset.Dataset
        Dataset with annotations
    term : str
        The term to search for (exact match)

    Returns
    -------
    int
        Number of papers with non-zero annotation for the term
    """
    if dset.annotations is None or dset.annotations.empty:
        return 0

    # Normalize the search term
    term_normalized = term.lower().strip()

    # Find columns that exactly match the term
    # Annotation columns have format like "terms_abstract__term" or "terms_abstract_tfidf__term"
    matching_cols = []
    for col in dset.annotations.columns:
        # Extract the term part after the last "__"
        if "__" in col:
            col_term = col.split("__")[-1].lower().strip()
            if col_term == term_normalized:
                matching_cols.append(col)

    if not matching_cols:
        return 0

    # Count studies with non-zero values for any matching term column
    count = 0
    for col in matching_cols:
        count += (dset.annotations[col] > 0).sum()

    return count


def count_papers_for_term_from_annotations_exact_list(dset, terms):
    """
    Count papers containing any term from a list using exact match.

    Parameters
    ----------
    dset : nimare.dataset.Dataset
        Dataset with annotations
    terms : list of str
        List of terms to search for (exact match)

    Returns
    -------
    int
        Number of papers with non-zero annotation for any of the terms
    """
    if dset.annotations is None or dset.annotations.empty:
        return 0

    # Normalize the search terms
    terms_normalized = [term.lower().strip() for term in terms]

    # Find columns that exactly match any term
    matching_cols = []
    for col in dset.annotations.columns:
        if "__" in col:
            col_term = col.split("__")[-1].lower().strip()
            if col_term in terms_normalized:
                matching_cols.append(col)

    if not matching_cols:
        return 0

    # Count studies with non-zero values for any matching term column
    count = 0
    for col in matching_cols:
        count += (dset.annotations[col] > 0).sum()

    return count


def count_papers_for_cogat_term_by_name(cogat_counts_df, id_df, term, strict):
    """
    Count papers mentioning a Cognitive Atlas term by looking up the term name.

    This function first finds the Cognitive Atlas ID(s) for the given term name,
    then counts papers with non-zero values for those IDs in the counts dataframe.

    Parameters
    ----------
    cogat_counts_df : pd.DataFrame
        Pre-computed Cognitive Atlas term counts (columns are CogAt IDs like trm_xxx)
    id_df : pd.DataFrame
        Cognitive Atlas IDs dataframe with 'id', 'name', and 'alias' columns
    term : str
        The Cognitive Atlas term name to search for (e.g., "working memory")

    Returns
    -------
    int
        Number of papers mentioning the term
    """
    if cogat_counts_df is None:
        return 0

    # Normalize the search term
    term_normalized = term.lower().strip()

    # Find matching term IDs by exact match on name or alias
    matching_ids = set()

    # Check name column (exact match)
    name_matches = id_df[id_df["name"].str.lower().str.strip() == term_normalized]
    matching_ids.update(name_matches["id"].tolist())

    # Check alias column (exact match)
    if not matching_ids or not strict:
        alias_matches = id_df[id_df["alias"].str.lower().str.strip() == term_normalized]
        matching_ids.update(alias_matches["id"].tolist())

    if not matching_ids:
        return 0

    # Find columns in cogat_counts_df that match these IDs
    matching_cols = [col for col in cogat_counts_df.columns if col in matching_ids]

    if not matching_cols:
        return 0

    if strict:
        assert (
            len(matching_cols) <= 1
        ), "Multiple matching Cognitive Atlas IDs found for the term."

    # Count papers with non-zero mentions (avoid double counting across columns)
    # Use logical OR across all matching columns
    mask = cogat_counts_df[matching_cols].gt(0).any(axis=1)
    count = mask.sum()

    return int(count)


def count_all_terms(
    csv_path="brains_vs_ai.csv",
    output_path="brains_vs_ai_paper_counts.csv",
    data_dir="data",
    pubmed_email=None,
    embedding_threshold=0.5,
):
    """
    Count papers for all terms in the CSV file using different methods.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file with 'function' and 'ca_concept' columns
    output_path : str
        Path to save the output CSV file with counts
    data_dir : str
        Directory to store/load data
    pubmed_email : str, optional
        Email for PubMed API to download abstracts
    embedding_threshold : float
        Similarity threshold for embedding-based search (default: 0.5)

    Returns
    -------
    pd.DataFrame
        DataFrame with all counts
    """
    # Load the input CSV
    input_df = pd.read_csv(csv_path)
    print(f"Loaded {len(input_df)} rows from {csv_path}")

    # Load datasets
    print("\n" + "=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    neurosynth_dset = load_neurosynth_dataset(data_dir, pubmed_email)
    neuroquery_dset = load_neuroquery_dataset(data_dir, pubmed_email)

    # Load Cognitive Atlas for text extraction
    print("\nLoading Cognitive Atlas...")
    id_df, rel_df = load_cognitive_atlas(data_dir)

    # Get or create Cognitive Atlas counts
    print("\n" + "=" * 60)
    print("Preparing Cognitive Atlas counts...")
    print("=" * 60)

    # Also get expanded counts with hierarchical expansion
    neurosynth_cogat_counts_expanded = get_or_create_cogat_counts(
        neurosynth_dset,
        "neurosynth_expanded",
        id_df,
        rel_df,
        data_dir,
        expand_hierarchy=True,
    )
    neuroquery_cogat_counts_expanded = get_or_create_cogat_counts(
        neuroquery_dset,
        "neuroquery_expanded",
        id_df,
        rel_df,
        data_dir,
        expand_hierarchy=True,
    )

    neurosynth_cogat_counts = get_or_create_cogat_counts(
        neurosynth_dset,
        "neurosynth",
        id_df,
        rel_df,
        data_dir,
        expand_hierarchy=False,
    )
    neuroquery_cogat_counts = get_or_create_cogat_counts(
        neuroquery_dset,
        "neuroquery",
        id_df,
        rel_df,
        data_dir,
        expand_hierarchy=False,
    )

    # Get or create embeddings
    print("\n" + "=" * 60)
    print("Preparing embeddings...")
    print("=" * 60)
    neurosynth_embeddings, neurosynth_index = get_or_create_embeddings(
        neurosynth_dset, "neurosynth", data_dir
    )
    neuroquery_embeddings, neuroquery_index = get_or_create_embeddings(
        neuroquery_dset, "neuroquery", data_dir
    )

    # Load homonyms and NeuroQuery encoder
    df_homonyms = pd.read_csv("data/neurosynth_homonyms.csv")
    encoder = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())

    # Count papers for each row, building list of result dicts
    print("\n" + "=" * 60)
    print("Counting papers for each term...")
    print("=" * 60)

    results = []

    for _, row in tqdm(
        input_df.iterrows(), total=len(input_df), desc="Processing terms"
    ):
        function = row["function"]
        ca_concept = row["ca_concept"]

        # Start with original row data
        result = row.to_dict()

        # Skip if ca_concept is NA
        if pd.isna(ca_concept) or str(ca_concept).strip().upper() == "NA":
            results.append(result)
            continue

        ca_concept_clean = str(ca_concept).strip()

        # 1. Annotation-based (exact match)
        result["ca_concept_neurosynth_annotations_exact"] = (
            count_papers_for_term_from_annotations_exact(
                neurosynth_dset, ca_concept_clean
            )
        )

        # Annotation-based with homonyms
        concept_list = [ca_concept_clean]
        extra_concepts = df_homonyms[df_homonyms["function"] == function]
        if not extra_concepts.empty:
            concept_list += extra_concepts["homonym"].tolist()

        result["ca_concept_neurosynth_annotations_homonyms"] = (
            count_papers_for_term_from_annotations_exact_list(
                neurosynth_dset, concept_list
            )
        )

        result["ca_concept_neuroquery_annotations_exact"] = (
            count_papers_for_term_from_annotations_exact(
                neuroquery_dset, ca_concept_clean
            )
        )

        # 2. Cognitive Atlas extraction (exact lookup by ID)
        result["ca_concept_neurosynth_cogat"] = count_papers_for_cogat_term_by_name(
            neurosynth_cogat_counts, id_df, ca_concept_clean, strict=True
        )
        result["ca_concept_neuroquery_cogat"] = count_papers_for_cogat_term_by_name(
            neuroquery_cogat_counts, id_df, ca_concept_clean, strict=True
        )

        # 3. Cognitive Atlas extraction (with hierarchical expansion)
        result["ca_concept_neurosynth_cogat_expanded"] = (
            count_papers_for_cogat_term_by_name(
                neurosynth_cogat_counts_expanded, id_df, ca_concept_clean, strict=False
            )
        )
        result["ca_concept_neuroquery_cogat_expanded"] = (
            count_papers_for_cogat_term_by_name(
                neuroquery_cogat_counts_expanded, id_df, ca_concept_clean, strict=False
            )
        )

        # 4. NeuroQuery similar docs
        similar_docs = encoder(ca_concept_clean)["similar_documents"]
        result["ca_concept_neuroquery_total_sim"] = similar_docs.similarity.sum()
        result["ca_concept_neuroquery_min_sum"] = (similar_docs.similarity >= 0.1).sum()

        # 5. Embedding-based
        result["ca_concept_neurosynth_embedding"] = (
            count_papers_by_embedding_similarity(
                neurosynth_embeddings,
                neurosynth_index,
                neurosynth_dset.texts,
                ca_concept_clean,
                threshold=embedding_threshold,
            )
        )
        result["ca_concept_neuroquery_embedding"] = (
            count_papers_by_embedding_similarity(
                neuroquery_embeddings,
                neuroquery_index,
                neuroquery_dset.texts,
                ca_concept_clean,
                threshold=embedding_threshold,
            )
        )

        results.append(result)

    # Create DataFrame from results
    df = pd.DataFrame(results)

    # Save results
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Count papers for terms in brains_vs_ai.csv across neuroimaging databases"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/brains_vs_ai.csv",
        help="Path to input CSV file (default: brains_vs_ai.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/brains_vs_ai_paper_counts.csv",
        help="Path to output CSV file (default: brains_vs_ai_paper_counts.csv)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store/load data (default: 'data')",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Email for PubMed API (required to download abstracts for embedding method)",
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for embedding-based search (default: 0.5)",
    )

    args = parser.parse_args()

    results = count_all_terms(
        csv_path=args.input,
        output_path=args.output,
        data_dir=args.data_dir,
        pubmed_email=args.email,
        embedding_threshold=args.embedding_threshold,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total rows processed: {len(results)}")
    print(f"Embedding threshold used: {args.embedding_threshold}")

    # Show sample of results
    print("\nSample results (first 5 rows):")
    cols_to_show = [
        "function",
        "ca_concept",
        "ca_concept_neurosynth_cogat",
        "ca_concept_neurosynth_embedding",
    ]
    available_cols = [c for c in cols_to_show if c in results.columns]
    print(results[available_cols].head().to_string())
