"""
Count papers relevant to a Cognitive Atlas term in neuroimaging databases.

This script uses NiMARE to query:
- Neurosynth: ~14k studies with coordinate-based meta-analysis data
- NeuroQuery: ~13k studies optimized for predictive brain mapping
- BrainMap: Semi-closed database (requires registration)

The script downloads the Cognitive Atlas ontology and counts papers
mentioning a specific term across these databases.
"""

import os
import re

import numpy as np
import pandas as pd
from nimare import annotate
from nimare.utils import _uk_to_us
from tqdm import tqdm

from datasets import (
    list_cognitive_atlas_terms,
    load_brainmap_dataset,
    load_cognitive_atlas,
    load_neuroquery_dataset,
    load_neurosynth_dataset,
)


def extract_cogat(text_df, id_df=None, text_column="abstract"):
    """Extract Cognitive Atlas terms and count instances using regular expressions.

    Parameters
    ----------
    text_df : (D x 2) :obj:`pandas.DataFrame`
        Pandas dataframe with at least two columns: 'id' and the text.
        D = document.
    id_df : (T x 3) :obj:`pandas.DataFrame`
        Cognitive Atlas ontology dataframe with at least three columns:
        'id' (unique identifier for term), 'alias' (natural language expression
        of term), and 'name' (preferred name of term; currently unused).
        T = term.
    text_column : :obj:`str`, optional
        Name of column in text_df that contains text. Default is 'abstract'.

    Returns
    -------
    counts_df : (D x T) :obj:`pandas.DataFrame`
        Term counts for documents in the corpus.
    rep_text_df : (D x 2) :obj:`pandas.DataFrame`
        Text DataFrame with terms replaced with their CogAt IDs.

    References
    ----------
    * Poldrack, Russell A., et al. "The cognitive atlas: toward a
      knowledge foundation for cognitive neuroscience." Frontiers in
      neuroinformatics 5 (2011): 17. https://doi.org/10.3389/fninf.2011.00017
    """
    text_df = text_df.copy()
    if id_df is None:
        cogat = download_cognitive_atlas()
        id_df = pd.read_csv(cogat["ids"])
    gazetteer = sorted(id_df["id"].unique().tolist())
    if "id" in text_df.columns:
        text_df.set_index("id", inplace=True)

    text_df[text_column] = text_df[text_column].fillna("")
    text_df[text_column] = text_df[text_column].apply(_uk_to_us)

    # Create regex dictionary
    regex_dict = {}
    for term in id_df["alias"].values:
        term_for_regex = term.replace("(", r"\(").replace(")", r"\)")
        regex = "\\b" + term_for_regex + "\\b"
        pattern = re.compile(regex, re.MULTILINE | re.IGNORECASE)
        regex_dict[term] = pattern

    # Count
    count_arr = np.zeros((text_df.shape[0], len(gazetteer)), int)
    counts_df = pd.DataFrame(columns=gazetteer, index=text_df.index, data=count_arr)
    for term_idx in tqdm(id_df.index):
        term = id_df["alias"].loc[term_idx]
        term_id = id_df["id"].loc[term_idx]
        pattern = regex_dict[term]
        counts_df[term_id] += text_df[text_column].str.count(pattern).astype(int)
        text_df[text_column] = text_df[text_column].str.replace(
            pattern, term_id, regex=True
        )

    return counts_df, text_df


def count_papers_for_term_from_annotations(dset, term):
    """
    Count papers containing a term using the dataset's existing annotations.

    This works with the pre-computed term frequencies in Neurosynth.

    Parameters
    ----------
    dset : nimare.dataset.Dataset
        Dataset with annotations
    term : str
        The term to search for

    Returns
    -------
    int
        Number of papers with non-zero annotation for the term
    """
    if dset.annotations is None or dset.annotations.empty:
        return 0

    # Find columns that match the term (annotations use format like "terms_abstract__term")
    matching_cols = [
        col for col in dset.annotations.columns if term.lower() in col.lower()
    ]

    if not matching_cols:
        return 0

    # Count studies with non-zero values for any matching term column
    count = 0
    for col in matching_cols:
        count += (dset.annotations[col] > 0).sum()

    return count


def get_or_create_cogat_counts(
    dset, dataset_name, id_df, rel_df, data_dir="data", expand_hierarchy=True
):
    """
    Get cached Cognitive Atlas term counts or compute and cache them.

    Parameters
    ----------
    dset : nimare.dataset.Dataset
        Dataset with abstracts in dset.texts
    dataset_name : str
        Name of the dataset (used for cache filename)
    id_df : pd.DataFrame
        Cognitive Atlas IDs dataframe
    rel_df : pd.DataFrame
        Cognitive Atlas relationships dataframe
    data_dir : str
        Directory to store cache files
    expand_hierarchy : bool
        Whether to expand counts using ontology relationships

    Returns
    -------
    pd.DataFrame or None
        Term counts DataFrame, or None if abstracts unavailable
    """
    if dset.texts is None or "abstract" not in dset.texts.columns:
        return None

    cache_filename = f"{dataset_name.lower()}_{expand_hierarchy}_cogat_counts.parquet"
    cache_path = os.path.join(data_dir, cache_filename)

    if os.path.exists(cache_path):
        print(f"Loading cached Cognitive Atlas counts for {dataset_name}...")
        cogat_counts_df = pd.read_parquet(cache_path)
    else:
        print(
            f"Extracting Cognitive Atlas terms for {dataset_name} (this may take ~15 minutes)..."
        )
        cogat_counts_df, _ = extract_cogat(dset.texts, id_df, text_column="abstract")

        if expand_hierarchy:
            weights = {"isKindOf": 1, "isPartOf": 1, "inCategory": 1}
            cogat_counts_df = annotate.cogat.expand_counts(
                cogat_counts_df, rel_df, weights
            )

        # Cache the results
        os.makedirs(data_dir, exist_ok=True)
        cogat_counts_df.to_parquet(cache_path)
        print(f"Cached counts to {cache_path}")

    return cogat_counts_df


def count_papers_for_cogat_term(cogat_counts_df, term):
    """
    Count papers mentioning a Cognitive Atlas term from pre-computed counts.

    Parameters
    ----------
    cogat_counts_df : pd.DataFrame
        Pre-computed Cognitive Atlas term counts
    term : str
        The Cognitive Atlas term to search for

    Returns
    -------
    int
        Number of papers mentioning the term
    """
    if cogat_counts_df is None:
        return 0

    # Find the column matching the term
    matching_cols = [
        col for col in cogat_counts_df.columns if term.lower() in col.lower()
    ]

    if not matching_cols:
        print(f"Term '{term}' not found in Cognitive Atlas.")
        return 0

    # Count papers with non-zero mentions
    count = 0
    for col in matching_cols:
        count += (cogat_counts_df[col] > 0).sum()

    return count


def get_or_create_embeddings(dset, dataset_name, data_dir="data"):
    """
    Get cached abstract embeddings or compute and cache them.

    Uses sentence-transformers all-MiniLM-L6-v2 model.

    Parameters
    ----------
    dset : nimare.dataset.Dataset
        Dataset with abstracts in dset.texts
    dataset_name : str
        Name of the dataset (used for cache filename)
    data_dir : str
        Directory to store cache files

    Returns
    -------
    np.ndarray or None
        Embeddings array of shape (n_documents, 384), or None if abstracts unavailable
    pd.Index or None
        Index of document IDs corresponding to embeddings
    """
    if dset.texts is None or "abstract" not in dset.texts.columns:
        return None, None

    cache_filename = f"{dataset_name.lower()}_embeddings.npy"
    index_filename = f"{dataset_name.lower()}_embeddings_index.npy"
    cache_path = os.path.join(data_dir, cache_filename)
    index_path = os.path.join(data_dir, index_filename)

    if os.path.exists(cache_path) and os.path.exists(index_path):
        print(f"Loading cached embeddings for {dataset_name}...")
        embeddings = np.load(cache_path)
        index = np.load(index_path, allow_pickle=True)
        return embeddings, pd.Index(index)

    print(f"Computing embeddings for {dataset_name} (this may take a few minutes)...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts_df = dset.texts.copy()
    if "id" in texts_df.columns:
        texts_df = texts_df.set_index("id")

    # Fill NaN and get abstracts
    abstracts = texts_df["abstract"].fillna("").tolist()
    index = texts_df.index

    # Compute embeddings with progress bar
    embeddings = model.encode(abstracts, show_progress_bar=True, convert_to_numpy=True)

    # Cache the results
    os.makedirs(data_dir, exist_ok=True)
    np.save(cache_path, embeddings)
    np.save(index_path, index.to_numpy())
    print(f"Cached embeddings to {cache_path}")

    return embeddings, index


def count_papers_by_embedding_similarity(
    embeddings, index, texts_df, term, threshold=0.2, output_csv=None
):
    """
    Count papers with embedding similarity above threshold for a term.

    Parameters
    ----------
    embeddings : np.ndarray
        Pre-computed abstract embeddings of shape (n_documents, embedding_dim)
    index : pd.Index
        Document IDs corresponding to embeddings
    texts_df : pd.DataFrame
        DataFrame with abstracts (must have 'abstract' column)
    term : str
        The term to search for
    threshold : float
        Similarity threshold (default 0.2)
    output_csv : str, optional
        Path to save CSV with abstracts and scores

    Returns
    -------
    int
        Number of papers with similarity above threshold
    """
    if embeddings is None:
        return 0

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create query with neuroscience context
    query = f"{term} in the context of neuroscience"
    query_embedding = model.encode([query], convert_to_numpy=True)[0]

    # Compute cosine similarities
    # Normalize embeddings for cosine similarity
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_norm = query_embedding / np.linalg.norm(query_embedding)

    similarities = embeddings_norm @ query_norm

    # Count papers above threshold
    count = (similarities > threshold).sum()

    # Save CSV if requested
    if output_csv:
        # Build results dataframe
        texts_indexed = texts_df.copy()
        if "id" in texts_indexed.columns:
            texts_indexed = texts_indexed.set_index("id")

        results_df = pd.DataFrame(
            {
                "id": index,
                "similarity": similarities,
            }
        )
        results_df = results_df.set_index("id")

        # Join with abstracts
        results_df = results_df.join(texts_indexed[["abstract"]], how="left")
        results_df = results_df.sort_values("similarity", ascending=False)
        results_df = results_df.reset_index()

        results_df.to_csv(output_csv, index=False)
        print(f"Saved similarity scores to {output_csv}")

    return int(count)


def _count_for_dataset(dset, name, term, cogat_counts_df, use_annotations):
    """Count papers for a term in a single dataset."""
    print(f"\n--- {name} ---")
    print(f"{name} contains {len(dset.ids)} studies")

    if use_annotations:
        count = count_papers_for_term_from_annotations(dset, term)
    else:
        if cogat_counts_df is None:
            print("Warning: No abstracts available. Using annotation-based counting.")
            count = count_papers_for_term_from_annotations(dset, term)
        else:
            count = count_papers_for_cogat_term(cogat_counts_df, term)

    print(f"Papers mentioning '{term}': {count}")
    return count


def get_paper_counts(
    term,
    data_dir="data",
    brainmap_sleuth_file=None,
    pubmed_email=None,
    use_annotations=True,
    include_neuroquery=True,
    use_embeddings=False,
    embedding_threshold=0.2,
    output_csv_prefix=None,
):
    """
    Get the number of papers relevant to a Cognitive Atlas term across databases.

    Parameters
    ----------
    term : str
        The Cognitive Atlas term to search for (e.g., "working memory", "attention")
    data_dir : str
        Directory to store/load data
    brainmap_sleuth_file : str, optional
        Path to BrainMap Sleuth export file. If None, BrainMap count will be skipped.
    pubmed_email : str, optional
        Email for PubMed API to download abstracts
    use_annotations : bool
        If True, use pre-computed annotations (faster). If False, extract from abstracts.
    include_neuroquery : bool
        If True, query NeuroQuery database for papers.
    use_embeddings : bool
        If True, use sentence-transformer embeddings for semantic similarity search.
    embedding_threshold : float
        Similarity threshold for embedding-based search (default 0.2).
    output_csv_prefix : str, optional
        Prefix for CSV output files with similarity scores (e.g., "results" creates
        "results_neurosynth.csv", "results_neuroquery.csv", etc.)

    Returns
    -------
    dict
        Dictionary with counts for each database
    """
    results = {"neurosynth": 0, "neuroquery": None, "brainmap": None}

    # Only load Cognitive Atlas if needed for text extraction (not for embeddings)
    id_df, rel_df = None, None
    if not use_annotations and not use_embeddings:
        print("Loading Cognitive Atlas...")
        id_df, rel_df = load_cognitive_atlas(data_dir)

        # Check if term exists in Cognitive Atlas (exact match, case-insensitive)
        term_matches = id_df[id_df["name"].str.lower() == term.lower()]
        if term_matches.empty:
            print(f"Warning: '{term}' not found in Cognitive Atlas.")
        else:
            print(f"Found Cognitive Atlas term: {term_matches['name'].tolist()[0]}")

    # Neurosynth (always included)
    neurosynth_dset = load_neurosynth_dataset(data_dir, pubmed_email)
    print(f"\n--- Neurosynth ---")
    print(f"Neurosynth contains {len(neurosynth_dset.ids)} studies")

    if use_embeddings:
        embeddings, index = get_or_create_embeddings(
            neurosynth_dset, "neurosynth", data_dir
        )
        output_csv = (
            f"{output_csv_prefix}_neurosynth.csv" if output_csv_prefix else None
        )
        results["neurosynth"] = count_papers_by_embedding_similarity(
            embeddings,
            index,
            neurosynth_dset.texts,
            term,
            embedding_threshold,
            output_csv,
        )
        print(
            f"Papers similar to '{term}' (threshold={embedding_threshold}): {results['neurosynth']}"
        )
    else:
        neurosynth_counts = None
        if not use_annotations:
            neurosynth_counts = get_or_create_cogat_counts(
                neurosynth_dset, "neurosynth", id_df, rel_df, data_dir
            )
        results["neurosynth"] = _count_for_dataset(
            neurosynth_dset, "Neurosynth", term, neurosynth_counts, use_annotations
        )

    # NeuroQuery (optional)
    if include_neuroquery:
        neuroquery_dset = load_neuroquery_dataset(data_dir, pubmed_email)
        print(f"\n--- NeuroQuery ---")
        print(f"NeuroQuery contains {len(neuroquery_dset.ids)} studies")

        if use_embeddings:
            embeddings, index = get_or_create_embeddings(
                neuroquery_dset, "neuroquery", data_dir
            )
            output_csv = (
                f"{output_csv_prefix}_neuroquery.csv" if output_csv_prefix else None
            )
            results["neuroquery"] = count_papers_by_embedding_similarity(
                embeddings,
                index,
                neuroquery_dset.texts,
                term,
                embedding_threshold,
                output_csv,
            )
            print(
                f"Papers similar to '{term}' (threshold={embedding_threshold}): {results['neuroquery']}"
            )
        else:
            neuroquery_counts = None
            if not use_annotations:
                neuroquery_counts = get_or_create_cogat_counts(
                    neuroquery_dset, "neuroquery", id_df, rel_df, data_dir
                )
            results["neuroquery"] = _count_for_dataset(
                neuroquery_dset, "NeuroQuery", term, neuroquery_counts, use_annotations
            )

    # BrainMap (requires file)
    if brainmap_sleuth_file:
        try:
            brainmap_dset = load_brainmap_dataset(brainmap_sleuth_file)
            print(f"\n--- BrainMap ---")
            print(f"BrainMap contains {len(brainmap_dset.ids)} studies")

            if use_embeddings:
                embeddings, index = get_or_create_embeddings(
                    brainmap_dset, "brainmap", data_dir
                )
                output_csv = (
                    f"{output_csv_prefix}_brainmap.csv" if output_csv_prefix else None
                )
                results["brainmap"] = count_papers_by_embedding_similarity(
                    embeddings,
                    index,
                    brainmap_dset.texts,
                    term,
                    embedding_threshold,
                    output_csv,
                )
                print(
                    f"Papers similar to '{term}' (threshold={embedding_threshold}): {results['brainmap']}"
                )
            else:
                brainmap_counts = None
                if not use_annotations:
                    brainmap_counts = get_or_create_cogat_counts(
                        brainmap_dset, "brainmap", id_df, rel_df, data_dir
                    )
                results["brainmap"] = _count_for_dataset(
                    brainmap_dset, "BrainMap", term, brainmap_counts, use_annotations
                )
        except FileNotFoundError as e:
            print(f"\n--- BrainMap ---\n{e}")
    else:
        print("\n--- BrainMap ---")
        print("Skipped (no Sleuth file provided). Register at brainmap.org for access.")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Count papers for a Cognitive Atlas term in neuroimaging databases"
    )
    parser.add_argument(
        "term",
        type=str,
        nargs="?",
        default="working memory",
        help="Cognitive Atlas term to search for (default: 'working memory')",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store/load data (default: 'data')",
    )
    parser.add_argument(
        "--brainmap-file",
        type=str,
        default=None,
        help="Path to BrainMap Sleuth export file",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Email for PubMed API (required to download abstracts)",
    )
    parser.add_argument(
        "--list-terms",
        action="store_true",
        help="List available Cognitive Atlas terms and exit",
    )
    parser.add_argument(
        "--search-terms",
        type=str,
        default=None,
        help="Search for terms containing this string",
    )
    parser.add_argument(
        "--no-neuroquery", action="store_true", help="Skip NeuroQuery query"
    )
    parser.add_argument(
        "--extract-from-abstracts",
        action="store_true",
        help="Extract terms from abstracts using Cognitive Atlas (slower, requires --email)",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Use sentence-transformer embeddings for semantic similarity search (requires --email)",
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.2,
        help="Similarity threshold for embedding-based search (default: 0.2)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Prefix for CSV output files with similarity scores (e.g., 'results' creates 'results_neurosynth.csv')",
    )

    args = parser.parse_args()

    if args.list_terms or args.search_terms:
        terms = list_cognitive_atlas_terms(args.data_dir, args.search_terms)
        print(f"Found {len(terms)} terms:")
        for t in terms[:50]:  # Limit output
            print(f"  - {t}")
        if len(terms) > 50:
            print(f"  ... and {len(terms) - 50} more")
    else:
        results = get_paper_counts(
            term=args.term,
            data_dir=args.data_dir,
            brainmap_sleuth_file=args.brainmap_file,
            pubmed_email=args.email,
            include_neuroquery=not args.no_neuroquery,
            use_annotations=not args.extract_from_abstracts,
            use_embeddings=args.use_embeddings,
            embedding_threshold=args.embedding_threshold,
            output_csv_prefix=args.output_csv,
        )

        print("\n" + "=" * 50)
        print(f"Results for term: '{args.term}'")
        if args.use_embeddings:
            print(
                f"Method: Embedding similarity (threshold={args.embedding_threshold})"
            )
        elif args.extract_from_abstracts:
            print("Method: Cognitive Atlas extraction")
        else:
            print("Method: Pre-computed annotations")
        print("=" * 50)
        print(f"Neurosynth:  {results['neurosynth']} papers")
        if results["neuroquery"] is not None:
            print(f"NeuroQuery:  {results['neuroquery']} papers")
        else:
            print("NeuroQuery:  N/A (skipped)")
        if results["brainmap"] is not None:
            print(f"BrainMap:    {results['brainmap']} papers")
        else:
            print("BrainMap:    N/A (Sleuth file not provided)")
