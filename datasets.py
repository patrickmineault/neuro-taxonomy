"""
Dataset loading utilities for neuroimaging databases.

This module provides functions to download and load datasets from:
- Neurosynth: ~14k studies with coordinate-based meta-analysis data
- NeuroQuery: ~13k studies optimized for predictive brain mapping
- BrainMap: Semi-closed database (requires registration)
- Cognitive Atlas: Ontology of cognitive concepts and tasks
"""

import os

import pandas as pd
from nimare import dataset, extract, io


def load_cognitive_atlas(data_dir="data"):
    """
    Download and load the Cognitive Atlas ontology.

    The Cognitive Atlas is an ontology of cognitive science concepts,
    tasks, and their relationships.

    Parameters
    ----------
    data_dir : str
        Directory to store/load data

    Returns
    -------
    tuple
        (id_df, rel_df) - DataFrames containing term IDs and relationships
    """
    cogatlas = extract.download_cognitive_atlas(data_dir=data_dir, overwrite=False)
    id_df = pd.read_csv(cogatlas["ids"])
    rel_df = pd.read_csv(cogatlas["relationships"])
    return id_df, rel_df


def load_neurosynth_dataset(data_dir="data", download_abstracts_email=None):
    """
    Load Neurosynth dataset, downloading if necessary.

    Neurosynth is a platform for large-scale, automated synthesis of
    functional neuroimaging data. Contains ~14,371 studies.

    Parameters
    ----------
    data_dir : str
        Directory to store/load data
    download_abstracts_email : str, optional
        Email for PubMed API to download abstracts. If None, abstracts won't be downloaded.

    Returns
    -------
    nimare.dataset.Dataset
        The Neurosynth dataset
    """
    dataset_path = os.path.join(data_dir, "neurosynth_dataset.pkl.gz")

    if os.path.exists(dataset_path):
        print("Loading existing Neurosynth dataset...")
        dset = dataset.Dataset.load(dataset_path)
    else:
        print("Downloading and converting Neurosynth dataset...")
        files = extract.fetch_neurosynth(
            data_dir=data_dir,
            version="7",
            source="abstract",
            vocab="terms",
            overwrite=False,
        )
        dset = io.convert_neurosynth_to_dataset(
            coordinates_file=files[0]["coordinates"],
            metadata_file=files[0]["metadata"],
            annotations_files=files[0]["features"],
        )
        dset.save(dataset_path)

    # Download abstracts if email provided and not already present
    if download_abstracts_email and (
        dset.texts is None or "abstract" not in dset.texts.columns
    ):
        print("Downloading abstracts from PubMed...")
        dset = extract.download_abstracts(dset, email=download_abstracts_email)
        dset.save(dataset_path)

    return dset


def load_neuroquery_dataset(data_dir="data", download_abstracts_email=None):
    """
    Load NeuroQuery dataset, downloading if necessary.

    NeuroQuery is similar to Neurosynth but uses a different vocabulary and
    is optimized for predictive brain mapping rather than meta-analysis.
    It contains ~13,459 studies.

    Parameters
    ----------
    data_dir : str
        Directory to store/load data
    download_abstracts_email : str, optional
        Email for PubMed API to download abstracts. If None, abstracts won't be downloaded.

    Returns
    -------
    nimare.dataset.Dataset
        The NeuroQuery dataset
    """
    dataset_path = os.path.join(data_dir, "neuroquery_dataset.pkl.gz")

    if os.path.exists(dataset_path):
        print("Loading existing NeuroQuery dataset...")
        dset = dataset.Dataset.load(dataset_path)
    else:
        print("Downloading and converting NeuroQuery dataset...")
        files = extract.fetch_neuroquery(
            data_dir=data_dir,
            version="1",
            source="combined",
            vocab="neuroquery6308",
            type="tfidf",
            overwrite=False,
        )
        # NeuroQuery uses the same conversion function as Neurosynth
        dset = io.convert_neurosynth_to_dataset(
            coordinates_file=files[0]["coordinates"],
            metadata_file=files[0]["metadata"],
            annotations_files=files[0]["features"],
        )
        dset.save(dataset_path)

    # Download abstracts if email provided and not already present
    if download_abstracts_email and (
        dset.texts is None or "abstract" not in dset.texts.columns
    ):
        print("Downloading abstracts from PubMed...")
        dset = extract.download_abstracts(dset, email=download_abstracts_email)
        dset.save(dataset_path)

    return dset


def load_brainmap_dataset(sleuth_file):
    """
    Load BrainMap dataset from a Sleuth export file.

    Note: BrainMap requires a collaboration agreement. Users must export
    data using the Sleuth tool after registration.

    Parameters
    ----------
    sleuth_file : str
        Path to the Sleuth export text file

    Returns
    -------
    nimare.dataset.Dataset
        The BrainMap dataset

    Raises
    ------
    FileNotFoundError
        If the Sleuth file does not exist
    """
    if not os.path.exists(sleuth_file):
        raise FileNotFoundError(
            f"BrainMap Sleuth file not found: {sleuth_file}\n"
            "BrainMap is a semi-closed resource. You need to:\n"
            "1. Register at brainmap.org\n"
            "2. Export studies using the Sleuth tool\n"
            "3. Provide the path to the exported .txt file"
        )

    print("Loading BrainMap dataset from Sleuth export...")
    dset = io.convert_sleuth_to_dataset(sleuth_file)
    return dset


def list_cognitive_atlas_terms(data_dir="data", search=None):
    """
    List available Cognitive Atlas terms.

    Parameters
    ----------
    data_dir : str
        Directory to store/load data
    search : str, optional
        Filter terms containing this string

    Returns
    -------
    list
        Sorted list of term names
    """
    id_df, _ = load_cognitive_atlas(data_dir)
    terms = id_df["name"].dropna().tolist()

    if search:
        terms = [t for t in terms if search.lower() in t.lower()]

    return sorted(terms)
