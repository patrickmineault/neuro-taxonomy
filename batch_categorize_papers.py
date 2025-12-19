"""
Batch categorize Neuroquery papers using Claude API.

This script:
1. Loads paper titles and abstracts from Neuroquery dataset
2. Loads taxonomy from cognitive_capacities.csv
3. Sends batch requests to Claude API (Sonnet 4.5) to categorize each paper
4. Caches results to handle failures and resume
5. Saves final results to CSV
"""

import asyncio
import json
import os
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from datasets import load_neuroquery_dataset

# Load environment variables from .env file
load_dotenv()


MODEL_NAME = "claude-sonnet-4-5-20250929"


def load_taxonomy(taxonomy_path="cognitive_capacities.csv"):
    """Load taxonomy from CSV file."""
    df = pd.read_csv(taxonomy_path)

    # Format taxonomy as a readable list
    taxonomy_lines = []
    for _, row in df.iterrows():
        capacity = row["Capacity"]
        description = row.get("Description", "")
        if pd.notna(description) and description:
            taxonomy_lines.append(f"- {capacity}: {description}")
        else:
            taxonomy_lines.append(f"- {capacity}")

    return "\n".join(taxonomy_lines)


def load_papers(source="neuroquery", data_dir="data"):
    """
    Load paper titles and abstracts from specified source.

    Parameters
    ----------
    source : str
        Either "neuroquery" to load from Neuroquery dataset, or a path to a CSV file
        with 'title' and 'abstract' columns
    data_dir : str
        Directory containing Neuroquery data (only used if source="neuroquery")

    Returns
    -------
    pd.DataFrame
        DataFrame with 'id', 'title', and 'abstract' columns
    """
    if source == "neuroquery":
        dset = load_neuroquery_dataset(data_dir)

        if dset.texts is None or "abstract" not in dset.texts.columns:
            raise ValueError("Neuroquery dataset does not have abstracts loaded")

        texts_df = dset.texts.copy()
        if "id" in texts_df.columns:
            texts_df = texts_df.set_index("id")

        # Get titles from metadata if available
        if dset.metadata is not None and "title" in dset.metadata.columns:
            metadata_df = dset.metadata.copy()
            if "id" in metadata_df.columns:
                metadata_df = metadata_df.set_index("id")
            texts_df = texts_df.join(metadata_df[["title"]], how="left")
        else:
            texts_df["title"] = ""

        # Filter out papers without abstracts
        texts_df = texts_df[texts_df["abstract"].notna() & (texts_df["abstract"] != "")]

        return texts_df.reset_index()
    else:
        # Load from CSV file
        if not os.path.exists(source):
            raise ValueError(f"CSV file not found: {source}")

        texts_df = pd.read_csv(source)

        # Check required columns
        if "abstract" not in texts_df.columns:
            raise ValueError(f"CSV file must have 'abstract' column: {source}")

        # Add title column if missing
        if "title" not in texts_df.columns:
            texts_df["title"] = ""

        # Add id column if missing (use index or doi if available)
        if "id" not in texts_df.columns:
            if "doi" in texts_df.columns:
                texts_df["id"] = texts_df["doi"]
            else:
                texts_df["id"] = texts_df.index.astype(str)

        # Filter out papers without abstracts
        texts_df = texts_df[texts_df["abstract"].notna() & (texts_df["abstract"] != "")]

        return texts_df


def build_prompt(taxonomy, title, abstract):
    """Build the categorization prompt."""
    return f"""Categorize the following paper with their abstract, according to the following taxonomy:

{taxonomy}

Answer as a parsable json list only. List at most two elements in the taxonomy. Start with the most obvious one, and an optional second element. It is valid to return an empty list if the core topic of the paper is outside of the taxonomy, for example if it addresses a structural or connectomic difference rather than a functional one, or is highly focused on a particular disease, or is a review rather than primary research, or is a methods paper, etc. When multiple elements could be selected that are overlapping, select the most granular one (e.g. visual attention over attention when visual attention is specifically addressed). Do not list incidental elements, for example, in a N-back task where visually presented word lists must be remembered and responses entered through a button, list working memory, not visual perception, language comprehension or motor production; in a stroop task, list inhibition, etc.; unless the abstract calls out that it is specifically about an aspect of a task that is not the obvious or standard one.

{title}
{abstract}"""


def load_cache(cache_path):
    """Load cached results from file."""
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_path):
    """Save results to cache file."""
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


async def categorize_paper_async(
    client, paper_id, title, abstract, taxonomy, semaphore
):
    """Categorize a single paper using Claude API (async)."""
    async with semaphore:
        prompt = build_prompt(taxonomy, title, abstract)

        try:
            message = await client.messages.create(
                model=MODEL_NAME,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = message.content[0].text.strip()

            # Try to parse as JSON
            try:
                # Handle potential markdown code blocks
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                    response_text = response_text.strip()

                categories = json.loads(response_text)
                return paper_id, categories, None
            except json.JSONDecodeError as e:
                return paper_id, response_text, f"JSON parse error: {e}"

        except anthropic.APIError as e:
            return paper_id, None, f"API error: {e}"
        except Exception as e:
            return paper_id, None, f"Error: {e}"


async def batch_categorize_async(
    papers_df, taxonomy, cache, cache_path, max_concurrent=50, save_every=100
):
    """Categorize papers in parallel using async."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Filter out already cached papers
    papers_to_process = []
    for _, row in papers_df.iterrows():
        paper_id = str(row["id"])
        if paper_id not in cache:
            papers_to_process.append(row)

    print(f"Papers to process: {len(papers_to_process)} (cached: {len(cache)})")

    if not papers_to_process:
        return cache

    # Create tasks
    tasks = []
    for row in papers_to_process:
        paper_id = str(row["id"])
        title = row.get("title", "") or ""
        abstract = row["abstract"]

        task = categorize_paper_async(
            client, paper_id, title, abstract, taxonomy, semaphore
        )
        tasks.append(task)

    # Process with progress bar
    results = []
    processed = 0

    for coro in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Categorizing"
    ):
        result = await coro
        results.append(result)
        processed += 1

        paper_id, categories, error = result
        # Only cache successful results, not errors
        if not error:
            cache[paper_id] = {"categories": categories}

        # Save cache periodically
        if processed % save_every == 0:
            save_cache(cache, cache_path)

    # Final save
    save_cache(cache, cache_path)

    return cache


def categorize_paper_sync(client, paper_id, title, abstract, taxonomy):
    """Categorize a single paper using Claude API (sync)."""
    prompt = build_prompt(taxonomy, title, abstract)

    try:
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text.strip()

        # Try to parse as JSON
        try:
            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            categories = json.loads(response_text)
            return categories, None
        except json.JSONDecodeError as e:
            return response_text, f"JSON parse error: {e}"

    except anthropic.APIError as e:
        return None, f"API error: {e}"
    except Exception as e:
        return None, f"Error: {e}"


def batch_categorize_sync(papers_df, taxonomy, cache, cache_path, save_every=100):
    """Categorize papers sequentially (for debugging or rate limit issues)."""
    client = anthropic.Anthropic()

    # Filter out already cached papers
    papers_to_process = []
    for _, row in papers_df.iterrows():
        paper_id = str(row["id"])
        if paper_id not in cache:
            papers_to_process.append(row)

    print(f"Papers to process: {len(papers_to_process)} (cached: {len(cache)})")

    for i, row in enumerate(papers_to_process):
        paper_id = str(row["id"])
        title = row.get("title", "") or ""
        abstract = row["abstract"]

        categories, error = categorize_paper_sync(
            client, paper_id, title, abstract, taxonomy
        )

        # Print results to screen
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(papers_to_process)}] {paper_id}")
        print(f"Title: {title}")
        print(
            f"Abstract: {abstract[:300]}..."
            if len(abstract) > 300
            else f"Abstract: {abstract}"
        )
        if error:
            print(f"Error: {error}")
        else:
            print(f"Categories: {categories}")

        # Only cache successful results, not errors
        if not error:
            cache[paper_id] = {"categories": categories}

        # Save cache periodically
        if (i + 1) % save_every == 0:
            save_cache(cache, cache_path)

    # Final save
    save_cache(cache, cache_path)

    return cache


def results_to_dataframe(papers_df, cache):
    """Convert cached results to a DataFrame."""
    results = []

    for _, row in papers_df.iterrows():
        paper_id = str(row["id"])
        result = {
            "id": paper_id,
            "title": row.get("title", ""),
            "abstract": (
                row["abstract"][:200] + "..."
                if len(row["abstract"]) > 200
                else row["abstract"]
            ),
        }

        if paper_id in cache:
            cached = cache[paper_id]
            if "categories" in cached:
                categories = cached["categories"]
                if isinstance(categories, list):
                    result["category_1"] = (
                        categories[0] if len(categories) > 0 else None
                    )
                    result["category_2"] = (
                        categories[1] if len(categories) > 1 else None
                    )
                else:
                    result["category_1"] = str(categories)
                    result["category_2"] = None
            else:
                result["category_1"] = None
                result["category_2"] = None
                result["error"] = cached.get("error", "Unknown error")
        else:
            result["category_1"] = None
            result["category_2"] = None
            result["error"] = "Not processed"

        results.append(result)

    return pd.DataFrame(results)


def get_source_name(source):
    """Extract a name from the source for use in file names."""
    if source == "neuroquery":
        return "neuroquery"
    else:
        # Extract base name from CSV path (without extension)
        return Path(source).stem


def main(
    taxonomy_path="cognitive_capacities.csv",
    source="neuroquery",
    data_dir="data",
    output_dir="results",
    cache_dir=".cache",
    max_concurrent=50,
    use_async=True,
    limit=None,
):
    """
    Main function to batch categorize papers.

    Parameters
    ----------
    taxonomy_path : str
        Path to taxonomy CSV file
    source : str
        Either "neuroquery" to load from Neuroquery dataset, or a path to a CSV file
        with 'title' and 'abstract' columns
    data_dir : str
        Directory containing Neuroquery data (only used if source="neuroquery")
    output_dir : str
        Directory to save final results CSV
    cache_dir : str
        Directory to save cache file for resuming
    max_concurrent : int
        Maximum concurrent API requests (for async mode)
    use_async : bool
        Use async API calls (faster) or sync (for debugging)
    limit : int, optional
        Limit number of papers to process (for testing)
    """
    # Generate file paths based on source name
    source_name = get_source_name(source)
    output_path = Path(output_dir) / f"paper_categories_{source_name}.csv"
    cache_path = Path(cache_dir) / f"paper_categories_cache_{source_name}.json"

    # Ensure directories exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"Source: {source_name}")
    print(f"Output: {output_path}")
    print(f"Cache: {cache_path}")

    print("\nLoading taxonomy...")
    taxonomy = load_taxonomy(taxonomy_path)
    print(f"Taxonomy loaded:\n{taxonomy[:500]}...")

    print("\nLoading papers...")
    papers_df = load_papers(source=source, data_dir=data_dir)
    print(f"Loaded {len(papers_df)} papers with abstracts")

    if limit:
        papers_df = papers_df.head(limit)
        print(f"Limited to {limit} papers for testing")

    print("\nLoading cache...")
    cache = load_cache(cache_path)
    print(f"Cache contains {len(cache)} entries")

    print("\nStarting categorization...")
    if use_async:
        cache = asyncio.run(
            batch_categorize_async(
                papers_df, taxonomy, cache, cache_path, max_concurrent=max_concurrent
            )
        )
    else:
        cache = batch_categorize_sync(papers_df, taxonomy, cache, cache_path)

    print("\nConverting results to DataFrame...")
    results_df = results_to_dataframe(papers_df, cache)

    print(f"\nSaving results to {output_path}...")
    results_df.to_csv(output_path, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total papers: {len(results_df)}")
    print(f"Successfully categorized: {results_df['category_1'].notna().sum()}")
    print(f"Errors: {results_df['category_1'].isna().sum()}")

    if "category_1" in results_df.columns:
        print("\nTop categories:")
        print(results_df["category_1"].value_counts().head(20))

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch categorize Neuroquery papers using Claude API"
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        default="data/cognitive_capacities.csv",
        help="Path to taxonomy CSV file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="neuroquery",
        help="Paper source: 'neuroquery' or path to CSV file with 'title' and 'abstract' columns",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing Neuroquery data (only used if source='neuroquery')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results CSV (file named paper_categories_{source}.csv)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help="Directory to save cache file (file named paper_categories_cache_{source}.json)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Maximum concurrent API requests",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous API calls instead of async",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers (for testing)",
    )

    args = parser.parse_args()

    results = main(
        taxonomy_path=args.taxonomy,
        source=args.source,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        max_concurrent=args.max_concurrent,
        use_async=not args.sync,
        limit=args.limit,
    )
