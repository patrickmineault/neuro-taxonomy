#!/usr/bin/env python3
"""
Download neuroscience abstracts from bioRxiv and PsyArXiv for 2025.
Aggregates results into a CSV with title, abstract, authors, source, and DOI.

Features intelligent caching to resume downloads on crash.
"""

import csv
import json
import os
import time
from typing import Generator

import requests

# Constants
BIORXIV_API_BASE = "https://api.biorxiv.org/details/biorxiv"
OSF_API_BASE = "https://api.osf.io/v2"
OUTPUT_FILE_BIORXIV = "biorxiv_neuroscience_abstracts_2025.csv"
OUTPUT_FILE_PSYARXIV = "psyarxiv_neuroscience_abstracts_2025.csv"
CACHE_FILE_BIORXIV = ".biorxiv_cache.json"
CACHE_FILE_PSYARXIV = ".psyarxiv_cache.json"

# Date range for 2025
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

# Request settings
REQUEST_DELAY = 0.5  # seconds between requests to be respectful to APIs
SAVE_EVERY = 100  # save cache every N records


def load_cache(cache_file: str) -> dict:
    """Load cache from file."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not load cache from {cache_file}, starting fresh")
    return {"records": [], "cursor": 0, "next_url": None, "completed": False}


def save_cache(cache: dict, cache_file: str) -> None:
    """Save cache to file."""
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def fetch_biorxiv_neuroscience(
    start_date: str, end_date: str, cache_file: str = CACHE_FILE_BIORXIV
) -> list[dict]:
    """
    Fetch neuroscience preprints from bioRxiv API with caching.

    The bioRxiv API returns results in pages of 100. We filter by the 'neuroscience' category.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        cache_file: Path to cache file for resuming

    Returns:
        List of dicts with title, abstract, authors, source, doi
    """
    # Load existing cache
    cache = load_cache(cache_file)

    if cache.get("completed"):
        print(f"bioRxiv download already completed ({len(cache['records'])} records)")
        return cache["records"]

    cursor = cache.get("cursor", 0)
    records = cache.get("records", [])
    seen_dois = {r["doi"] for r in records if r.get("doi")}

    print(f"Fetching bioRxiv neuroscience preprints from {start_date} to {end_date}...")
    if records:
        print(f"  Resuming from cursor {cursor} with {len(records)} cached records")

    while True:
        # bioRxiv API endpoint with neuroscience category filter
        url = (
            f"{BIORXIV_API_BASE}/{start_date}/{end_date}/{cursor}?category=neuroscience"
        )

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching bioRxiv data at cursor {cursor}: {e}")
            print("Saving progress and exiting...")
            cache["cursor"] = cursor
            cache["records"] = records
            save_cache(cache, cache_file)
            break

        messages = data.get("messages", [])
        collection = data.get("collection", [])

        if not collection:
            print(f"No more bioRxiv results. Total fetched: {len(records)}")
            cache["completed"] = True
            cache["records"] = records
            save_cache(cache, cache_file)
            break

        for item in collection:
            doi = item.get("doi", "").strip()
            # Skip duplicates
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)

            record = {
                "title": item.get("title", "").strip(),
                "abstract": item.get("abstract", "").strip(),
                "authors": item.get("authors", "").strip(),
                "source": "bioRxiv",
                "doi": doi,
                "date": item.get("date", "").strip(),
            }
            records.append(record)

        # Check if we've fetched all results
        if messages:
            msg = messages[0]
            total_count = msg.get("total", 0)
            if isinstance(total_count, str):
                total_count = int(total_count) if total_count.isdigit() else 0
            if len(records) >= total_count or cursor + 100 >= total_count:
                print(
                    f"Finished fetching all {len(records)} bioRxiv neuroscience preprints"
                )
                cache["completed"] = True
                cache["records"] = records
                save_cache(cache, cache_file)
                break

        # Move to next page
        cursor += 100
        cache["cursor"] = cursor
        cache["records"] = records

        # Save cache periodically
        if len(records) % SAVE_EVERY == 0:
            save_cache(cache, cache_file)

        print(f"  Fetched {len(records)} bioRxiv records so far...")
        time.sleep(REQUEST_DELAY)

    return records


def fetch_psyarxiv_preprints(
    start_date: str, end_date: str, cache_file: str = CACHE_FILE_PSYARXIV
) -> list[dict]:
    """
    Fetch preprints from PsyArXiv via OSF API with caching.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        cache_file: Path to cache file for resuming

    Returns:
        List of dicts with title, abstract, authors, source, doi
    """
    # Load existing cache
    cache = load_cache(cache_file)

    if cache.get("completed"):
        print(f"PsyArXiv download already completed ({len(cache['records'])} records)")
        return cache["records"]

    records = cache.get("records", [])
    seen_dois = {r["doi"] for r in records if r.get("doi")}

    print(f"Fetching PsyArXiv preprints from {start_date} to {end_date}...")
    if records:
        print(f"  Resuming with {len(records)} cached records")

    # Use the provider-specific endpoint for PsyArXiv
    base_url = f"{OSF_API_BASE}/preprint_providers/psyarxiv/preprints/?filter[subjects]=Neuroscience"

    params = {
        "filter[date_created][gte]": start_date,
        "filter[date_created][lte]": end_date,
        "page[size]": 100,
        "embed": "contributors",
    }

    headers = {
        "Accept": "application/vnd.api+json",
    }

    # Resume from cached next_url or start fresh
    next_url = cache.get("next_url") or base_url
    is_first_request = next_url == base_url

    max_retries = 5
    retry_delay = 5  # seconds

    while next_url:
        # Retry logic for transient errors
        data = None
        for attempt in range(max_retries):
            try:
                if is_first_request:
                    response = requests.get(
                        next_url, params=params, headers=headers, timeout=60
                    )
                    is_first_request = False
                else:
                    response = requests.get(next_url, headers=headers, timeout=60)

                response.raise_for_status()
                data = response.json()
                break  # Success, exit retry loop
            except requests.exceptions.RequestException as e:
                print(f"Error fetching PsyArXiv data (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("Max retries reached. Saving progress and exiting...")
                    cache["next_url"] = next_url
                    cache["records"] = records
                    save_cache(cache, cache_file)

        if data is None:
            break  # Failed after all retries

        preprints = data.get("data", [])

        if not preprints:
            print(f"No more PsyArXiv results. Total fetched: {len(records)}")
            cache["completed"] = True
            cache["records"] = records
            save_cache(cache, cache_file)
            break

        for preprint in preprints:
            attrs = preprint.get("attributes", {})

            # Get DOI
            doi = attrs.get("doi", "") or ""
            if not doi:
                preprint_id = preprint.get("id", "")
                if preprint_id and "_v" in preprint_id:
                    preprint_id = preprint_id.split("_v")[0]
                if preprint_id:
                    doi = f"10.31234/osf.io/{preprint_id}"

            # Skip duplicates
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)

            # Get authors
            authors = _get_psyarxiv_authors(preprint)

            record = {
                "title": attrs.get("title", "").strip(),
                "abstract": attrs.get("description", "").strip(),
                "authors": authors,
                "source": "PsyArXiv",
                "doi": doi.strip() if doi else "",
                "date": attrs.get("date_created", "")[:10],
            }
            records.append(record)

        # Get next page URL
        links = data.get("links", {})
        next_url = links.get("next")

        cache["next_url"] = next_url
        cache["records"] = records

        # Save cache periodically
        if len(records) % SAVE_EVERY == 0:
            save_cache(cache, cache_file)

        print(f"  Fetched {len(records)} PsyArXiv records so far...")
        time.sleep(REQUEST_DELAY)

    if not next_url:
        cache["completed"] = True
        save_cache(cache, cache_file)
        print(f"Finished fetching all {len(records)} PsyArXiv preprints")

    return records


def _get_psyarxiv_authors(preprint: dict) -> str:
    """
    Extract author names from a PsyArXiv preprint.

    Uses embedded contributors data when available (via embed=contributors param).
    Falls back to separate API request if not embedded.
    """
    # Try to get from embedded data first
    embeds = preprint.get("embeds", {})
    contributors_data = embeds.get("contributors", {}).get("data", [])

    if contributors_data:
        authors = []
        for contributor in contributors_data:
            user_embeds = contributor.get("embeds", {})
            user_data = user_embeds.get("users", {}).get("data", {})
            if user_data:
                full_name = user_data.get("attributes", {}).get("full_name", "")
                if full_name:
                    authors.append(full_name)
            else:
                attrs = contributor.get("attributes", {})
                full_name = attrs.get("full_name", "")
                if full_name:
                    authors.append(full_name)
        if authors:
            return "; ".join(authors)

    # Fallback: return empty string if no authors found
    return ""


def write_to_csv(records: list, filename: str) -> None:
    """Write records to a CSV file."""
    if not records:
        print("No records to write.")
        return

    fieldnames = ["title", "abstract", "authors", "source", "doi", "date"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} records to {filename}")


def clear_cache(source: str = "both") -> None:
    """Clear cache files to force fresh download."""
    if source in ("biorxiv", "both"):
        if os.path.exists(CACHE_FILE_BIORXIV):
            os.remove(CACHE_FILE_BIORXIV)
            print(f"Removed {CACHE_FILE_BIORXIV}")
    if source in ("psyarxiv", "both"):
        if os.path.exists(CACHE_FILE_PSYARXIV):
            os.remove(CACHE_FILE_PSYARXIV)
            print(f"Removed {CACHE_FILE_PSYARXIV}")


def main(
    source: str = "both",
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    fresh: bool = False,
):
    """
    Main function to download and aggregate abstracts.

    Args:
        source: Which source to download from ('biorxiv', 'psyarxiv', or 'both')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        fresh: If True, clear cache and start fresh download
    """
    print("=" * 60)
    print("Neuroscience Abstracts Downloader")
    print("=" * 60)
    print(f"Source: {source}")
    print(f"Date range: {start_date} to {end_date}")
    print()

    if fresh:
        print("Clearing cache for fresh download...")
        clear_cache(source)
        print()

    biorxiv_count = 0
    psyarxiv_count = 0

    # Fetch from bioRxiv
    if source in ("biorxiv", "both"):
        print("-" * 40)
        biorxiv_records = fetch_biorxiv_neuroscience(start_date, end_date)
        write_to_csv(biorxiv_records, OUTPUT_FILE_BIORXIV)
        biorxiv_count = len(biorxiv_records)
        print()

    # Fetch from PsyArXiv
    if source in ("psyarxiv", "both"):
        print("-" * 40)
        psyarxiv_records = fetch_psyarxiv_preprints(start_date, end_date)
        write_to_csv(psyarxiv_records, OUTPUT_FILE_PSYARXIV)
        psyarxiv_count = len(psyarxiv_records)
        print()

    print("-" * 40)

    # Summary
    print()
    print("=" * 60)
    print("Summary:")
    if source in ("biorxiv", "both"):
        print(f"  bioRxiv neuroscience preprints: {biorxiv_count}")
        print(f"  Output file: {OUTPUT_FILE_BIORXIV}")
    if source in ("psyarxiv", "both"):
        print(f"  PsyArXiv preprints: {psyarxiv_count}")
        print(f"  Output file: {OUTPUT_FILE_PSYARXIV}")
    print(f"  Total records: {biorxiv_count + psyarxiv_count}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download neuroscience abstracts from bioRxiv and/or PsyArXiv"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["biorxiv", "psyarxiv", "both"],
        default="both",
        help="Which source to download from (default: both)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=START_DATE,
        help=f"Start date in YYYY-MM-DD format (default: {START_DATE})",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=END_DATE,
        help=f"End date in YYYY-MM-DD format (default: {END_DATE})",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear cache and start fresh download",
    )

    args = parser.parse_args()

    main(
        source=args.source,
        start_date=args.start_date,
        end_date=args.end_date,
        fresh=args.fresh,
    )
