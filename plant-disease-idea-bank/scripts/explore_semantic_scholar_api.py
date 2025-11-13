#!/usr/bin/env python3
"""
Semantic Scholar Dataset API Explorer

This script explores the Semantic Scholar Datasets API to understand:
1. What datasets are available
2. Dataset sizes and structure
3. Download URLs and methods
4. Latest release information

API Documentation: https://api.semanticscholar.org/api-docs/datasets
"""

import requests
import json
from pathlib import Path
from typing import Dict, List
import time

class SemanticScholarDatasetExplorer:
    """Explore Semantic Scholar Dataset API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.semanticscholar.org/datasets/v1"
        self.headers = {"x-api-key": api_key}

    def get_latest_release(self) -> Dict:
        """Get information about the latest dataset release"""
        url = f"{self.base_url}/release/latest"
        print(f"📡 Fetching latest release info from: {url}")

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            print("✅ Successfully fetched latest release info\n")
            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching release info: {e}")
            return {}

    def get_all_releases(self) -> List[Dict]:
        """Get list of all available releases"""
        url = f"{self.base_url}/release"
        print(f"📡 Fetching all releases from: {url}")

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            print(f"✅ Found {len(data)} releases\n")
            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching releases: {e}")
            return []

    def get_dataset_info(self, dataset_name: str, release_id: str = "latest") -> Dict:
        """
        Get information about a specific dataset

        Args:
            dataset_name: Name of dataset (e.g., 'papers', 'authors', 'citations')
            release_id: Release ID or 'latest'
        """
        url = f"{self.base_url}/release/{release_id}/dataset/{dataset_name}"
        print(f"📡 Fetching {dataset_name} dataset info...")

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            print(f"✅ Successfully fetched {dataset_name} info\n")
            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching {dataset_name}: {e}")
            return {}

    def display_release_info(self, release_data: Dict):
        """Pretty print release information"""
        if not release_data:
            print("❌ No release data to display")
            return

        print("=" * 70)
        print("📦 SEMANTIC SCHOLAR DATASET RELEASE INFORMATION")
        print("=" * 70)
        print()

        # Basic info
        print(f"🆔 Release ID: {release_data.get('release_id', 'N/A')}")
        print(f"📅 Release Date: {release_data.get('release_date', 'N/A')}")
        print(f"📝 README: {release_data.get('README', 'N/A')}")
        print()

        # Datasets
        datasets = release_data.get('datasets', [])
        if datasets:
            print(f"📊 Available Datasets ({len(datasets)}):")
            print("-" * 70)
            for ds in datasets:
                print(f"\n  📁 {ds.get('name', 'Unknown').upper()}")
                print(f"     Description: {ds.get('description', 'No description')}")

                if 'README' in ds:
                    print(f"     README: {ds['README']}")
        else:
            print("⚠️  No datasets found in release")

        print()
        print("=" * 70)

    def display_dataset_details(self, dataset_data: Dict, dataset_name: str):
        """Pretty print detailed dataset information"""
        if not dataset_data:
            print(f"❌ No data for {dataset_name}")
            return

        print("=" * 70)
        print(f"📊 DATASET: {dataset_name.upper()}")
        print("=" * 70)
        print()

        # Basic info
        print(f"📝 Name: {dataset_data.get('name', 'N/A')}")
        print(f"📅 Release Date: {dataset_data.get('release_date', 'N/A')}")
        print(f"📄 README: {dataset_data.get('README', 'N/A')}")
        print()

        # Files
        files = dataset_data.get('files', [])
        if files:
            total_size = sum(f.get('size', 0) for f in files)
            total_size_gb = total_size / (1024**3)

            print(f"📦 Files: {len(files)} files")
            print(f"💾 Total Size: {total_size:,} bytes ({total_size_gb:.2f} GB)")
            print()

            print("📋 File List (first 10):")
            print("-" * 70)
            for idx, file_info in enumerate(files[:10], 1):
                name = file_info.get('name', 'Unknown')
                size = file_info.get('size', 0)
                size_mb = size / (1024**2)
                url = file_info.get('url', 'No URL')

                print(f"\n  {idx}. {name}")
                print(f"     Size: {size_mb:.2f} MB")
                print(f"     URL: {url[:80]}...")

            if len(files) > 10:
                print(f"\n  ... and {len(files) - 10} more files")
        else:
            print("⚠️  No files found")

        print()
        print("=" * 70)

    def estimate_download_time(self, total_size_gb: float, speed_mbps: float = 100):
        """Estimate download time"""
        speed_mbs = speed_mbps / 8  # Convert to MB/s
        total_size_mb = total_size_gb * 1024
        time_seconds = total_size_mb / speed_mbs
        time_hours = time_seconds / 3600

        print(f"⏱️  Estimated download time @ {speed_mbps} Mbps: {time_hours:.1f} hours")

    def save_to_file(self, data: Dict, filename: str):
        """Save data to JSON file"""
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Saved to: {output_path}")


def main():
    """Main exploration function"""
    print()
    print("=" * 70)
    print("🔍 SEMANTIC SCHOLAR DATASET API EXPLORER")
    print("=" * 70)
    print()

    # API Key
    API_KEY = "2LfDzUPRgO3CmrmPFMeXy7gX4PQW1iIHqr3kCMQ9"

    # Initialize explorer
    explorer = SemanticScholarDatasetExplorer(API_KEY)

    # Output directory
    output_dir = Path("/tmp/semantic_scholar_info")
    output_dir.mkdir(exist_ok=True)

    print(f"📁 Output directory: {output_dir}\n")

    # Step 1: Get all releases
    print("STEP 1: Fetching all releases...")
    print("-" * 70)
    all_releases = explorer.get_all_releases()
    if all_releases:
        explorer.save_to_file(all_releases, output_dir / "all_releases.json")
        print(f"   Found {len(all_releases)} releases")
        print(f"   Latest: {all_releases[0].get('release_id', 'N/A')}" if all_releases else "")
    print()
    time.sleep(1)  # Respect rate limit

    # Step 2: Get latest release info
    print("STEP 2: Fetching latest release details...")
    print("-" * 70)
    latest_release = explorer.get_latest_release()
    if latest_release:
        explorer.save_to_file(latest_release, output_dir / "latest_release.json")
        explorer.display_release_info(latest_release)
    print()
    time.sleep(1)

    # Step 3: Get detailed info for each dataset
    datasets_to_check = ['papers', 'authors', 'citations', 'embeddings-specter_v2']

    for dataset_name in datasets_to_check:
        print(f"\nSTEP 3.{datasets_to_check.index(dataset_name)+1}: Fetching {dataset_name} dataset...")
        print("-" * 70)

        dataset_info = explorer.get_dataset_info(dataset_name)
        if dataset_info:
            explorer.save_to_file(
                dataset_info,
                output_dir / f"dataset_{dataset_name}.json"
            )
            explorer.display_dataset_details(dataset_info, dataset_name)

            # Estimate download time
            files = dataset_info.get('files', [])
            if files:
                total_size = sum(f.get('size', 0) for f in files)
                total_size_gb = total_size / (1024**3)
                explorer.estimate_download_time(total_size_gb)

        print()
        time.sleep(1)  # Respect rate limit (1 req/sec)

    # Summary
    print()
    print("=" * 70)
    print("✅ EXPLORATION COMPLETE!")
    print("=" * 70)
    print()
    print(f"📁 All information saved to: {output_dir}")
    print()
    print("📋 Files created:")
    for f in sorted(output_dir.glob("*.json")):
        print(f"   - {f.name}")
    print()
    print("💡 Next steps:")
    print("   1. Review the JSON files to understand dataset structure")
    print("   2. Use download_semantic_scholar_full.py to download datasets")
    print("   3. Use filter_papers.py to extract plant disease papers")
    print()


if __name__ == "__main__":
    main()
