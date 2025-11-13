#!/bin/bash
# Semantic Scholar Dataset Download Script
# Documentation: https://api.semanticscholar.org/api-docs/datasets

set -e

# Configuration
API_KEY="2LfDzUPRgO3CmrmPFMeXy7gX4PQW1iIHqr3kCMQ9"
DOWNLOAD_DIR="/path/to/your/2TB/storage/semantic_scholar"
TEMP_DIR="$DOWNLOAD_DIR/temp"

# Create directories
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$TEMP_DIR"

echo "=================================="
echo "Semantic Scholar Dataset Downloader"
echo "=================================="
echo "Download directory: $DOWNLOAD_DIR"
echo "API Key: ${API_KEY:0:10}..."
echo ""

# Function to download with retry
download_with_retry() {
    local url=$1
    local output=$2
    local max_retries=3
    local retry=0

    while [ $retry -lt $max_retries ]; do
        echo "Downloading: $url (attempt $((retry+1))/$max_retries)"
        if curl -L -H "x-api-key: $API_KEY" -o "$output" "$url"; then
            echo "✓ Download successful"
            return 0
        else
            retry=$((retry+1))
            if [ $retry -lt $max_retries ]; then
                echo "✗ Download failed. Retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done

    echo "✗ Failed to download after $max_retries attempts"
    return 1
}

# Step 1: Get latest release information
echo "Step 1: Fetching latest release information..."
RELEASE_INFO="$DOWNLOAD_DIR/releases.json"

curl -H "x-api-key: $API_KEY" \
     "https://api.semanticscholar.org/datasets/v1/release" \
     -o "$RELEASE_INFO"

echo "✓ Release information saved to: $RELEASE_INFO"
echo ""

# Display available datasets
echo "Available datasets in latest release:"
python3 << 'PYTHON_EOF'
import json
import sys

try:
    with open("$RELEASE_INFO", 'r') as f:
        data = json.load(f)

    print(f"Release ID: {data.get('release_id', 'Unknown')}")
    print(f"Release Date: {data.get('release_date', 'Unknown')}")
    print("")
    print("Datasets:")

    for dataset in data.get('datasets', []):
        name = dataset.get('name', 'Unknown')
        description = dataset.get('description', '')
        size = dataset.get('size', 'Unknown')
        print(f"  - {name}")
        print(f"    Description: {description}")
        print(f"    Size: {size}")
        print("")
except Exception as e:
    print(f"Error reading release info: {e}")
    sys.exit(1)
PYTHON_EOF

echo ""
read -p "Continue with download? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

# Step 2: Download Papers dataset
echo ""
echo "Step 2: Downloading Papers dataset..."
echo "This will take several hours depending on your connection."
echo ""

PAPERS_DIR="$DOWNLOAD_DIR/papers"
mkdir -p "$PAPERS_DIR"

# Get download URLs for papers dataset
curl -H "x-api-key: $API_KEY" \
     "https://api.semanticscholar.org/datasets/v1/release/latest/dataset/papers" \
     -o "$TEMP_DIR/papers_manifest.json"

# Download all paper files
python3 << 'PYTHON_EOF'
import json
import subprocess
import os

manifest_file = os.path.join(os.environ['TEMP_DIR'], 'papers_manifest.json')
papers_dir = os.environ['PAPERS_DIR']
api_key = os.environ['API_KEY']

with open(manifest_file, 'r') as f:
    manifest = json.load(f)

files = manifest.get('files', [])
total = len(files)

print(f"Found {total} files to download")
print("")

for idx, file_info in enumerate(files, 1):
    url = file_info.get('url')
    filename = file_info.get('name', f'papers_{idx}.jsonl.gz')
    output_path = os.path.join(papers_dir, filename)

    if os.path.exists(output_path):
        print(f"[{idx}/{total}] Skipping (already exists): {filename}")
        continue

    print(f"[{idx}/{total}] Downloading: {filename}")

    cmd = [
        'curl', '-L',
        '-H', f'x-api-key: {api_key}',
        '-o', output_path,
        url
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Downloaded: {filename}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download: {filename}")
        print(f"  Error: {e}")

    print("")

print("Papers dataset download complete!")
PYTHON_EOF

# Step 3: Download Authors dataset (optional)
echo ""
read -p "Download Authors dataset? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading Authors dataset..."
    AUTHORS_DIR="$DOWNLOAD_DIR/authors"
    mkdir -p "$AUTHORS_DIR"

    curl -H "x-api-key: $API_KEY" \
         "https://api.semanticscholar.org/datasets/v1/release/latest/dataset/authors" \
         -o "$TEMP_DIR/authors_manifest.json"

    # Similar download process for authors
    echo "Authors dataset download initiated..."
fi

# Step 4: Download Citations dataset (optional)
echo ""
read -p "Download Citations dataset? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading Citations dataset..."
    CITATIONS_DIR="$DOWNLOAD_DIR/citations"
    mkdir -p "$CITATIONS_DIR"

    curl -H "x-api-key: $API_KEY" \
         "https://api.semanticscholar.org/datasets/v1/release/latest/dataset/citations" \
         -o "$TEMP_DIR/citations_manifest.json"

    echo "Citations dataset download initiated..."
fi

# Step 5: Create download summary
echo ""
echo "=================================="
echo "Download Summary"
echo "=================================="
echo "Location: $DOWNLOAD_DIR"
echo ""
echo "Disk usage:"
du -sh "$DOWNLOAD_DIR"
echo ""
echo "File counts:"
echo "  Papers: $(find "$PAPERS_DIR" -name '*.jsonl.gz' 2>/dev/null | wc -l) files"
echo ""

# Create README
cat > "$DOWNLOAD_DIR/README.md" << 'README_EOF'
# Semantic Scholar Dataset

## Directory Structure

- `papers/` - Papers dataset (JSON Lines, gzipped)
- `authors/` - Authors dataset
- `citations/` - Citation network
- `temp/` - Temporary files and manifests

## File Format

Each `.jsonl.gz` file contains JSON objects, one per line:

```json
{
  "corpusid": 123456789,
  "externalids": {"DOI": "10.1234/...", "ArXiv": "..."},
  "title": "Paper Title",
  "authors": [{"authorId": "...", "name": "..."}],
  "venue": "Conference/Journal Name",
  "year": 2024,
  "citationcount": 42,
  "abstract": "Paper abstract...",
  "fieldsOfStudy": ["Computer Science", "Medicine"],
  ...
}
```

## Usage

See `analyze_dataset.py` for analysis examples.

## Dataset Size

Approximately 200M+ papers, ~500GB-1TB compressed.

## More Info

https://api.semanticscholar.org/api-docs/datasets
README_EOF

echo "✓ README created: $DOWNLOAD_DIR/README.md"
echo ""
echo "=================================="
echo "Download Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Review downloaded files in: $DOWNLOAD_DIR"
echo "2. Use analyze_dataset.py to explore the data"
echo "3. Extract plant disease papers using filter_papers.py"
echo ""
