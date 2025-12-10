#!/usr/bin/env python3
"""
Download and combine multiple OpenSky data files for better diversity.
Usage: python scripts/download_opensky_data.py [options]
"""

import argparse
import os
from datetime import datetime, timedelta
import requests
from pathlib import Path
import random


def generate_download_urls(base_date: str, num_days: int = 3, hours_per_day: int = 4):
    """Generate OpenSky download URLs for multiple time periods."""
    urls = []
    base_dt = datetime.strptime(base_date, "%Y-%m-%d")

    print(f"Generating URLs for {num_days} days, {hours_per_day} hours each")

    for day_offset in range(num_days):
        current_date = base_dt + timedelta(days=day_offset)
        date_str = current_date.strftime("%Y-%m-%d")

        # Suggest peak hours (adjust for timezone)
        # sample without replacement
        selected_hours = random.sample([hour for hour in range(24)], k=hours_per_day)

        for hour in selected_hours:
            filename = f"states_{date_str}-{hour:02d}.csv.tar"
            # Direct S3 URL for OpenSky data
            url = f"https://s3.opensky-network.org/data-samples/states/{date_str}/{hour:02d}/{filename}"
            urls.append(
                {
                    "url": url,
                    "filename": filename,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "hour": hour,
                }
            )

    return urls


def estimate_data_size(urls):
    """Estimate total download size."""
    # Typical OpenSky hourly file is 100-500MB compressed
    avg_size_mb = 250
    total_size_gb = len(urls) * avg_size_mb / 1024

    print(f"Estimated download size: ~{total_size_gb:.1f} GB")
    print(f"Expected unique aircraft: ~{len(urls) * 150}-{len(urls) * 400}")
    return total_size_gb


def main():
    parser = argparse.ArgumentParser(
        description="Download OpenSky data for better diversity"
    )
    parser.add_argument(
        "--base_date", default="2022-06-20", help="Base date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--num_days", type=int, default=1, help="Number of days to download"
    )
    parser.add_argument(
        "--hours_per_day", type=int, default=10, help="Hours per day (peak times)"
    )
    parser.add_argument(
        "--output_dir", default="data/raw", help="Output directory for downloaded files"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Just show URLs, don't download"
    )

    args = parser.parse_args()

    print(f"OpenSky Data Downloader")
    print(f"Base date: {args.base_date}")
    print(f"Days: {args.num_days}, Hours per day: {args.hours_per_day}")
    print("=" * 50)

    # Generate URLs
    urls = generate_download_urls(args.base_date, args.num_days, args.hours_per_day)

    print(f"Generated {len(urls)} download URLs:")
    for i, url_info in enumerate(urls[:5]):  # Show first 5
        print(
            f"  {i+1}. {url_info['filename']} ({url_info['date']} {url_info['hour']:02d}:00)"
        )
    if len(urls) > 5:
        print(f"  ... and {len(urls)-5} more")

    print()

    # Estimate size
    total_size_gb = estimate_data_size(urls)

    if args.dry_run:
        print("DRY RUN - URLs generated but not downloading")
        print("\nFull URL list:")
        for url_info in urls:
            print(f"  {url_info['url']}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Warning about download size
    if total_size_gb > 2:
        response = input(f"Large download ({total_size_gb:.1f} GB). Continue? (y/N): ")
        if response.lower() != "y":
            print("Download cancelled.")
            return

    print("Starting downloads...")

    # Download files
    successful_downloads = 0
    for i, url_info in enumerate(urls):
        url = url_info["url"]
        filename = url_info["filename"]
        filepath = Path(args.output_dir) / filename

        print(f"[{i+1}/{len(urls)}] Downloading {filename}...")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Check if we're getting HTML instead of data
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type:
                print(f"  WARNING: Got HTML response for {filename}, likely wrong URL")
                print(f"  URL: {url}")
                continue

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            
            # Additional check - if file is very small, it might be an error page
            if file_size_mb < 1.0:
                print(f"  WARNING: File {filename} is only {file_size_mb:.2f} MB, checking content...")
                with open(filepath, 'rb') as f:
                    first_100_bytes = f.read(100).decode('utf-8', errors='ignore')
                    if '<html' in first_100_bytes.lower() or '<!doctype' in first_100_bytes.lower():
                        print(f"  ERROR: File contains HTML, not data. Deleting {filename}")
                        filepath.unlink()  # Delete the HTML file
                        continue
            
            print(f"  Downloaded {filename} ({file_size_mb:.1f} MB)")
            successful_downloads += 1

        except requests.exceptions.RequestException as e:
            print(f"  Failed to download {filename}: {e}")
        except Exception as e:
            print(f"  Error saving {filename}: {e}")

    print(f"\nDownload Summary:")
    print(f"  Successful: {successful_downloads}/{len(urls)} files")
    print(f"  Location: {args.output_dir}")


if __name__ == "__main__":
    main()
