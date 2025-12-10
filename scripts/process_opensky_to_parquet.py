#!/usr/bin/env python3
"""
Process OpenSky tar files and convert to parquet format.
Extracts tar files, loads CSV data, and creates/appends to parquet files.

Usage: python scripts/process_opensky_to_parquet.py [options]
"""

import argparse
import os
import tarfile
import gzip
import pandas as pd
from typing import Optional
import tempfile
import shutil

from aircraft_heading_gnn.data.preprocessing import ADSB_COLUMNS, load_adsb_csv


def extract_and_process_tar(tar_path: str, temp_dir: str) -> Optional[pd.DataFrame]:
    """
    Extract tar file and load CSV data.

    Args:
        tar_path: Path to the tar file
        temp_dir: Temporary directory for extraction

    Returns:
        DataFrame with ADS-B data or None if failed
    """
    try:
        print(f"  Extracting {os.path.basename(tar_path)}...")

        # Extract tar file
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(temp_dir)

        # Find CSV files (might be gzipped)
        csv_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".csv") or file.endswith(".csv.gz"):
                    csv_files.append(os.path.join(root, file))

        if not csv_files:
            print(f"    No CSV files found in {tar_path}")
            return None

        # Process each CSV file
        all_data = []
        for csv_file in csv_files:
            print(f"    Processing {os.path.basename(csv_file)}...")

            if csv_file.endswith(".gz"):
                # Handle gzipped CSV
                df = pd.read_csv(
                    csv_file,
                    names=ADSB_COLUMNS,
                    header=None,
                    compression="gzip",
                    low_memory=False,
                )
            else:
                # Regular CSV
                df = load_adsb_csv(csv_file)

            # Clean up data types for parquet compatibility
            if len(df) > 0:
                # Handle mixed types by converting to appropriate types
                for col in df.columns:
                    if col in ["time", "lastposupdate", "lastcontact"]:
                        # Convert time columns to numeric, handling strings
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif col in [
                        "lat",
                        "lon",
                        "velocity",
                        "heading",
                        "vertrate",
                        "baroaltitude",
                        "geoaltitude",
                    ]:
                        # Convert numeric columns
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif col in ["icao24", "callsign"]:
                        # Keep as string but handle NaNs
                        df[col] = df[col].astype(str).replace("nan", None)
                    elif col in ["onground", "alert", "spi"]:
                        # Boolean-like columns
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                            "Int64"
                        )  # Nullable integer
                    elif col == "squawk":
                        # Squawk codes can be strings
                        df[col] = df[col].astype(str).replace("nan", None)

                all_data.append(df)
                print(f"      Loaded {len(df):,} records")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"    Total records from tar: {len(combined_df):,}")
            return combined_df
        else:
            print(f"    No data extracted from {tar_path}")
            return None

    except Exception as e:
        print(f"    Error processing {tar_path}: {e}")
        return None


def process_opensky_files(
    input_dir: str, output_dir: str, max_files: Optional[int] = None
) -> None:
    """
    Process all OpenSky tar files in a directory.

    Args:
        input_dir: Directory containing tar files
        output_dir: Output directory for parquet files
        max_files: Maximum number of files to process (None = all)
    """

    # Find all tar files
    tar_files = []
    for file in os.listdir(input_dir):
        if file.endswith(".tar") and file.startswith("states_"):
            tar_files.append(os.path.join(input_dir, file))

    if not tar_files:
        print(f"No OpenSky tar files found in {input_dir}")
        return

    # Sort by filename for consistent processing order
    tar_files.sort()

    if max_files:
        tar_files = tar_files[:max_files]

    print(f"Found {len(tar_files)} tar file(s) to process")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    processed_files = []

    # Process each tar file
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, tar_path in enumerate(tar_files):
            tar_name = os.path.basename(tar_path).replace(".tar", "")
            print(
                f"\n[{i+1}/{len(tar_files)}] Processing {os.path.basename(tar_path)}..."
            )

            # Clear temp directory for each file
            temp_extract_dir = os.path.join(temp_dir, f"extract_{i}")
            os.makedirs(temp_extract_dir, exist_ok=True)

            df = extract_and_process_tar(tar_path, temp_extract_dir)

            if df is not None and len(df) > 0:
                # Save each tar file as separate parquet
                output_file = os.path.join(output_dir, f"{tar_name}.parquet")
                print(f"  Saving {len(df):,} records to {output_file}...")
                df.to_parquet(output_file, index=False, engine="pyarrow")

                file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                print(f"  Saved: {file_size_mb:.1f} MB")
                processed_files.append(output_file)

            # Clean up temp extraction directory
            shutil.rmtree(temp_extract_dir)

    print(f"\nProcessed {len(processed_files)} files:")
    total_size_mb = 0
    total_records = 0
    for pf in processed_files:
        size_mb = os.path.getsize(pf) / (1024 * 1024)
        total_size_mb += size_mb
        # Quick count without loading full file
        df_info = pd.read_parquet(pf, columns=["icao24"])
        records = len(df_info)
        total_records += records
        print(f"  {os.path.basename(pf)}: {records:,} records ({size_mb:.1f} MB)")

    print(
        f"\nTotal: {total_records:,} records across {len(processed_files)} files ({total_size_mb:.1f} MB)"
    )
    print(f"Files saved in: {output_dir}")
    return


def main():
    parser = argparse.ArgumentParser(
        description="Process OpenSky tar files and convert to parquet"
    )
    parser.add_argument(
        "--input_dir", default="data/raw", help="Directory containing OpenSky tar files"
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/batches",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        help="Maximum number of tar files to process (for testing)",
    )

    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        response = input(
            f"Output directory {args.output_dir} exists. Continue? (y/N): "
        )
        if response.lower() != "y":
            print("Cancelled.")
            return

    print("OpenSky to Parquet Processor")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("Mode: Separate files (one parquet per tar)")
    print("=" * 50)

    process_opensky_files(
        input_dir=args.input_dir, output_dir=args.output_dir, max_files=args.max_files
    )

    print("\nProcessing complete!")
    print(f"Parquet files saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
