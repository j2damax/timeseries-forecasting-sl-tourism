"""
Helper script for downloading SLTDA PDF files manually.

This script provides guidance and utilities for manually downloading
PDF files when automated access is restricted.
"""

import os
import sys

def print_instructions():
    """Print detailed instructions for manual PDF download."""
    print("=" * 80)
    print("SLTDA TOURISM DATA - MANUAL DOWNLOAD GUIDE")
    print("=" * 80)
    print()
    print("The SLTDA website may be blocked in automated environments.")
    print("Follow these steps to manually download the PDF reports:")
    print()
    print("STEP 1: Access the SLTDA Website")
    print("-" * 40)
    print("  Visit: https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports")
    print()
    print("STEP 2: Download Monthly Reports")
    print("-" * 40)
    print("  - Look for monthly tourist arrival reports")
    print("  - Download PDFs for each month you need")
    print("  - Suggested: Download all available reports for comprehensive data")
    print()
    print("STEP 3: Save to Correct Directory")
    print("-" * 40)
    raw_data_dir = os.path.join(os.getcwd(), "data", "raw")
    print(f"  - Save all PDF files to: {raw_data_dir}")
    print("  - Any .pdf filename is acceptable")
    print("  - The script will automatically extract data from all PDFs")
    print()
    print("STEP 4: Run the Ingestion Script")
    print("-" * 40)
    print("  Execute: python scripts/ingest_tourism_data.py")
    print("  - The script will process all PDFs in data/raw/")
    print("  - Extracted data will be saved to data/processed/monthly_tourist_arrivals.csv")
    print()
    print("ALTERNATIVE: Using wget or curl")
    print("-" * 40)
    print("  If you have direct access to PDF URLs, you can use:")
    print()
    print("  wget -P data/raw/ <PDF_URL>")
    print("  or")
    print("  curl -o data/raw/report.pdf <PDF_URL>")
    print()
    print("TIPS:")
    print("-" * 40)
    print("  - The script handles various PDF formats automatically")
    print("  - Duplicate months are handled (first occurrence kept)")
    print("  - The script can be re-run safely; it skips existing PDFs")
    print("  - Check the processed CSV for completeness after running")
    print()
    print("=" * 80)
    print()

def check_pdf_status():
    """Check the current status of PDF files."""
    raw_data_dir = os.path.join(os.getcwd(), "data", "raw")
    processed_data_dir = os.path.join(os.getcwd(), "data", "processed")
    
    print("\n" + "=" * 80)
    print("CURRENT STATUS")
    print("=" * 80)
    
    # Check raw directory
    if not os.path.exists(raw_data_dir):
        print(f"\n⚠ Raw data directory does not exist: {raw_data_dir}")
        print(f"  Run: mkdir -p {raw_data_dir}")
    else:
        pdf_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.pdf')]
        print(f"\nPDF files in {raw_data_dir}:")
        if pdf_files:
            print(f"  ✓ Found {len(pdf_files)} PDF files:")
            for i, pdf in enumerate(sorted(pdf_files)[:10], 1):
                size = os.path.getsize(os.path.join(raw_data_dir, pdf))
                print(f"    {i}. {pdf} ({size:,} bytes)")
            if len(pdf_files) > 10:
                print(f"    ... and {len(pdf_files) - 10} more files")
        else:
            print("  ✗ No PDF files found")
            print("  → Please download PDF files and place them in this directory")
    
    # Check processed directory
    if os.path.exists(processed_data_dir):
        csv_file = os.path.join(processed_data_dir, "monthly_tourist_arrivals.csv")
        if os.path.exists(csv_file):
            size = os.path.getsize(csv_file)
            print(f"\nProcessed data:")
            print(f"  ✓ CSV file exists: {csv_file}")
            print(f"  ✓ File size: {size:,} bytes")
            
            # Try to read and show summary
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                print(f"  ✓ Records: {len(df)}")
                if len(df) > 0:
                    print(f"  ✓ Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
            except Exception as e:
                print(f"  ⚠ Could not read CSV: {e}")
        else:
            print(f"\nProcessed data:")
            print("  ✗ No CSV file found")
            print("  → Run the ingestion script to process PDFs")
    
    print("\n" + "=" * 80)
    print()

def main():
    """Main function."""
    print_instructions()
    check_pdf_status()
    
    print("\nREADY TO PROCEED?")
    print("-" * 40)
    print("If you have downloaded PDFs, run:")
    print("  python scripts/ingest_tourism_data.py")
    print()
    print("If you need to download PDFs, follow the instructions above.")
    print()

if __name__ == "__main__":
    main()
