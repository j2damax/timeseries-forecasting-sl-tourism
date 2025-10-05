"""
Data ingestion script for Sri Lanka Tourism Development Authority (SLTDA) reports.

This script scrapes the SLTDA website, downloads monthly tourist arrival reports (PDFs),
extracts the relevant data, and compiles it into a consolidated time series dataset.
"""

import os
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pdfplumber
from typing import List, Dict, Optional
from tqdm import tqdm


# Constants
BASE_URL = "https://www.sltda.gov.lk"
TARGET_URL = f"{BASE_URL}/en/monthly-tourist-arrivals-reports"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_CSV = "monthly_tourist_arrivals.csv"


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print(f"Created directories: {RAW_DATA_DIR}, {PROCESSED_DATA_DIR}")


def get_yearly_report_pages(url: str, retries: int = 3) -> List[str]:
    """
    Scrape the main page to find all yearly report page links.
    
    Parameters
    ----------
    url : str
        The main URL of the monthly tourist arrivals reports page
    retries : int
        Number of retry attempts
        
    Returns
    -------
    List[str]
        List of URLs to yearly report pages
    """
    import time
    
    print(f"Fetching main page: {url}")
    
    for attempt in range(retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links that contain yearly report pages
            yearly_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                # Look for links containing 'monthly-tourist-arrivals-reports' followed by a year
                if 'monthly-tourist-arrivals-reports' in href and re.search(r'20\d{2}', href):
                    full_url = href if href.startswith('http') else f"{BASE_URL}{href}"
                    if full_url not in yearly_links:
                        yearly_links.append(full_url)
            
            print(f"Found {len(yearly_links)} yearly report pages")
            return yearly_links
        except Exception as e:
            if attempt < retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"Error fetching main page (attempt {attempt + 1}/{retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch main page after {retries} attempts: {e}")
                print("\n" + "="*70)
                print("WEBSITE ACCESS ISSUE DETECTED")
                print("="*70)
                print("The SLTDA website appears to be unreachable from this environment.")
                print("This could be due to:")
                print("  1. Network restrictions or firewall blocking")
                print("  2. Website temporarily down")
                print("  3. DNS resolution issues")
                print("\nSuggested solutions:")
                print("  1. Manually download PDF files from:")
                print("     https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports")
                print("  2. Place the PDF files in: data/raw/")
                print("  3. Re-run this script - it will skip download and process existing PDFs")
                print("="*70 + "\n")
                return []
    
    return []


def get_pdf_links_from_page(url: str, retries: int = 3) -> List[str]:
    """
    Get all PDF links from a yearly report page.
    
    Parameters
    ----------
    url : str
        URL of the yearly report page
    retries : int
        Number of retry attempts
        
    Returns
    -------
    List[str]
        List of PDF URLs
    """
    import time
    
    print(f"Fetching PDF links from: {url}")
    
    for attempt in range(retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            pdf_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.lower().endswith('.pdf'):
                    full_url = href if href.startswith('http') else f"{BASE_URL}{href}"
                    pdf_links.append(full_url)
            
            print(f"Found {len(pdf_links)} PDF files on this page")
            return pdf_links
        except Exception as e:
            if attempt < retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"Error fetching PDF links (attempt {attempt + 1}/{retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch PDF links after {retries} attempts: {e}")
                return []
    
    return []


def download_pdf(url: str, output_path: str, retries: int = 3) -> bool:
    """
    Download a PDF file from a URL with retry logic.
    
    Parameters
    ----------
    url : str
        URL of the PDF file
    output_path : str
        Path to save the downloaded PDF
    retries : int
        Number of retry attempts
        
    Returns
    -------
    bool
        True if download was successful, False otherwise
    """
    import time
    
    for attempt in range(retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=60, headers=headers)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            if attempt < retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"Error downloading {url} (attempt {attempt + 1}/{retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {url} after {retries} attempts: {e}")
                return False
    
    return False


def download_all_pdfs():
    """
    Phase 1: Scrape and download all PDF reports.
    """
    print("\n=== Phase 1: Scraping and Downloading PDF Reports ===\n")
    
    # Check if PDFs already exist
    existing_pdfs = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.pdf')]
    if existing_pdfs:
        print(f"Found {len(existing_pdfs)} existing PDF files in {RAW_DATA_DIR}")
        print("These will be processed. To re-download, delete them first.\n")
    
    # Get yearly report pages
    yearly_pages = get_yearly_report_pages(TARGET_URL)
    
    # Also include the main page itself
    all_pages = [TARGET_URL] + yearly_pages
    
    # Collect all PDF links
    all_pdf_links = []
    for page_url in all_pages:
        pdf_links = get_pdf_links_from_page(page_url)
        all_pdf_links.extend(pdf_links)
    
    # Remove duplicates
    all_pdf_links = list(set(all_pdf_links))
    
    if all_pdf_links:
        print(f"\nTotal unique PDF files found: {len(all_pdf_links)}")
        
        # Download each PDF
        print("\nDownloading PDF files...")
        downloaded_count = 0
        skipped_count = 0
        failed_count = 0
        
        for idx, pdf_url in enumerate(tqdm(all_pdf_links), start=1):
            filename = f"report_{idx:03d}.pdf"
            output_path = os.path.join(RAW_DATA_DIR, filename)
            
            if os.path.exists(output_path):
                # Check if file is not empty
                if os.path.getsize(output_path) > 0:
                    skipped_count += 1
                    continue
                else:
                    # Re-download if file is empty
                    print(f"\nRe-downloading {filename} (previous file was empty)")
            
            if download_pdf(pdf_url, output_path):
                downloaded_count += 1
            else:
                failed_count += 1
        
        print(f"\nDownload Summary:")
        print(f"  - Successfully downloaded: {downloaded_count}")
        print(f"  - Skipped (already exist): {skipped_count}")
        print(f"  - Failed: {failed_count}")
        print(f"  - Total PDF files in {RAW_DATA_DIR}: {len([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.pdf')])}")
    else:
        print("\nNo PDF links found to download.")
        existing_count = len([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.pdf')])
        if existing_count > 0:
            print(f"However, {existing_count} PDF files already exist in {RAW_DATA_DIR}")
            print("Will proceed with data extraction from existing files.")
        else:
            print("\n" + "!"*70)
            print("WARNING: No PDFs available for processing!")
            print("!"*70)
            print("Please manually download PDF files from the SLTDA website and")
            print(f"place them in: {RAW_DATA_DIR}/")
            print("Then re-run this script.")
            print("!"*70 + "\n")


def extract_month_year_from_text(text: str) -> Optional[tuple]:
    """
    Extract month and year from PDF text.
    
    Parameters
    ----------
    text : str
        Text content from PDF
        
    Returns
    -------
    Optional[tuple]
        Tuple of (year, month) if found, None otherwise
    """
    # Common month patterns
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    text_lower = text.lower()
    
    # Try to find year (4-digit number starting with 19 or 20)
    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    if not year_match:
        return None
    
    year = int(year_match.group())
    
    # Try to find month name
    for month_name, month_num in months.items():
        if month_name in text_lower:
            return (year, month_num)
    
    # Try to find month as number (e.g., "Month 01" or "01/2023")
    month_match = re.search(r'\b(0?[1-9]|1[0-2])\b', text)
    if month_match:
        month = int(month_match.group())
        if 1 <= month <= 12:
            return (year, month)
    
    return None


def extract_total_arrivals_from_text(text: str) -> Optional[int]:
    """
    Extract total tourist arrivals from PDF text.
    
    Parameters
    ----------
    text : str
        Text content from PDF
        
    Returns
    -------
    Optional[int]
        Total arrivals if found, None otherwise
    """
    # Search for keywords indicating total arrivals
    keywords = [
        r'total\s+arrivals?',
        r'grand\s+total',
        r'total\s+tourists?',
        r'total',
    ]
    
    for keyword in keywords:
        # Look for the keyword followed by numbers
        # Try to match large numbers first (4+ digits), then numbers with commas
        patterns = [
            rf'{keyword}\s*[:,]?\s*(\d{{1,3}}(?:,\d{{3}})+)',  # Numbers with commas (e.g., 125,000)
            rf'{keyword}\s*[:,]?\s*(\d{{4,}})',  # Large numbers without commas (e.g., 150000)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract the number and remove commas
                number_str = match.group(1).replace(',', '')
                try:
                    return int(number_str)
                except ValueError:
                    continue
    
    # Alternative: Look for large numbers (likely to be tourist counts)
    # This is a fallback and may not be accurate
    numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})+)\b', text)
    if numbers:
        # Get the largest number (likely to be total)
        largest = max([int(n.replace(',', '')) for n in numbers])
        if largest > 1000:  # Reasonable threshold for tourist arrivals
            return largest
    
    return None


def extract_data_from_pdf(pdf_path: str) -> Optional[Dict]:
    """
    Extract year, month, and total arrivals from a PDF file.
    
    Parameters
    ----------
    pdf_path : str
        Path to the PDF file
        
    Returns
    -------
    Optional[Dict]
        Dictionary with 'year', 'month', and 'arrivals' if successful, None otherwise
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from all pages
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() or ""
            
            # Extract month and year
            month_year = extract_month_year_from_text(full_text)
            if not month_year:
                return None
            
            year, month = month_year
            
            # Extract total arrivals
            total_arrivals = extract_total_arrivals_from_text(full_text)
            if not total_arrivals:
                return None
            
            return {
                'year': year,
                'month': month,
                'arrivals': total_arrivals
            }
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None


def extract_all_data() -> List[Dict]:
    """
    Phase 2: Extract data from all PDF files.
    
    Returns
    -------
    List[Dict]
        List of dictionaries containing extracted data
    """
    print("\n=== Phase 2: Extracting Data from PDFs ===\n")
    
    arrivals_data = []
    
    # Get all PDF files in the raw data directory
    pdf_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.pdf')])
    
    if not pdf_files:
        print(f"No PDF files found in {RAW_DATA_DIR}")
        return arrivals_data
    
    print(f"Processing {len(pdf_files)} PDF files...")
    
    successful = 0
    failed = 0
    
    for pdf_file in tqdm(pdf_files):
        pdf_path = os.path.join(RAW_DATA_DIR, pdf_file)
        data = extract_data_from_pdf(pdf_path)
        
        if data:
            arrivals_data.append(data)
            successful += 1
        else:
            print(f"Failed to extract data from {pdf_file}")
            failed += 1
    
    print(f"\nSuccessfully extracted data from {successful} PDFs")
    print(f"Failed to extract data from {failed} PDFs")
    
    return arrivals_data


def consolidate_and_clean_data(arrivals_data: List[Dict]) -> pd.DataFrame:
    """
    Phase 3: Consolidate and clean the extracted data.
    
    Parameters
    ----------
    arrivals_data : List[Dict]
        List of dictionaries containing extracted data
        
    Returns
    -------
    pd.DataFrame
        Cleaned and consolidated DataFrame
    """
    print("\n=== Phase 3: Consolidating and Cleaning Data ===\n")
    
    if not arrivals_data:
        print("No data to consolidate")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(arrivals_data)
    print(f"Created DataFrame with {len(df)} rows")
    
    # Create Date column from year and month
    # Using month names for readability
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    
    df['month_name'] = df['month'].map(month_names)
    
    # Create datetime objects (first day of each month)
    df['Date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01')
    
    # Ensure arrivals is numeric
    df['Arrivals'] = pd.to_numeric(df['arrivals'], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna(subset=['Date', 'Arrivals'])
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['Date'], keep='first')
    
    # Select only Date and Arrivals columns
    df_final = df[['Date', 'Arrivals']].copy()
    
    # Sort by date
    df_final = df_final.sort_values('Date').reset_index(drop=True)
    
    print(f"Final dataset contains {len(df_final)} records")
    print(f"Date range: {df_final['Date'].min()} to {df_final['Date'].max()}")
    
    return df_final


def save_to_csv(df: pd.DataFrame):
    """
    Save the final DataFrame to CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    """
    if df.empty:
        print("No data to save")
        return
    
    output_path = os.path.join(PROCESSED_DATA_DIR, OUTPUT_CSV)
    df.to_csv(output_path, index=False)
    print(f"\nSaved final dataset to: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head(10))
    print("\nLast few rows:")
    print(df.tail(10))


def main():
    """
    Main function to orchestrate the data ingestion pipeline.
    """
    print("=" * 70)
    print("Sri Lanka Tourism Data Ingestion Pipeline")
    print("=" * 70)
    
    # Create directories
    create_directories()
    
    # Phase 1: Download PDFs
    download_all_pdfs()
    
    # Phase 2: Extract data from PDFs
    arrivals_data = extract_all_data()
    
    # Phase 3: Consolidate and clean data
    df_final = consolidate_and_clean_data(arrivals_data)
    
    # Save to CSV
    save_to_csv(df_final)
    
    print("\n" + "=" * 70)
    print("Data ingestion pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
