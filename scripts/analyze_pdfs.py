#!/usr/bin/env python3
"""
PDF Analysis Script for Sri Lanka Tourism Data

This script analyzes PDF files in data/raw/ directory and extracts monthly tourist arrival data.
It creates a review CSV file to validate the extraction before processing all PDFs.
"""

import pdfplumber
import re
import pandas as pd
import os

def extract_tourism_data_from_pdf(pdf_path):
    """
    Extract month, year, and tourist arrivals from a SLTDA PDF report.
    
    Parameters
    ----------
    pdf_path : str
        Path to the PDF file
        
    Returns
    -------
    dict or None
        Dictionary with extraction results, or None if extraction failed
    """
    month_to_num = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract all text from PDF
            full_text = ''.join([page.extract_text() or '' for page in pdf.pages])
            
            # Extract month and year from report title
            month_year_match = re.search(
                r'Monthly\s+Tourist\s+Arrivals\s+Report\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
                full_text,
                re.IGNORECASE
            )
            
            if not month_year_match:
                return {
                    'status': 'FAILED',
                    'reason': 'Month/Year not found in report title',
                    'month': None,
                    'year': None,
                    'arrivals': None,
                    'method': None
                }
            
            month_name = month_year_match.group(1).capitalize()
            year = month_year_match.group(2)
            month_num = month_to_num[month_name.lower()]
            
            # Multiple extraction strategies (in order of reliability)
            arrivals_str = None
            method = None
            
            # Strategy 1: "influx of XXX tourists in Month"
            pattern1 = rf'influx\s+of\s+([\d,]+)\s+tourists?\s+in'
            match1 = re.search(pattern1, full_text, re.IGNORECASE)
            if match1:
                arrivals_str = match1.group(1)
                method = "influx pattern"
            
            # Strategy 2: "arrivals reached XXX"
            if not arrivals_str:
                pattern2 = r'arrivals?\s+reached\s+([\d,]+)'
                match2 = re.search(pattern2, full_text, re.IGNORECASE)
                if match2:
                    arrivals_str = match2.group(1)
                    method = "arrivals reached pattern"
            
            # Strategy 3: "XXX, a X% increase" (first 2000 chars)
            if not arrivals_str:
                pattern3 = r'([\d,]+),\s+a\s+[\d.]+%\s+increase'
                match3 = re.search(pattern3, full_text[:2500], re.IGNORECASE)
                if match3:
                    arrivals_str = match3.group(1)
                    method = "percentage increase pattern"
            
            # Strategy 4: "welcomed XXX visitors" (first 2000 chars)
            if not arrivals_str:
                pattern4 = r'welcomed\s+([\d,]+)\s+visitors?'
                match4 = re.search(pattern4, full_text[:2500], re.IGNORECASE)
                if match4:
                    arrivals_str = match4.group(1)
                    method = "welcomed visitors pattern"
            
            # Strategy 5: Look for "March with XXX" or "March recorded XXX"
            if not arrivals_str:
                pattern5 = rf'{month_name}\s+(?:with|recorded)\s+([\d,]+)'
                match5 = re.search(pattern5, full_text[:3000], re.IGNORECASE)
                if match5:
                    arrivals_str = match5.group(1)
                    method = f"{month_name} with/recorded pattern"
            
            # Strategy 6: Look for "April saw XXX arrivals" or "April recorded XXX"
            if not arrivals_str:
                pattern6 = rf'{month_name}\s+(?:saw|recorded)\s+([\d,]+)\s+arrivals?'
                match6 = re.search(pattern6, full_text[:3000], re.IGNORECASE)
                if match6:
                    arrivals_str = match6.group(1)
                    method = f"{month_name} saw/recorded arrivals pattern"
            
            # Strategy 7: Look for "Month YYYY arrivals reached XXX" (allows text/newlines in between)
            if not arrivals_str:
                # Match "Month YYYY arrivals reached" followed by anything, then a number
                pattern7 = rf'{month_name}\s+\d{{4}}\s+arrivals?\s+reached[\s\S]{{0,200}}?([\d,]+)'
                match7 = re.search(pattern7, full_text[:4000], re.IGNORECASE)
                if match7:
                    # Remove any trailing commas and clean up
                    arrivals_str = match7.group(1).strip().rstrip(',')
                    # Verify it's a reasonable number
                    try:
                        test_val = int(arrivals_str.replace(',', ''))
                        if test_val > 100000 and test_val < 500000:
                            method = f"{month_name} YYYY arrivals reached pattern"
                        else:
                            arrivals_str = None  # Reset if not in range
                    except:
                        arrivals_str = None
            
            if arrivals_str:
                arrivals = int(arrivals_str.replace(',', ''))
                
                # Sanity check: monthly tourism arrivals for Sri Lanka
                # Reasonable range: 10,000 - 500,000 visitors per month
                if 10000 <= arrivals <= 500000:
                    return {
                        'status': 'SUCCESS',
                        'reason': 'Extraction successful',
                        'month': month_name,
                        'year': int(year),
                        'month_num': month_num,
                        'arrivals': arrivals,
                        'arrivals_formatted': arrivals_str,
                        'method': method
                    }
                else:
                    return {
                        'status': 'WARNING',
                        'reason': f'Value {arrivals:,} outside expected range (10k-500k)',
                        'month': month_name,
                        'year': int(year),
                        'month_num': month_num,
                        'arrivals': arrivals,
                        'arrivals_formatted': arrivals_str,
                        'method': method
                    }
            else:
                return {
                    'status': 'FAILED',
                    'reason': 'Arrivals number not found with any pattern',
                    'month': month_name,
                    'year': int(year),
                    'month_num': month_num,
                    'arrivals': None,
                    'method': None
                }
                
    except Exception as e:
        return {
            'status': 'ERROR',
            'reason': f'Exception: {str(e)}',
            'month': None,
            'year': None,
            'arrivals': None,
            'method': None
        }


def analyze_pdfs_in_directory(pdf_dir='data/raw'):
    """
    Analyze all PDFs in the specified directory and extract tourism data.
    
    Parameters
    ----------
    pdf_dir : str
        Directory containing PDF files
        
    Returns
    -------
    tuple
        (extracted_data_df, extraction_details_df)
    """
    print("="*80)
    print("COMPREHENSIVE PDF DATA EXTRACTION ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing PDFs in: {pdf_dir}/")
    
    # Get all PDF files
    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
    
    if not pdf_files:
        print(f"\n⚠️  No PDF files found in {pdf_dir}/")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Found {len(pdf_files)} PDF files to process\n")
    
    data_extracted = []
    extraction_details = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        
        print(f"{'='*80}")
        print(f"Processing: {pdf_file}")
        print('='*80)
        
        result = extract_tourism_data_from_pdf(pdf_path)
        result['file'] = pdf_file
        extraction_details.append(result)
        
        if result['status'] == 'SUCCESS':
            print(f"  ✅ {result['month']} {result['year']}")
            print(f"     Arrivals: {result['arrivals_formatted']} ({result['arrivals']:,})")
            print(f"     Method: {result['method']}")
            
            data_extracted.append({
                'Date': f"{result['year']}-{result['month_num']:02d}-01",
                'Arrivals': result['arrivals']
            })
        elif result['status'] == 'WARNING':
            print(f"  ⚠️  {result['month']} {result['year']}")
            print(f"     Arrivals: {result['arrivals_formatted']} ({result['arrivals']:,})")
            print(f"     Warning: {result['reason']}")
        else:
            print(f"  ❌ {result['status']}: {result['reason']}")
    
    # Create summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Successful extractions: {sum(1 for d in extraction_details if d['status'] == 'SUCCESS')}")
    print(f"Warnings: {sum(1 for d in extraction_details if d['status'] == 'WARNING')}")
    print(f"Failed extractions: {sum(1 for d in extraction_details if d['status'] in ['FAILED', 'ERROR'])}")
    
    # Create DataFrames
    if data_extracted:
        df = pd.DataFrame(data_extracted)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=['Date', 'Arrivals'])
    
    details_df = pd.DataFrame(extraction_details)
    
    return df, details_df


def main():
    """Main function to run the PDF analysis."""
    # Ensure directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Analyze PDFs
    data_df, details_df = analyze_pdfs_in_directory('data/raw')
    
    # Display results
    if not data_df.empty:
        print("\n" + "="*80)
        print("EXTRACTED DATA FOR REVIEW")
        print("="*80)
        print(data_df.to_string(index=False))
        
        # Save review CSV
        review_path = 'data/processed/monthly_tourist_arrivals_review.csv'
        data_df.to_csv(review_path, index=False)
        print(f"\n✅ Review CSV saved to: {review_path}")
    else:
        print("\n⚠️  No data was successfully extracted!")
    
    # Save extraction details
    details_path = 'data/processed/extraction_details.csv'
    details_df.to_csv(details_path, index=False)
    print(f"✅ Extraction details saved to: {details_path}")
    
    # Print next steps
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Review 'monthly_tourist_arrivals_review.csv' to verify accuracy")
    print("2. Check 'extraction_details.csv' to see extraction methods and any failures")
    print("3. If data looks good, upload more PDF files to data/raw/")
    print("4. Re-run this script to process all PDF files and create final dataset")
    print("="*80)


if __name__ == "__main__":
    main()
