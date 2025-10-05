# Data Ingestion Script

This directory contains the data ingestion pipeline for Sri Lanka tourism data.

## Script: `ingest_tourism_data.py`

This script automates the process of collecting monthly tourist arrival data from the Sri Lanka Tourism Development Authority (SLTDA) website.

### Features

The script performs three main phases:

1. **Phase 1: Scrape and Download PDF Reports**
   - Scrapes the SLTDA website for monthly tourist arrival reports
   - Downloads all PDF files to `data/raw/` directory
   - Files are named sequentially (e.g., `report_001.pdf`, `report_002.pdf`)

2. **Phase 2: Extract Data from PDFs**
   - Opens each PDF file using pdfplumber
   - Extracts month, year, and total tourist arrivals
   - Handles various report formats and text patterns
   - Gracefully handles errors for unreadable PDFs

3. **Phase 3: Consolidate and Clean Data**
   - Combines all extracted data into a single DataFrame
   - Creates a proper Date column (datetime format)
   - Removes duplicates and handles missing values
   - Sorts data chronologically
   - Saves the final dataset to `data/processed/monthly_tourist_arrivals.csv`

### Usage

```bash
# From the project root directory
python scripts/ingest_tourism_data.py
```

### Output

The script creates a CSV file at `data/processed/monthly_tourist_arrivals.csv` with the following structure:

| Date       | Arrivals |
|------------|----------|
| 2022-01-01 | 82327    |
| 2022-02-01 | 96507    |
| 2022-03-01 | 106500   |
| ...        | ...      |

### Dependencies

The script requires the following Python packages:
- `requests` - For HTTP requests
- `beautifulsoup4` - For HTML parsing
- `pandas` - For data manipulation
- `pdfplumber` - For PDF text extraction
- `tqdm` - For progress bars

Install dependencies:
```bash
pip install requests beautifulsoup4 pandas pdfplumber tqdm
```

### Directory Structure

```
data/
├── raw/              # Downloaded PDF files
│   ├── report_001.pdf
│   ├── report_002.pdf
│   └── ...
└── processed/        # Cleaned CSV data
    └── monthly_tourist_arrivals.csv
```

### Notes

- PDF files are stored in `data/raw/` and are excluded from version control via `.gitignore`
- The script can be re-run safely; it will skip already downloaded PDFs
- Data extraction uses pattern matching to identify total arrivals from various report formats
- The script handles duplicate entries by keeping only the first occurrence

### Troubleshooting

If you encounter issues:

1. **Connection errors**: Ensure you have internet access and the SLTDA website is accessible
2. **PDF extraction errors**: Some PDFs may have non-standard formats; these will be logged but won't stop the script
3. **Missing data**: Check the console output for any PDFs that failed to extract

### Testing

A test script is available to verify the data extraction functions:

```bash
python /tmp/test_ingestion.py
```

This tests the core extraction and consolidation logic without requiring network access.
