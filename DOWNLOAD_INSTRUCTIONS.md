# Data Download Instructions

## Overview

This document provides instructions for downloading PDF files from the Sri Lanka Tourism Development Authority (SLTDA) website and running the data ingestion pipeline.

## Current Status

✅ **Data Ingestion Script**: Fully functional with enhanced error handling and retry logic
✅ **Data Extraction**: Successfully tested with sample PDFs
✅ **CSV Generation**: Working correctly
⚠️ **Automated Download**: Limited by network access restrictions in some environments

## Quick Start

### Option 1: Automated Download (Recommended)

If you have unrestricted internet access:

```bash
# Run the complete pipeline
python scripts/ingest_tourism_data.py
```

The script will:
1. Scrape the SLTDA website for PDF links
2. Download all monthly tourist arrival reports
3. Extract data from each PDF
4. Generate consolidated CSV file

### Option 2: Manual Download (If Automated Fails)

If automated download fails due to network restrictions:

1. **Visit the SLTDA website:**
   ```
   https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports
   ```

2. **Download PDF reports:**
   - Navigate through yearly report pages
   - Download monthly tourist arrival PDFs
   - Save all files to: `data/raw/`
   - Any filename is acceptable (e.g., `january_2024.pdf`)

3. **Run the ingestion script:**
   ```bash
   python scripts/ingest_tourism_data.py
   ```
   
   The script will automatically:
   - Skip the download phase if PDFs exist
   - Process all PDFs in `data/raw/`
   - Extract year, month, and arrival data
   - Save results to `data/processed/monthly_tourist_arrivals.csv`

### Option 3: Use Helper Script

For guidance and status checks:

```bash
python scripts/download_helper.py
```

This script provides:
- Step-by-step download instructions
- Current status of downloaded PDFs
- Verification of processed data
- Tips and troubleshooting

## Script Features

### Enhanced Error Handling

The `ingest_tourism_data.py` script includes:

- **Retry Logic**: Automatically retries failed downloads (3 attempts with exponential backoff)
- **User-Agent Headers**: Mimics browser requests to avoid blocking
- **Graceful Degradation**: Continues processing even if some PDFs fail
- **Informative Messages**: Clear feedback on download and extraction status
- **Resume Capability**: Skips already downloaded PDFs

### Data Extraction

The script handles various PDF formats and extracts:
- Year (4-digit number: 2019-2099)
- Month (name or number: January, February, etc.)
- Total Arrivals (handles comma-separated numbers and various formats)

### Data Processing

- Creates proper datetime index
- Removes duplicates (keeps first occurrence)
- Handles missing values
- Sorts chronologically
- Generates clean CSV with `Date` and `Arrivals` columns

## Output

### Raw Data
- **Location**: `data/raw/`
- **Format**: PDF files
- **Note**: Excluded from version control (see `.gitignore`)

### Processed Data
- **Location**: `data/processed/monthly_tourist_arrivals.csv`
- **Format**: CSV with columns:
  - `Date`: First day of month (YYYY-MM-DD)
  - `Arrivals`: Total tourist arrivals (integer)

### Example Output
```csv
Date,Arrivals
2023-01-01,82327
2023-02-01,96507
2023-03-01,106500
...
```

## Troubleshooting

### Network Access Issues

If you see errors like:
```
Failed to resolve 'www.sltda.gov.lk'
HTTPSConnectionPool: Max retries exceeded
```

**Solution**: The website is unreachable from your environment. Use Manual Download (Option 2).

### PDF Extraction Failures

If some PDFs fail to extract:
```
Failed to extract data from report_xyz.pdf
```

**Possible causes**:
- PDF is corrupted
- PDF has non-standard format
- PDF is encrypted or password-protected

**Solution**: 
- Try re-downloading the problematic PDF
- Check if the PDF opens correctly in a PDF viewer
- The script will continue processing other PDFs

### Empty CSV Output

If the CSV is empty or has fewer records than expected:

**Check**:
1. Verify PDFs are in `data/raw/`
2. Check console output for extraction errors
3. Ensure PDFs contain the expected data format

**Debug**:
```bash
python scripts/download_helper.py  # Check status
```

## Network Restrictions

Some environments have limited internet access or firewall restrictions that block the SLTDA website. This is a known limitation in:

- Sandboxed environments
- Corporate networks with strict firewalls
- Some cloud computing platforms
- Containers with network policies

In such cases, manual download is the recommended approach.

## Sample Data

For testing purposes, the script can be tested with sample PDFs. To create sample data:

```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Create a sample PDF
c = canvas.Canvas("data/raw/sample_jan_2024.pdf", pagesize=letter)
width, height = letter
c.setFont("Helvetica-Bold", 16)
c.drawString(100, height - 100, "SRI LANKA TOURISM DEVELOPMENT AUTHORITY")
c.setFont("Helvetica", 14)
c.drawString(100, height - 130, "Monthly Tourist Arrivals Report")
c.drawString(100, height - 160, "January 2024")
c.setFont("Helvetica", 12)
c.drawString(100, height - 200, "Total Arrivals: 102,345")
c.save()
```

## Best Practices

1. **Regular Updates**: Run the script monthly to get the latest data
2. **Backup**: Keep a backup of the processed CSV file
3. **Validation**: Check the date range and record count after processing
4. **Version Control**: The processed CSV is excluded from git (see `.gitignore`)
5. **Error Logs**: Review console output for any warnings or errors

## Dependencies

Required Python packages:
- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing
- `pandas` - Data manipulation
- `pdfplumber` - PDF text extraction
- `tqdm` - Progress bars

Install with:
```bash
pip install requests beautifulsoup4 pandas pdfplumber tqdm
```

Or use the project requirements:
```bash
pip install -r requirements.txt
```

## Support

For issues or questions:
1. Check this documentation
2. Run the helper script: `python scripts/download_helper.py`
3. Review the console output for specific error messages
4. Check the repository issues for similar problems

## Related Files

- `scripts/ingest_tourism_data.py` - Main ingestion script
- `scripts/download_helper.py` - Helper utility for manual downloads
- `scripts/README.md` - Technical documentation
- `.gitignore` - Excludes PDF and CSV files from version control
