# PDF Analysis Process for Sri Lanka Tourism Data

## Overview

This document describes the process of analyzing and extracting monthly tourist arrival data from SLTDA (Sri Lanka Tourism Development Authority) PDF reports.

## Problem Statement

The task was to:
1. Analyze PDF files in `data/raw/` directory
2. Identify the data requirements from repository documentation
3. Extract monthly tourist arrival statistics
4. Create a CSV file for review before processing all PDFs

## Data Requirements

Based on the repository documentation (`scripts/README.md` and existing codebase), the required data format is:

```
Date, Arrivals
YYYY-MM-DD (datetime), <integer>
```

**Example:**
```
Date,Arrivals
2025-01-01,252761
2025-02-01,240217
```

### Required Columns:
- **Date**: First day of the month in datetime format (YYYY-MM-DD)
- **Arrivals**: Total international tourist arrivals for that month (integer)

## PDF Files Analyzed

Current PDF files in `data/raw/`:
- `01-2025.pdf` - January 2025
- `02-2025.pdf` - February 2025  
- `03-2025.pdf` - March 2025
- `04-2025.pdf` - April 2025

## PDF Structure Analysis

Each SLTDA PDF report follows this general structure:

### 1. Report Title
```
Monthly Tourist Arrivals Report [Month] [Year]
```

### 2. Summary Section
Contains narrative text describing the month's tourism performance with the total arrivals embedded in the text. Different months use different phrasings:

**January 2025:**
```
"Sri Lanka commenced 2025 with a remarkable influx of 252,761 tourists in January..."
```

**February 2025:**
```
"International tourist arrivals reached 240,217, a 10.01% increase..."
```

**March 2025:**
```
"...followed by February with 240,217 (+10.0%) and March with 229,298 (+9.62%)."
```

**April 2025:**
```
"April 2025 arrivals reached
174,608, representing a 17.3% increase..."
```
*(Note: Newline breaks between "reached" and the number)*

### 3. Tables and Charts
Statistical tables with monthly comparisons and source market breakdowns.

## Extraction Strategy

The `scripts/analyze_pdfs.py` script implements **7 different extraction patterns** to handle the varying formats:

### Pattern 1: "influx of X tourists"
```regex
influx\s+of\s+([\d,]+)\s+tourists?\s+in
```
Works for: January 2025

### Pattern 2: "arrivals reached X"
```regex
arrivals?\s+reached\s+([\d,]+)
```
Works for: February 2025 (simple cases)

### Pattern 3: "X, a Y% increase"
```regex
([\d,]+),\s+a\s+[\d.]+%\s+increase
```
Works for: February 2025

### Pattern 4: "welcomed X visitors"
```regex
welcomed\s+([\d,]+)\s+visitors?
```
Backup pattern for cumulative figures

### Pattern 5: "Month with/recorded X"
```regex
{month_name}\s+(?:with|recorded)\s+([\d,]+)
```
Works for: March 2025

### Pattern 6: "Month saw X arrivals"
```regex
{month_name}\s+(?:saw|recorded)\s+([\d,]+)\s+arrivals?
```
Additional pattern for narrative descriptions

### Pattern 7: "Month YYYY arrivals reached X" (with flexibility)
```regex
{month_name}\s+\d{4}\s+arrivals?\s+reached[\s\S]{0,200}?([\d,]+)
```
Works for: April 2025 (handles newlines and intermediate text)

## Validation

Each extracted value is validated:
1. **Range check**: 10,000 ≤ arrivals ≤ 500,000 (reasonable monthly range for Sri Lanka)
2. **Data type**: Must be a valid integer
3. **Format**: Removes commas and trailing punctuation

## Output Files

### 1. `monthly_tourist_arrivals_review.csv`
**Purpose**: For human review before processing all PDFs

**Format:**
```csv
Date,Arrivals
2025-01-01,252761
2025-02-01,240217
2025-03-01,229298
2025-04-01,174608
```

**Current Data:**
| Date       | Arrivals |
|------------|----------|
| 2025-01-01 | 252,761  |
| 2025-02-01 | 240,217  |
| 2025-03-01 | 229,298  |
| 2025-04-01 | 174,608  |

### 2. `extraction_details.csv`
**Purpose**: Detailed extraction metadata for debugging and verification

**Columns:**
- `file`: PDF filename
- `status`: SUCCESS, FAILED, WARNING, or ERROR
- `reason`: Description of extraction result
- `month`: Month name extracted
- `year`: Year extracted
- `month_num`: Month number (1-12)
- `arrivals`: Numeric value
- `arrivals_formatted`: Original formatted string (e.g., "252,761")
- `method`: Which extraction pattern was used

## Usage Instructions

### Run the Analysis Script

```bash
# From project root directory
python scripts/analyze_pdfs.py
```

### Expected Output

```
================================================================================
COMPREHENSIVE PDF DATA EXTRACTION ANALYSIS
================================================================================

Analyzing PDFs in: data/raw/
Found 4 PDF files to process

================================================================================
Processing: 01-2025.pdf
================================================================================
  ✅ January 2025
     Arrivals: 252,761 (252,761)
     Method: influx pattern
     
[... similar for other PDFs ...]

================================================================================
EXTRACTION SUMMARY
================================================================================
Total PDFs processed: 4
Successful extractions: 4
Warnings: 0
Failed extractions: 0

✅ Review CSV saved to: data/processed/monthly_tourist_arrivals_review.csv
✅ Extraction details saved to: data/processed/extraction_details.csv
```

## Verification Steps

1. **Review the CSV files:**
   ```bash
   cat data/processed/monthly_tourist_arrivals_review.csv
   cat data/processed/extraction_details.csv
   ```

2. **Verify data accuracy:**
   - Check that dates are sequential
   - Verify arrival numbers match PDF content
   - Ensure no duplicates or missing months

3. **Cross-reference with PDF:**
   - Open one or more PDFs manually
   - Compare extracted numbers with the summary section
   - Confirm extraction method makes sense

## Next Steps

Once the review CSV is validated:

1. **Upload more PDFs** to `data/raw/` directory
   - Download historical monthly reports from SLTDA website
   - Name them consistently (e.g., `05-2025.pdf`, `01-2024.pdf`, etc.)

2. **Re-run the analysis script:**
   ```bash
   python scripts/analyze_pdfs.py
   ```

3. **Use the data** in forecasting models:
   - The output CSV format matches the expected format for `data/processed/monthly_tourist_arrivals.csv`
   - Can be used directly in Jupyter notebooks (01_Data_Ingestion_and_EDA.ipynb, etc.)

## Integration with Existing Pipeline

The `analyze_pdfs.py` script complements the existing `ingest_tourism_data.py`:

### Differences:

| Feature | ingest_tourism_data.py | analyze_pdfs.py |
|---------|----------------------|-----------------|
| **Purpose** | Full pipeline (scrape→extract→save) | Focused PDF analysis only |
| **Web scraping** | Yes (downloads from SLTDA) | No (uses local PDFs) |
| **Review mode** | No | Yes (creates review CSV) |
| **Pattern matching** | 2 basic patterns | 7 advanced patterns |
| **Extraction details** | No | Yes (detailed CSV) |
| **Use case** | Production automation | Analysis & review |

### Recommendation:
- Use `analyze_pdfs.py` for **initial analysis and validation**
- Use `ingest_tourism_data.py` for **automated production updates**
- Both scripts produce compatible CSV outputs

## Troubleshooting

### PDF Extraction Fails

If a PDF fails to extract:

1. Check `extraction_details.csv` for the failure reason
2. Manually open the PDF and locate the arrivals number
3. Check which section contains the number (summary, table, etc.)
4. Add a new pattern to `analyze_pdfs.py` if needed

### Numbers Look Wrong

If extracted numbers seem incorrect:

1. Verify range: Should be 10k-500k for monthly Sri Lanka tourism
2. Check if it's cumulative (YTD) instead of monthly
3. Look at the context in `extraction_details.csv` to see which pattern matched
4. Review the PDF to confirm the correct number

### Missing Month/Year

If month or year extraction fails:

1. Check if the PDF follows a different title format
2. Look for "Monthly Tourist Arrivals Report [Month] [Year]"  pattern
3. Update the `month_year_match` regex if needed

## Technical Details

### Dependencies

```python
pdfplumber  # PDF text extraction
pandas      # Data manipulation
re          # Regular expressions for pattern matching
os          # File system operations
```

### Key Functions

- `extract_tourism_data_from_pdf()`: Core extraction logic
- `analyze_pdfs_in_directory()`: Batch processing
- `main()`: Entry point with output generation

## Data Quality Notes

### Strengths:
- ✅ All 4 current PDFs extracted successfully
- ✅ Multiple fallback patterns for robustness
- ✅ Validation ensures reasonable values
- ✅ Detailed logging for transparency

### Limitations:
- ⚠️ Assumes PDF text is extractable (not scanned images)
- ⚠️ Patterns are tuned for current SLTDA format
- ⚠️ May need updates if report format changes significantly

## Support

For questions or issues:
1. Review this documentation
2. Check `extraction_details.csv` for specific failures
3. Consult the existing `scripts/README.md` for general pipeline info
4. Refer to repository documentation in `TASK_COMPLETION_REPORT.md`

---

**Last Updated**: 2025-01-06  
**Script Version**: 1.0  
**PDFs Analyzed**: 4 (January-April 2025)
