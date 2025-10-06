# Quick Start Guide - PDF Analysis

## What Was Done

Analyzed 4 PDF files in `data/raw/` and extracted monthly tourist arrival data for Sri Lanka.

## Review the Results

```bash
# View the extracted data
cat data/processed/monthly_tourist_arrivals_review.csv

# View extraction details
cat data/processed/extraction_details.csv
```

## Current Data (January-April 2025)

| Month | Year | Arrivals |
|-------|------|----------|
| January | 2025 | 252,761 |
| February | 2025 | 240,217 |
| March | 2025 | 229,298 |
| April | 2025 | 174,608 |

## Next Steps

### Option 1: Add More PDFs and Re-analyze

1. Download additional monthly reports from SLTDA:
   - https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports

2. Save PDFs to `data/raw/` directory:
   ```bash
   # Example naming:
   # data/raw/05-2025.pdf
   # data/raw/06-2025.pdf
   # data/raw/01-2024.pdf
   # etc.
   ```

3. Re-run the analysis:
   ```bash
   python scripts/analyze_pdfs.py
   ```

### Option 2: Use the Existing Automated Pipeline

If you have internet access to SLTDA website:

```bash
python scripts/ingest_tourism_data.py
```

This will:
- Scrape the SLTDA website
- Download all available PDFs
- Extract data automatically
- Save to `data/processed/monthly_tourist_arrivals.csv`

## Files Created

| File | Purpose |
|------|---------|
| `scripts/analyze_pdfs.py` | PDF analysis script (reusable) |
| `data/processed/monthly_tourist_arrivals_review.csv` | Extracted data for review |
| `data/processed/extraction_details.csv` | Extraction metadata |
| `PDF_ANALYSIS_README.md` | Detailed technical documentation |
| `PDF_ANALYSIS_SUMMARY.md` | Executive summary |

## Data Format

The output CSV is in the required format for the forecasting pipeline:

```csv
Date,Arrivals
2025-01-01,252761
2025-02-01,240217
...
```

**Columns:**
- `Date`: First day of month (YYYY-MM-DD format)
- `Arrivals`: Total monthly tourist arrivals (integer)

## Compatibility

The extracted data is compatible with:
- ✅ Jupyter notebooks (`01_Data_Ingestion_and_EDA.ipynb`, etc.)
- ✅ `scripts/data_loader.py` functions
- ✅ All forecasting models (Prophet, LSTM, Chronos, N-HiTS)

## Troubleshooting

**If extraction fails for a PDF:**
1. Check `data/processed/extraction_details.csv` for the error
2. Manually open the PDF to verify the data
3. See `PDF_ANALYSIS_README.md` for detailed troubleshooting

**If you need more help:**
- Review `PDF_ANALYSIS_README.md` for technical details
- Check `scripts/README.md` for pipeline documentation
- See `TASK_COMPLETION_REPORT.md` for overall project context

## Success Metrics

Current extraction:
- ✅ **4/4 PDFs** successfully processed
- ✅ **100% success rate**
- ✅ All values validated (within 10k-500k range)
- ✅ Data verified against source PDFs

---

**Quick Commands:**

```bash
# View results
cat data/processed/monthly_tourist_arrivals_review.csv

# Re-run analysis (if you add more PDFs)
python scripts/analyze_pdfs.py

# Use automated pipeline (needs internet)
python scripts/ingest_tourism_data.py
```
