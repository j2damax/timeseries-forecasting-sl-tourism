# Copilot Agent Instructions

This directory contains instructions for GitHub Copilot agents working on this time series forecasting project.

## Available Instructions

### tourism-forecasting.md
Comprehensive instructions for implementing the Sri Lankan tourism forecasting project. This includes:
- Complete project context and motivation
- Data specifications and requirements
- Model portfolio (SARIMA, Prophet, LSTM, Chronos, N-HiTS)
- Implementation guidelines and best practices
- Evaluation framework
- Research context and objectives

## How to Use

These instructions provide context for AI agents (like GitHub Copilot) to understand:
1. The overall project goals and constraints
2. The specific models to be implemented
3. The evaluation criteria for success
4. Best practices for code organization
5. The research questions being addressed

## Project Structure

```
timeseries-forecasting-sl-tourism/
├── .github/
│   └── agents/           # Agent instructions (this directory)
├── notebooks/            # Jupyter notebooks for analysis
│   ├── 01_Data_Ingestion_and_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Development_Prophet.ipynb
│   ├── 04_Model_Development_LSTM.ipynb
│   ├── 05_Model_Development_Chronos.ipynb
│   └── 06_Model_Development_Novel.ipynb
├── scripts/              # Utility scripts
│   ├── data_loader.py    # Data loading utilities
│   └── evaluation.py     # Model evaluation utilities
├── data/                 # Data directory (to be created)
└── requirements.txt      # Python dependencies
```

## Key References

The project is based on research into forecasting methodologies for Sri Lankan tourism, with particular focus on:
- SARIMA models as the established benchmark
- Facebook Prophet for interpretable forecasting
- LSTM networks for capturing non-linear patterns
- Amazon Chronos for zero-shot forecasting
- N-HiTS as a state-of-the-art neural architecture

For detailed information, see the project wiki at:
https://github.com/j2damax/timeseries-forecasting-sl-tourism/wiki
