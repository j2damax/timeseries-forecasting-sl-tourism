# GitHub Issues Documentation

This directory contains comprehensive templates and guides for creating GitHub issues for the Sri Lanka Tourism Forecasting project.

## 📋 Files in This Collection

### 1. GITHUB_ISSUES_TEMPLATE.md
**The main template file with all 12 detailed issues.**

This file contains:
- Complete issue descriptions
- Detailed implementation tasks with checkboxes
- Acceptance criteria
- References to project documentation
- Model-specific considerations
- Technical requirements

Each issue is structured with:
```
## Issue N: Title
- Labels
- Description
- Context
- References
- Implementation Tasks (detailed checklist)
- Acceptance Criteria
- Additional Notes
```

### 2. ISSUES_SUMMARY.md
**Quick reference guide for all issues.**

Contains:
- Priority classification (High/Medium/Low)
- Quick issue summaries
- Implementation workflow by phase
- Key project constraints
- Success criteria
- Dependencies summary

Use this for:
- Understanding issue priorities
- Planning implementation phases
- Quick reference during development
- Tracking overall progress

### 3. HOW_TO_CREATE_ISSUES.md
**Step-by-step guide for creating issues.**

Includes:
- Three methods to create issues (Manual, CLI, API)
- Detailed checklist for each issue
- Tips for issue management
- Post-creation organization suggestions
- Links to relevant documentation

Use this to:
- Create issues on GitHub
- Organize project workflow
- Set up milestones and projects
- Track implementation progress

### 4. ISSUES_README.md
**This file - navigation guide for the documentation.**

---

## 🎯 Quick Start Guide

### If you want to CREATE issues on GitHub:
1. Read `HOW_TO_CREATE_ISSUES.md`
2. Open `GITHUB_ISSUES_TEMPLATE.md`
3. Copy each issue's content to GitHub

### If you want to UNDERSTAND the project scope:
1. Read `ISSUES_SUMMARY.md` for high-level overview
2. Review `GITHUB_ISSUES_TEMPLATE.md` for details
3. Check `.github/agents/tourism-forecasting.md` for full context

### If you want to START IMPLEMENTING:
1. Review priorities in `ISSUES_SUMMARY.md`
2. Start with Issue 1 (Data Ingestion)
3. Follow the sequential workflow (Issues 1→2→3→4→5→7→8)
4. Check off tasks as you complete them

---

## 📊 Project Overview

The Sri Lanka Tourism Forecasting project aims to:
- Forecast monthly tourist arrivals to Sri Lanka
- Compare multiple forecasting approaches
- Answer: Can deep learning outperform statistical methods?

### Models to Implement
1. **Prophet** - Statistical/ML approach with event handling
2. **LSTM** - Deep learning for complex patterns
3. **Chronos** - Pre-trained foundation model (zero-shot)
4. **N-HiTS** - State-of-the-art hierarchical neural architecture

### Project Phases
1. **Data Foundation** (Issues 1-3) - Ingest, explore, prepare data
2. **Model Development** (Issues 4-7) - Implement four models
3. **Evaluation** (Issue 8) - Compare and analyze
4. **Infrastructure** (Issues 9-10) - Build utilities
5. **Documentation** (Issue 11) - Ensure reproducibility
6. **Quality** (Issue 12) - Testing framework

---

## 🔑 Key Information

### Data Specifications
- **Source**: Sri Lanka Tourism Development Authority (SLTDA)
- **Time Range**: January 2000 - December 2024
- **Frequency**: Monthly
- **Training**: 2000-2022
- **Testing**: 2023 (12-month forecast)

### Evaluation Metrics
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

### Critical Conventions
- ✅ **Chronological splits** (never shuffle time series data)
- ✅ **Forward fill** for missing values
- ✅ **MinMaxScaler** for neural networks
- ✅ **Raw values** for Prophet/statistical models
- ✅ **Standard columns**: `Date` (datetime), `Arrivals` (int)

---

## 📚 Related Documentation

### Project Documentation
- `.github/copilot-instructions.md` - Development guidelines and patterns
- `.github/agents/tourism-forecasting.md` - Complete project requirements
- `.github/agents/README.md` - Agent instructions overview
- `README.md` - Main project documentation
- `scripts/README.md` - Data ingestion documentation

### Technical References
- **Prophet**: https://facebook.github.io/prophet/
- **LSTM**: TensorFlow/Keras documentation
- **Chronos**: https://github.com/amazon-science/chronos-forecasting
- **N-HiTS**: https://github.com/Nixtla/neuralforecast

---

## 🎓 Understanding the Issues

### Issue Priorities Explained

**High Priority (Must Complete First)**
- Issues 1-3: Data pipeline is foundation for everything
- Issues 4-5, 7: Core models for comparison
- Issue 8: Final analysis and conclusions

**Medium Priority (Important but Dependent)**
- Issue 6: Chronos adds value but not critical path
- Issues 9-10: Utilities support but not block implementation
- Issue 11: Documentation important for reproducibility

**Low Priority (Quality Improvement)**
- Issue 12: Testing helps maintain quality but not required for initial implementation

### Dependencies Between Issues

```
Issue 1 (Data Ingestion)
    ↓
Issue 2 (EDA)
    ↓
Issue 3 (Feature Engineering)
    ↓
Issues 4-7 (All Models in parallel)
    ↓
Issue 8 (Comparison)

Issues 9-10 (Utilities) ← Support all notebook issues
Issue 11 (Documentation) ← After all implementation
Issue 12 (Testing) ← Optional, for quality assurance
```

---

## 💡 Tips for Success

### For Implementation
1. **Follow the sequence**: Complete data issues before model issues
2. **Test early**: Run data ingestion first to verify pipeline
3. **Document as you go**: Update notebooks with findings
4. **Save artifacts**: Store trained models for comparison
5. **Use utilities**: Leverage `scripts/` modules for consistency

### For Issue Management
1. **Check off tasks**: Update checkboxes as you progress
2. **Add comments**: Document challenges and solutions
3. **Link commits**: Reference issue numbers in commit messages
4. **Close when done**: Verify acceptance criteria before closing
5. **Review regularly**: Keep track of overall progress

### For Code Quality
1. **Follow conventions**: See `.github/copilot-instructions.md`
2. **Reuse code**: Use `scripts/` utilities, don't duplicate
3. **Document choices**: Explain hyperparameters and decisions
4. **Visualize results**: Create clear, labeled plots
5. **Test thoroughly**: Verify train/test splits have no leakage

---

## 🚀 Getting Started

Ready to begin? Here's your action plan:

1. **Read the summary**: Start with `ISSUES_SUMMARY.md`
2. **Create issues**: Follow `HOW_TO_CREATE_ISSUES.md`
3. **Review templates**: Check `GITHUB_ISSUES_TEMPLATE.md` for details
4. **Start implementing**: Begin with Issue 1 (Data Ingestion)
5. **Track progress**: Check off tasks, add comments, close completed issues

---

## 📞 Questions or Issues?

If you encounter problems:
1. Check the detailed task lists in `GITHUB_ISSUES_TEMPLATE.md`
2. Review project conventions in `.github/copilot-instructions.md`
3. Consult requirements in `.github/agents/tourism-forecasting.md`
4. Add comments to GitHub issues with specific questions
5. Document challenges and solutions for others

---

## 📈 Success Criteria

The project is successful when:
- ✅ All 12 issues completed with acceptance criteria met
- ✅ Four models implemented and evaluated
- ✅ Comprehensive comparison analysis completed
- ✅ Research question answered with evidence
- ✅ Code is reproducible and well-documented
- ✅ Results provide actionable insights for tourism forecasting

---

**Last Updated**: 2025
**Project**: Sri Lanka Tourism Forecasting
**Repository**: j2damax/timeseries-forecasting-sl-tourism

---

Good luck with your implementation! 🎯
