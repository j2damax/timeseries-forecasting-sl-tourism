# How to Create GitHub Issues from Templates

This guide explains how to use the issue templates to create GitHub issues for the Sri Lanka Tourism Forecasting project.

## Files Overview

1. **GITHUB_ISSUES_TEMPLATE.md** - Contains 12 detailed issues with full implementation tasks
2. **ISSUES_SUMMARY.md** - Quick reference guide with priorities and workflow
3. **HOW_TO_CREATE_ISSUES.md** - This file - instructions for creating issues

## Method 1: Manual Creation via GitHub Web Interface

### Step 1: Navigate to Issues Page
1. Go to repository: https://github.com/j2damax/timeseries-forecasting-sl-tourism
2. Click on "Issues" tab
3. Click "New issue" button

### Step 2: Create Each Issue
For each of the 12 issues in `GITHUB_ISSUES_TEMPLATE.md`:

1. **Title**: Copy the issue title (e.g., "Data Ingestion Pipeline Validation and Enhancement")
2. **Labels**: Add the labels specified in the issue (e.g., `enhancement`, `data-pipeline`, `high-priority`)
3. **Description**: Copy the entire issue content from the template, including:
   - Description
   - Context
   - References
   - Implementation Tasks (with checkboxes)
   - Acceptance Criteria
   - Any additional sections
4. **Assignee**: Assign to yourself or leave unassigned
5. Click "Submit new issue"

### Step 3: Repeat for All 12 Issues
Create issues in priority order (see ISSUES_SUMMARY.md):
- High Priority: Issues 1-5, 7-8
- Medium Priority: Issues 6, 9-11
- Low Priority: Issue 12

## Method 2: Using GitHub CLI (gh)

If you have GitHub CLI installed, you can create issues from the command line:

```bash
# Install gh CLI if needed
# https://cli.github.com/

# Authenticate
gh auth login

# Navigate to repository directory
cd /path/to/timeseries-forecasting-sl-tourism

# Create issues using the template
# You'll need to extract each issue's content and create a separate file for each

# Example for Issue 1
gh issue create \
  --title "Data Ingestion Pipeline Validation and Enhancement" \
  --label "enhancement,data-pipeline,high-priority" \
  --body-file issue_01_body.md

# Repeat for each issue
```

## Method 3: Using GitHub API (Advanced)

You can use the GitHub API with a script to create all issues programmatically:

```bash
# Create a script to parse GITHUB_ISSUES_TEMPLATE.md and create issues
# This requires:
# 1. Personal access token
# 2. Python/Node.js script to parse markdown and call API
# 3. GitHub REST API knowledge

# Example using curl:
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  https://api.github.com/repos/j2damax/timeseries-forecasting-sl-tourism/issues \
  -d '{"title":"Issue Title","body":"Issue body","labels":["label1","label2"]}'
```

## Recommended Approach

**For this project, we recommend Method 1 (Manual Creation)** because:
1. Only 12 issues to create (manageable manually)
2. Allows review of each issue before creation
3. Can customize based on current project state
4. No additional tools required
5. Easy to add comments or context during creation

## Issue Creation Checklist

Use this checklist when creating each issue:

- [ ] Issue 1: Data Ingestion Pipeline Validation and Enhancement
  - Labels: `enhancement`, `data-pipeline`, `high-priority`
  - [ ] Copy title and full content from template
  - [ ] Verify all implementation tasks included
  - [ ] Submit issue

- [ ] Issue 2: Exploratory Data Analysis (EDA) Implementation
  - Labels: `notebook`, `data-analysis`, `high-priority`
  - [ ] Copy title and full content from template
  - [ ] Verify all implementation tasks included
  - [ ] Submit issue

- [ ] Issue 3: Feature Engineering Implementation
  - Labels: `notebook`, `feature-engineering`, `high-priority`
  - [ ] Copy title and full content from template
  - [ ] Verify all implementation tasks included
  - [ ] Submit issue

- [ ] Issue 4: Facebook Prophet Model Development
  - Labels: `notebook`, `model-development`, `prophet`, `high-priority`
  - [ ] Copy title and full content from template
  - [ ] Verify all implementation tasks included
  - [ ] Submit issue

- [ ] Issue 5: LSTM Model Development
  - Labels: `notebook`, `model-development`, `deep-learning`, `lstm`, `high-priority`
  - [ ] Copy title and full content from template
  - [ ] Verify all implementation tasks included
  - [ ] Submit issue

- [ ] Issue 6: Amazon Chronos Model Development
  - Labels: `notebook`, `model-development`, `deep-learning`, `chronos`, `medium-priority`
  - [ ] Copy title and full content from template
  - [ ] Verify all implementation tasks included
  - [ ] Submit issue

- [ ] Issue 7: N-HiTS Model Development (Novel Approach)
  - Labels: `notebook`, `model-development`, `deep-learning`, `n-hits`, `high-priority`
  - [ ] Copy title and full content from template
  - [ ] Verify all implementation tasks included
  - [ ] Submit issue

- [ ] Issue 8: Model Comparison and Analysis
  - Labels: `notebook`, `evaluation`, `analysis`, `high-priority`
  - [ ] Copy title and full content from template
  - [ ] Verify all implementation tasks included
  - [ ] Submit issue

- [ ] Issue 9: Preprocessing Module Enhancement
  - Labels: `enhancement`, `utility-scripts`, `medium-priority`
  - [ ] Copy title and full content from template
  - [ ] Verify all implementation tasks included
  - [ ] Submit issue

- [ ] Issue 10: Evaluation Module Enhancement
  - Labels: `enhancement`, `utility-scripts`, `medium-priority`
  - [ ] Copy title and full content from template
  - [ ] Verify all implementation tasks included
  - [ ] Submit issue

- [ ] Issue 11: Documentation and README Updates
  - Labels: `documentation`, `medium-priority`
  - [ ] Copy title and full content from template
  - [ ] Verify all implementation tasks included
  - [ ] Submit issue

- [ ] Issue 12: Testing and Validation Framework
  - Labels: `testing`, `quality-assurance`, `low-priority`
  - [ ] Copy title and full content from template
  - [ ] Verify all implementation tasks included
  - [ ] Submit issue

## After Creating Issues

### 1. Organize with Projects (Optional)
Create a GitHub Project board to track progress:
- **Backlog**: All issues
- **In Progress**: Currently working on
- **In Review**: Completed, awaiting review
- **Done**: Completed and verified

### 2. Set Milestones (Optional)
Create milestones for major phases:
- Milestone 1: Data Foundation (Issues 1-3)
- Milestone 2: Model Development (Issues 4-7)
- Milestone 3: Analysis (Issue 8)
- Milestone 4: Infrastructure (Issues 9-10)
- Milestone 5: Documentation (Issues 11-12)

### 3. Link Related Issues
In issue descriptions, reference related issues:
- Issue 8 (Model Comparison) depends on Issues 4-7 (all models)
- Issues 9-10 (utility modules) support Issues 2-8
- Issue 11 (documentation) comes after all implementation

### 4. Track Progress
As you work through issues:
1. Check off implementation task items
2. Add comments with progress updates
3. Close issues when all acceptance criteria met
4. Link commits and PRs to issues

## Tips for Issue Management

1. **Start with High-Priority Issues**: Focus on Issues 1-5, 7-8 first
2. **Work Sequentially**: Complete data issues before model issues
3. **Update Task Lists**: Check off items as you complete them
4. **Add Context**: Comment on issues with findings or challenges
5. **Link PRs**: Reference issues in commit messages (`Closes #1`, `Part of #2`)
6. **Keep Issues Focused**: Don't combine multiple issues into one PR
7. **Review Regularly**: Check issue list to stay on track

## Template Customization

Feel free to customize issues based on:
- Current project state (some work may already be done)
- Available time and resources
- Specific requirements or constraints
- Team preferences and workflow

## References

All issues reference these key documents:
- `.github/copilot-instructions.md` - Project conventions
- `.github/agents/tourism-forecasting.md` - Detailed requirements
- `.github/agents/README.md` - Agent instructions overview
- `README.md` - Project overview
- `scripts/README.md` - Script documentation

## Questions?

If you have questions about:
- **Issue content**: Refer to `GITHUB_ISSUES_TEMPLATE.md` for details
- **Priorities**: See `ISSUES_SUMMARY.md` for workflow
- **Implementation**: Check referenced documentation files
- **Technical details**: See `.github/agents/tourism-forecasting.md`

---

## Quick Start

**To get started right now:**

1. Open `GITHUB_ISSUES_TEMPLATE.md`
2. Copy content for Issue 1
3. Go to https://github.com/j2damax/timeseries-forecasting-sl-tourism/issues/new
4. Paste content and add labels
5. Submit issue
6. Repeat for remaining 11 issues

**Estimated time**: ~20-30 minutes to create all 12 issues manually.

---

Happy issue creation! 🚀
