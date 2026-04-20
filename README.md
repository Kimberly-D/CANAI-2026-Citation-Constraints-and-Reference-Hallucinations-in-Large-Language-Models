# Reference Verification Tool

This repository contains the code used for the reference verification component of the paper.

## Main script
`reference_verifier.py`

## Requirements
- Python 3
- requests

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python reference_verifier.py references.txt
```

You can also provide references through standard input:

```bash
python reference_verifier.py
```

## Output
The script generates:
- `*_results.csv`
- `*_summary.txt`

## What the tool does
This tool performs heuristic metadata cross-checking against Crossref, OpenAlex, and arXiv. When a DOI is present in the reference text, the tool also attempts a direct DOI lookup.

For each reference, the script:
1. extracts metadata heuristically from the reference text
2. queries Crossref, OpenAlex, and arXiv for possible matches
3. compares returned records against the extracted metadata
4. computes an Integrity Score from 0 to 100
5. assigns a verification status and exports detailed results

## Notes
- Results may vary slightly over time because external data sources change.
- The verifier is a heuristic cross-checking tool and should not be treated as a definitive ground-truth oracle for every individual citation.
- Borderline cases such as unusually formatted citations, books, preprints, and web-based sources may require cautious interpretation.

## Contact email for API requests
The script does not hardcode a personal email address. Users can optionally set an environment variable before running the tool:

```bash
ARV_CONTACT_EMAIL=you@example.com python reference_verifier.py references.txt
```

On Windows PowerShell:

```powershell
$env:ARV_CONTACT_EMAIL="you@example.com"
python reference_verifier.py references.txt
```
