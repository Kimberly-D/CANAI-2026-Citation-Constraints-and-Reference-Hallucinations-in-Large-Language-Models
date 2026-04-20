import csv
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import requests

# --- Configuration ---
# Public code should not hardcode a personal email address.
# Users can set ARV_CONTACT_EMAIL if they want to identify themselves to the APIs.
CROSSREF_API_URL = "https://api.crossref.org/works"
OPENALEX_API_URL = "https://api.openalex.org/works"
ARXIV_API_URL = "http://export.arxiv.org/api/query"
CONTACT_EMAIL = os.getenv("ARV_CONTACT_EMAIL", "contact@example.invalid")
USER_AGENT = f"AcademicReferenceVerifier/2.0 (mailto:{CONTACT_EMAIL})"
REQUEST_TIMEOUT = 15
TOP_K_CROSSREF = 5
TOP_K_OPENALEX = 5
TOP_K_ARXIV = 3
CSV_SUFFIX = "_results.csv"
SUMMARY_SUFFIX = "_summary.txt"
COMMON_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'in', 'of', 'to', 'for', 'with', 'on', 'at', 'by',
    'as', 'is', 'was', 'are', 'be', 'that', 'this', 'from', 'into', 'using', 'via',
    'their', 'its', 'our', 'your', 'we', 'they', 'these', 'those', 'than'
}
MIN_TITLE_SIMILARITY = 0.60
MIN_AUTHOR_SIMILARITY = 0.72
MIN_KEYWORD_COVERAGE = 0.60
MIN_VALID_SCORE = 70
HIGH_CONFIDENCE_SCORE = 85
MEDIUM_CONFIDENCE_SCORE = 70
LOW_MATCH_SCORE = 50
MIN_CROSSREF_SCORE = 5.0
HIGH_CONFIDENCE_CROSSREF_SCORE = 10.0
MEDIUM_CONFIDENCE_CROSSREF_SCORE = 8.0
LOW_CROSSREF_SCORE_PENALTY_THRESHOLD = 5.0


@dataclass
class Candidate:
    source: str
    title: str = "N/A"
    authors: str = "N/A"
    container_title: str = "N/A"
    year: str = "N/A"
    doi: str = "N/A"
    url: str = "N/A"
    source_score: Optional[float] = None
    raw_score_label: str = "N/A"
    direct_doi_verified: bool = False


def headers() -> Dict[str, str]:
    return {"User-Agent": USER_AGENT}


def base_status(status: str) -> str:
    return status.split(" (", 1)[0].strip()


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def similarity_ratio(left: Optional[str], right: Optional[str]) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def safe_year(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    m = re.search(r"\b(19\d{2}|20\d{2})\b", str(value))
    return int(m.group(1)) if m else None


def extract_year_heuristic(text: str) -> Optional[str]:
    cleaned = re.sub(r'(?i)(retrieved|accessed|available|viewed)\s+[a-z]*\s*\d{1,2},?\s*\d{4}', '', text)
    cleaned = re.sub(r'(?i),?\s*from\s+[a-z]*\s*\d{1,2},?\s*\d{4}', '', cleaned)
    cleaned = re.sub(r'https?://[^\s]+', '', cleaned)
    cleaned = re.sub(r'(?i)[a-z]+\.?\s+\d{1,2},?\s+\d{4}\s*$', '', cleaned)

    for segment in (text[:200], cleaned[:250], cleaned):
        match = re.search(r'\((19\d{2}|20\d{2})\)', segment)
        if match:
            return match.group(1)
    match = re.search(r'\b(19\d{2}|20\d{2})\b', cleaned[:150])
    return match.group(1) if match else None


def extract_first_author_lastname(text: str) -> Optional[str]:
    text = text.strip()
    patterns = [
        r'^([A-Z][A-Za-z\-\']+),\s*[A-Z]',
        r'^([A-Z][A-Za-z\-\']+)\s*,',
        r'^([A-Z][A-Za-z\-\']+)\s+\(',
        r'^([A-Z][A-Za-z\-\']+)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def extract_title_keywords(text: str) -> List[str]:
    year = extract_year_heuristic(text)
    if not year:
        return []

    year_index = text.find(year)
    if year_index == -1:
        return []

    start_index = text.find(')', year_index)
    start_index = start_index + 1 if start_index != -1 else year_index + 4
    trailing = text[start_index:].strip(' .')

    title_segment = trailing
    parts = re.split(r'\.\s+(?=[A-Z])', trailing, maxsplit=1)
    if parts:
        title_segment = parts[0]

    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-]+\b', title_segment.lower())
    return [word for word in words if word not in COMMON_WORDS and len(word) > 2]


def extract_doi_heuristic(text: str) -> Optional[str]:
    # Keep DOI matching permissive, but stop before obvious trailing punctuation.
    doi_pattern = r'(10\.\d{4,9}/[^\s"<>\]\[}{,;]+)'
    match = re.search(r'(?:doi[:=/\s]*|https?://doi\.org/)' + doi_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip(' .,:;').lower()
    bare = re.search(r'\b' + doi_pattern, text, re.IGNORECASE)
    return bare.group(1).strip(' .,:;').lower() if bare else None


def output_base() -> str:
    if len(sys.argv) > 1:
        return os.path.splitext(sys.argv[1])[0]
    return "results"


def parse_crossref_item(item: Dict, direct_doi_verified: bool = False) -> Candidate:
    authors = item.get('author', []) or []
    author_names = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in authors if a.get('family')]
    year = 'N/A'
    for field in ('published-print', 'published-online', 'issued', 'created'):
        date_parts = item.get(field, {}).get('date-parts', [])
        if date_parts and date_parts[0]:
            year = str(date_parts[0][0])
            break
    score = item.get('score')
    return Candidate(
        source='CR_DOI' if direct_doi_verified else 'CR',
        title=(item.get('title') or ['N/A'])[0],
        authors=", ".join(author_names) if author_names else 'N/A',
        container_title=(item.get('container-title') or ['N/A'])[0],
        year=year,
        doi=(item.get('DOI') or 'N/A').lower(),
        url=item.get('URL', 'N/A'),
        source_score=float(score) if isinstance(score, (int, float)) else None,
        raw_score_label=(f"{score:.2f}" if isinstance(score, (int, float)) else 'N/A'),
        direct_doi_verified=direct_doi_verified,
    )


def verify_doi_direct(doi: Optional[str]) -> Optional[Candidate]:
    if not doi:
        return None
    try:
        response = requests.head(f"https://doi.org/{doi}", headers=headers(), timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if response.status_code >= 400:
            return None
        cr_response = requests.get(f"{CROSSREF_API_URL}/{doi}", headers=headers(), timeout=REQUEST_TIMEOUT)
        if cr_response.status_code != 200:
            return None
        message = cr_response.json().get('message', {})
        return parse_crossref_item(message, direct_doi_verified=True)
    except requests.exceptions.RequestException:
        return None


def search_crossref(reference_text: str) -> List[Candidate]:
    try:
        response = requests.get(
            CROSSREF_API_URL,
            params={'query.bibliographic': reference_text, 'rows': TOP_K_CROSSREF},
            headers=headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        items = response.json().get('message', {}).get('items', [])
        return [parse_crossref_item(item) for item in items]
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError):
        return []


def parse_openalex_item(item: Dict) -> Candidate:
    authors = item.get('authorships', []) or []
    author_names = [a.get('author', {}).get('display_name') for a in authors if a.get('author')]
    year = str(item.get('publication_year') or 'N/A')
    host = item.get('primary_location', {}).get('source', {}) or {}
    doi = item.get('doi') or 'N/A'
    if doi != 'N/A':
        doi = doi.replace('https://doi.org/', '').lower()
    return Candidate(
        source='OA',
        title=item.get('title', 'N/A'),
        authors=", ".join(filter(None, author_names)) if author_names else 'N/A',
        container_title=host.get('display_name', 'N/A'),
        year=year,
        doi=doi,
        url=item.get('id', 'N/A'),
        raw_score_label='N/A',
    )


def search_openalex(reference_text: str) -> List[Candidate]:
    search_text = reference_text[:250]
    try:
        response = requests.get(
            OPENALEX_API_URL,
            params={'search': search_text, 'per-page': TOP_K_OPENALEX},
            headers={**headers(), 'mailto': CONTACT_EMAIL},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get('results', [])
        return [parse_openalex_item(item) for item in results]
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError):
        return []


def parse_arxiv_entry(entry: ET.Element) -> Candidate:
    atom_ns = '{http://www.w3.org/2005/Atom}'
    arxiv_ns = '{http://arxiv.org/schemas/atom}'
    title = (entry.findtext(f'{atom_ns}title', 'N/A') or 'N/A').strip().replace('\n', ' ')
    url = (entry.findtext(f'{atom_ns}id', 'N/A') or 'N/A').strip()
    published = (entry.findtext(f'{atom_ns}published', 'N/A') or 'N/A').strip()
    year = published[:4] if published != 'N/A' else 'N/A'
    doi = entry.findtext(f'{arxiv_ns}doi') or entry.findtext('doi') or 'N/A'
    doi = doi.strip().lower() if doi != 'N/A' else 'N/A'
    authors = ", ".join(
        (author.findtext(f'{atom_ns}name', '') or '').strip()
        for author in entry.findall(f'{atom_ns}author')
    ) or 'N/A'
    return Candidate(
        source='AX',
        title=title,
        authors=authors,
        container_title='arXiv',
        year=year,
        doi=doi,
        url=url,
        raw_score_label='N/A',
    )


def search_arxiv(reference_text: str) -> List[Candidate]:
    try:
        response = requests.get(
            ARXIV_API_URL,
            params={'search_query': f'all:"{reference_text[:250]}"', 'start': 0, 'max_results': TOP_K_ARXIV},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        atom_ns = '{http://www.w3.org/2005/Atom}'
        root = ET.fromstring(response.text)
        entries = root.findall(f'{atom_ns}entry')
        return [parse_arxiv_entry(entry) for entry in entries]
    except (requests.exceptions.RequestException, ET.ParseError, ValueError, TypeError):
        return []


def candidate_author_match(original_author: Optional[str], found_authors: str) -> Tuple[bool, float]:
    if not original_author or not found_authors or found_authors == 'N/A':
        return False, 0.0
    surname = normalize_text(original_author)
    parts = [part.strip() for part in found_authors.split(',') if part.strip()]
    best = 0.0
    for part in parts:
        tokens = normalize_text(part).split()
        if not tokens:
            continue
        last = tokens[-1]
        best = max(best, similarity_ratio(surname, last), similarity_ratio(surname, part))
        if surname == last or surname in normalize_text(part):
            return True, 1.0
    return best >= MIN_AUTHOR_SIMILARITY, best


def keyword_coverage(keywords: List[str], title: str) -> Tuple[int, float]:
    if not keywords or not title:
        return 0, 0.0
    title_norm = normalize_text(title)
    matched = sum(1 for keyword in keywords if keyword in title_norm)
    return matched, matched / len(keywords)


def score_candidate(
    candidate: Candidate,
    original_year: Optional[str],
    original_author: Optional[str],
    original_keywords: List[str],
    original_doi: Optional[str],
) -> Tuple[int, List[str], Dict[str, object]]:
    score = 100
    reasons: List[str] = []
    debug: Dict[str, object] = {}
    breakdown = {
        'year_adjustment': 0,
        'author_adjustment': 0,
        'title_adjustment': 0,
        'doi_adjustment': 0,
        'source_adjustment': 0,
    }

    title_similarity = similarity_ratio(" ".join(original_keywords) if original_keywords else "", candidate.title)
    matched_keywords, coverage = keyword_coverage(original_keywords, candidate.title)
    author_ok, author_similarity = candidate_author_match(original_author, candidate.authors)

    debug['title_similarity'] = round(title_similarity, 3)
    debug['keyword_matches'] = matched_keywords
    debug['keyword_total'] = len(original_keywords)
    debug['keyword_coverage'] = round(coverage, 3)
    debug['author_similarity'] = round(author_similarity, 3)

    orig_year_int = safe_year(original_year)
    cand_year_int = safe_year(candidate.year)
    if orig_year_int and cand_year_int:
        diff = abs(orig_year_int - cand_year_int)
        if diff > 1:
            score -= 25
            breakdown['year_adjustment'] -= 25
            reasons.append(f"year mismatch ({original_year} vs {candidate.year})")
        elif diff == 1:
            score -= 10
            breakdown['year_adjustment'] -= 10
            reasons.append(f"minor year difference ({original_year} vs {candidate.year})")

    if original_author and candidate.authors != 'N/A' and not author_ok:
        score -= 25
        breakdown['author_adjustment'] -= 25
        reasons.append(f"first-author mismatch (best similarity {author_similarity:.2f})")

    if original_keywords and candidate.title != 'N/A':
        if coverage < MIN_KEYWORD_COVERAGE and title_similarity < MIN_TITLE_SIMILARITY:
            score -= 40
            breakdown['title_adjustment'] -= 40
            reasons.append(
                f"title mismatch (similarity {title_similarity:.2f}; keyword coverage {matched_keywords}/{len(original_keywords)})"
            )
        elif coverage < MIN_KEYWORD_COVERAGE or title_similarity < MIN_TITLE_SIMILARITY:
            score -= 15
            breakdown['title_adjustment'] -= 15
            reasons.append(
                f"partial title mismatch (similarity {title_similarity:.2f}; keyword coverage {matched_keywords}/{len(original_keywords)})"
            )

    candidate_doi = candidate.doi.lower() if candidate.doi and candidate.doi != 'N/A' else None
    if original_doi and candidate_doi:
        if original_doi != candidate_doi:
            score -= 15
            breakdown['doi_adjustment'] -= 15
            reasons.append(f"DOI mismatch ({original_doi} vs {candidate_doi})")
        else:
            reasons.append("DOI matched")
    elif original_doi and not candidate_doi:
        score -= 5
        breakdown['doi_adjustment'] -= 5
        reasons.append("DOI missing from matched record")

    if candidate.direct_doi_verified:
        reasons.append("DOI resolved directly")

    if candidate.source == 'CR' and candidate.source_score is not None and candidate.source_score < LOW_CROSSREF_SCORE_PENALTY_THRESHOLD:
        score -= 10
        breakdown['source_adjustment'] -= 10
        reasons.append(f"low Crossref score ({candidate.source_score:.2f})")

    debug.update(breakdown)
    debug['final_score'] = max(0, min(100, score))
    return max(0, min(100, score)), reasons, debug


def classify_result(score: int, crossref_score: Optional[float], direct_doi_verified: bool) -> str:
    has_high_confidence = (crossref_score is not None and crossref_score >= HIGH_CONFIDENCE_CROSSREF_SCORE) or direct_doi_verified
    has_medium_confidence = (crossref_score is not None and crossref_score >= MEDIUM_CONFIDENCE_CROSSREF_SCORE) or (
        direct_doi_verified and score >= 75
    )

    if score >= HIGH_CONFIDENCE_SCORE:
        return "VERIFIED - HIGH CONFIDENCE" if has_high_confidence else "VERIFIED"
    if score >= MEDIUM_CONFIDENCE_SCORE:
        return "VERIFIED - MEDIUM CONFIDENCE" if has_medium_confidence else "SUSPICIOUS - Review Recommended"
    if score >= LOW_MATCH_SCORE:
        return "SUSPICIOUS - Likely Mismatch"
    return "UNVERIFIED - Poor Match"


def verify_reference(reference_text: str) -> Tuple[str, Dict[str, object]]:
    original_year = extract_year_heuristic(reference_text)
    original_author = extract_first_author_lastname(reference_text)
    original_keywords = extract_title_keywords(reference_text)
    original_doi = extract_doi_heuristic(reference_text)

    candidates: List[Candidate] = []
    seen: set = set()

    doi_candidate = verify_doi_direct(original_doi)
    if doi_candidate:
        key = (normalize_text(doi_candidate.title), doi_candidate.doi)
        seen.add(key)
        candidates.append(doi_candidate)

    for source_candidates in (
        search_crossref(reference_text),
        search_openalex(reference_text),
        search_arxiv(reference_text),
    ):
        for candidate in source_candidates:
            key = (normalize_text(candidate.title), candidate.doi)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

    if not candidates:
        return "NOT FOUND (All APIs)", {
            'reason': 'No matching record found in Crossref, OpenAlex, arXiv, or direct DOI lookup.',
            'integrity_score': '0',
            'extracted_year': original_year or 'Not extracted',
            'title': 'N/A',
            'authors': 'N/A',
            'container_title': 'N/A',
            'year': 'N/A',
            'doi': original_doi or 'N/A',
            'url': 'N/A',
            'score': 'N/A',
            'matched_source': 'N/A',
            'supporting_sources': 'None',
        }

    scored: List[Tuple[int, Candidate, List[str], Dict[str, object]]] = []
    for candidate in candidates:
        score, reasons, debug = score_candidate(candidate, original_year, original_author, original_keywords, original_doi)
        scored.append((score, candidate, reasons, debug))

    scored.sort(
        key=lambda item: (
            item[0],
            item[1].direct_doi_verified,
            item[1].source == 'CR',
            item[1].source_score if item[1].source_score is not None else -1,
        ),
        reverse=True,
    )
    best_score, best_candidate, reasons, debug = scored[0]

    crossref_score = best_candidate.source_score if best_candidate.source in {'CR', 'CR_DOI'} else None
    supporting_sources = sorted({candidate.source for score, candidate, _, _ in scored if score >= MIN_VALID_SCORE})
    status = classify_result(best_score, crossref_score, best_candidate.direct_doi_verified)

    if best_candidate.source == 'CR' and crossref_score is not None and crossref_score < MIN_CROSSREF_SCORE and status.startswith("VERIFIED"):
        status = status.replace("VERIFIED", "SUSPICIOUS", 1)
        if not any("Low Crossref score" in reason or "low Crossref score" in reason for reason in reasons):
            reasons = [f"Low Crossref score ({crossref_score:.2f})."] + reasons

    details: Dict[str, object] = asdict(best_candidate)
    details.update({
        'score': best_candidate.raw_score_label,
        'matched_source': best_candidate.source,
        'supporting_sources': ", ".join(supporting_sources) if supporting_sources else 'None',
        'integrity_score': str(best_score),
        'reason': '; '.join(reasons) if reasons else 'All checks passed.',
        'extracted_year': original_year or 'Not extracted',
        'extracted_author': original_author or 'Not extracted',
        'extracted_keywords': ", ".join(original_keywords) if original_keywords else 'Not extracted',
        'title_similarity': debug.get('title_similarity', ''),
        'keyword_coverage': debug.get('keyword_coverage', ''),
        'author_similarity': debug.get('author_similarity', ''),
        'year_adjustment': debug.get('year_adjustment', 0),
        'author_adjustment': debug.get('author_adjustment', 0),
        'title_adjustment': debug.get('title_adjustment', 0),
        'doi_adjustment': debug.get('doi_adjustment', 0),
        'source_adjustment': debug.get('source_adjustment', 0),
        'candidate_count': len(candidates),
    })
    return f"{status} ({details['supporting_sources']})", details


def write_to_csv(results: List[Dict[str, object]], filename: str) -> None:
    fieldnames = [
        'Status', 'Original Reference', 'Matched Source', 'Supporting Sources', 'Source Score',
        'Integrity Score (0-100)', 'Extracted Year', 'Database Year', 'Title Similarity',
        'Keyword Coverage', 'Author Similarity', 'Year Adjustment', 'Author Adjustment',
        'Title Adjustment', 'DOI Adjustment', 'Source Adjustment', 'Title', 'Authors',
        'Container Title', 'DOI', 'URL', 'Reason'
    ]
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                'Status': row['status'],
                'Original Reference': row['original_ref'],
                'Matched Source': row.get('matched_source', ''),
                'Supporting Sources': row.get('supporting_sources', ''),
                'Source Score': row.get('score', ''),
                'Integrity Score (0-100)': row.get('integrity_score', ''),
                'Extracted Year': row.get('extracted_year', ''),
                'Database Year': row.get('year', ''),
                'Title Similarity': row.get('title_similarity', ''),
                'Keyword Coverage': row.get('keyword_coverage', ''),
                'Author Similarity': row.get('author_similarity', ''),
                'Year Adjustment': row.get('year_adjustment', 0),
                'Author Adjustment': row.get('author_adjustment', 0),
                'Title Adjustment': row.get('title_adjustment', 0),
                'DOI Adjustment': row.get('doi_adjustment', 0),
                'Source Adjustment': row.get('source_adjustment', 0),
                'Title': row.get('title', ''),
                'Authors': row.get('authors', ''),
                'Container Title': row.get('container_title', ''),
                'DOI': row.get('doi', ''),
                'URL': row.get('url', ''),
                'Reason': row.get('reason', ''),
            })


def write_summary_report(results: List[Dict[str, object]], filename: Optional[str] = None) -> None:
    if filename is None:
        filename = output_base() + SUMMARY_SUFFIX
    total = len(results)
    if total == 0:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("No results to summarize.\n")
        return

    statuses = [base_status(str(r.get('status', ''))) for r in results]
    high_conf = sum(1 for status in statuses if status == 'VERIFIED - HIGH CONFIDENCE')
    med_conf = sum(1 for status in statuses if status == 'VERIFIED - MEDIUM CONFIDENCE')
    verified_basic = sum(1 for status in statuses if status == 'VERIFIED')
    suspicious = sum(1 for status in statuses if status.startswith('SUSPICIOUS'))
    unverified = sum(1 for status in statuses if status == 'UNVERIFIED - Poor Match')
    not_found = sum(1 for status in statuses if status == 'NOT FOUND (All APIs)')
    scores = [int(r['integrity_score']) for r in results if str(r.get('integrity_score', '')).isdigit()]

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('REFERENCE VERIFICATION SUMMARY\n')
        f.write('=' * 80 + '\n\n')
        f.write(f'Total References Analyzed: {total}\n\n')
        f.write('STATUS BREAKDOWN\n')
        f.write('-' * 80 + '\n')
        f.write(f'High Confidence (Verified):   {high_conf:3d} ({high_conf / total * 100:5.1f}%)\n')
        f.write(f'Medium Confidence (Verified): {med_conf:3d} ({med_conf / total * 100:5.1f}%)\n')
        f.write(f'Verified (Basic):             {verified_basic:3d} ({verified_basic / total * 100:5.1f}%)\n')
        f.write(f'Suspicious:                   {suspicious:3d} ({suspicious / total * 100:5.1f}%)\n')
        f.write(f'Unverified:                   {unverified:3d} ({unverified / total * 100:5.1f}%)\n')
        f.write(f'Not Found:                    {not_found:3d} ({not_found / total * 100:5.1f}%)\n\n')

        total_verified = high_conf + med_conf + verified_basic
        total_problematic = suspicious + unverified + not_found
        f.write('ANALYSIS GROUPS\n')
        f.write('-' * 80 + '\n')
        f.write(f'Valid:                        {total_verified:3d} ({total_verified / total * 100:5.1f}%)\n')
        f.write(f'Problematic:                  {total_problematic:3d} ({total_problematic / total * 100:5.1f}%)\n\n')

        if scores:
            f.write('SCORE SUMMARY\n')
            f.write('-' * 80 + '\n')
            f.write(f'Average Integrity Score: {sum(scores) / len(scores):.1f}/100\n')
            f.write(f'Min: {min(scores)}/100\n')
            f.write(f'Max: {max(scores)}/100\n\n')

        review_rows = [r for r in results if not base_status(str(r.get('status', ''))).startswith('VERIFIED')]
        f.write('REFERENCES TO REVIEW\n')
        f.write('-' * 80 + '\n')
        if not review_rows:
            f.write('None.\n')
        else:
            for idx, row in enumerate(review_rows, start=1):
                f.write(f"{idx}. [{row['status']}]\n")
                f.write(f"   Reference: {row['original_ref']}\n")
                f.write(f"   Score: {row.get('integrity_score', 'N/A')}/100\n")
                f.write(f"   Reason: {row.get('reason', 'N/A')}\n\n")

def get_references() -> List[str]:
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
        try:
            with open(input_filename, 'r', encoding='utf-8') as f:
                input_source = f.read()
            print(f"Reading references from file: {input_filename}")
        except FileNotFoundError:
            print(f"Error: File '{input_filename}' not found.")
            sys.exit(1)
    else:
        print("Reading references from console input (Ctrl+D/Ctrl+Z then Enter to finish)...")
        input_source = sys.stdin.read()

    if not input_source:
        return []

    clean_input = input_source.strip()
    if re.search(r'\[\s*\d+\s*\]', clean_input[:500]):
        reference_parts = re.split(r'(\[\s*\d+\s*\])', clean_input)
        references: List[str] = []
        for i in range(1, len(reference_parts) - 1, 2):
            full_ref = (reference_parts[i] + reference_parts[i + 1]).strip()
            if full_ref:
                references.append(full_ref)
        if references:
            print("Detected bracketed format, processing multi-line references.")
            return references

    return [ref.strip() for ref in clean_input.splitlines() if ref.strip()]


def generate_summary_report(results: List[Dict[str, object]]) -> None:
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return

    statuses = [base_status(str(r.get('status', ''))) for r in results]
    buckets = {
        'VERIFIED - HIGH CONFIDENCE': sum(1 for status in statuses if status == 'VERIFIED - HIGH CONFIDENCE'),
        'VERIFIED - MEDIUM CONFIDENCE': sum(1 for status in statuses if status == 'VERIFIED - MEDIUM CONFIDENCE'),
        'VERIFIED': sum(1 for status in statuses if status == 'VERIFIED'),
        'SUSPICIOUS': sum(1 for status in statuses if status.startswith('SUSPICIOUS')),
        'UNVERIFIED': sum(1 for status in statuses if status == 'UNVERIFIED - Poor Match'),
        'NOT FOUND': sum(1 for status in statuses if status == 'NOT FOUND (All APIs)'),
    }
    scores = [int(r['integrity_score']) for r in results if str(r.get('integrity_score', '')).isdigit()]

    print("\n📊 OVERALL STATISTICS")
    print('─' * 80)
    print(f"Total References Processed: {total}")
    for label, count in buckets.items():
        print(f"{label:34} {count:3d} ({count / total * 100:5.1f}%)")
    if scores:
        print(f"Average Integrity Score: {sum(scores) / len(scores):.1f}/100")
        print(f"Min / Max Score: {min(scores)}/100 - {max(scores)}/100")
    total_verified = buckets['VERIFIED - HIGH CONFIDENCE'] + buckets['VERIFIED - MEDIUM CONFIDENCE'] + buckets['VERIFIED']
    total_problematic = buckets['SUSPICIOUS'] + buckets['UNVERIFIED'] + buckets['NOT FOUND']
    print(f"{'VALID':34} {total_verified:3d} ({total_verified / total * 100:5.1f}%)")
    print(f"{'PROBLEMATIC':34} {total_problematic:3d} ({total_problematic / total * 100:5.1f}%)")
    print('─' * 80)

def main() -> None:
    print("\n--- Unified Academic Reference Verifier (Crossref + OpenAlex + arXiv) ---")
    references = get_references()
    if not references:
        print("No references provided. Exiting.")
        return

    print(f"\nFound {len(references)} references. Starting verification process...\n")
    all_results: List[Dict[str, object]] = []

    for index, reference in enumerate(references, start=1):
        print("\n" + "=" * 80)
        print(f"[{index}/{len(references)}] VERIFYING REFERENCE")
        print("=" * 80)
        print(f"Reference: {reference[:140]}{'...' if len(reference) > 140 else ''}")

        status, details = verify_reference(reference)
        integrity_score = details.get('integrity_score', '0')
        print(f"  STATUS: {status}")
        print(f"  INTEGRITY SCORE: {integrity_score}/100")
        print(f"  MATCHED SOURCE: {details.get('matched_source', 'N/A')}")
        print(f"  SUPPORTING SOURCES: {details.get('supporting_sources', 'None')}")
        print(f"  TITLE: {details.get('title', 'N/A')}")
        print(f"  AUTHORS: {details.get('authors', 'N/A')}")
        print(f"  YEAR: {details.get('year', 'N/A')} | EXTRACTED YEAR: {details.get('extracted_year', 'N/A')}")
        print(f"  REASON: {details.get('reason', 'N/A')}")

        result_row = {'status': status, 'original_ref': reference}
        result_row.update(details)
        all_results.append(result_row)
        time.sleep(0.2)

    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY REPORT")
    print("=" * 80)
    generate_summary_report(all_results)

    base = output_base()
    csv_path = base + CSV_SUFFIX
    summary_path = base + SUMMARY_SUFFIX
    write_to_csv(all_results, csv_path)
    write_summary_report(all_results, summary_path)
    print(f"Detailed CSV saved to: {csv_path}")
    print(f"Summary report saved to: {summary_path}")


if __name__ == "__main__":
    main()