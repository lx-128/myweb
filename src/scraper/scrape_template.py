"""Playwright-based scraper template (boss / other sites)

Usage:
  python3 scrape_template.py --keyword "Python" --city "北京" --target 100000 --out data/raw_jobs.csv

Notes:
- Selectors are placeholders and must be adjusted to the target site's HTML structure.
- For large-scale scraping (100k rows) use proxies, randomized delays and persistent storage.
- This script writes incrementally to CSV to avoid large memory usage.
"""
import argparse
import csv
import time
import math
from datetime import datetime
from urllib.parse import quote
from pathlib import Path
from tqdm import tqdm
from playwright.sync_api import sync_playwright

DEFAULT_HEADERS = [
    "source",
    "job_title",
    "company",
    "salary",
    "city",
    "district",
    "experience",
    "education",
    "job_type",
    "posted_date",
    "job_url",
    "description",
    "crawl_timestamp"
]


def init_csv(path: Path, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def parse_salary(s):
    # Keep raw for now. ETL will try to normalize ranges like "15k-30k/月"
    return s.strip() if s else ""


def build_search_url(keyword, city, page):
    # Example for zhipin (may not be exact). Replace with the correct search URL pattern.
    q = quote(keyword)
    city_q = quote(city) if city else ""
    # Placeholder URL -- open the site and copy the exact URL structure for production
    return f"https://www.zhipin.com/web/geek/job?query={q}&city={city_q}&page={page}"


def extract_from_card(card):
    """Extract fields from a job card element. Adjust selectors per site."""
    try:
        title_el = card.query_selector("h3, .job-title, .job-name")
        job_title = title_el.inner_text().strip() if title_el else ""

        company_el = card.query_selector(".company-text, .company-name, .company-info a")
        company = company_el.inner_text().strip() if company_el else ""

        salary_el = card.query_selector(".red, .salary, .job-salary")
        salary = parse_salary(salary_el.inner_text()) if salary_el else ""

        meta_el = card.query_selector(".job-meta, .info-primary p, .job-limit")
        meta_text = meta_el.inner_text().strip() if meta_el else ""
        city = ""
        district = ""
        experience = ""
        education = ""
        posted_date = ""
        # best-effort parse
        parts = [p.strip() for p in meta_text.split("\n") if p.strip()]
        if parts:
            if len(parts) >= 1:
                city = parts[0]
            if len(parts) >= 2:
                experience = parts[1]
            if len(parts) >= 3:
                education = parts[2]

        link_el = card.query_selector("a")
        href = link_el.get_attribute("href") if link_el else ""
        job_url = href if href.startswith("http") else ("https://www.zhipin.com" + href if href else "")

        description = ""  # optional: open job_url and extract description

        return {
            "source": "zhipin",
            "job_title": job_title,
            "company": company,
            "salary": salary,
            "city": city,
            "district": district,
            "experience": experience,
            "education": education,
            "job_type": "",
            "posted_date": posted_date,
            "job_url": job_url,
            "description": description,
            "crawl_timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print("card extract error:", e)
        return None


def scrape(keyword, city, target_count, out_path: Path, headless=True):
    init_csv(out_path, DEFAULT_HEADERS)
    page_number = 1
    collected = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()

        pbar = tqdm(total=target_count)
        while collected < target_count:
            url = build_search_url(keyword, city, page_number)
            try:
                page.goto(url, timeout=30000)
                time.sleep(1.2)
                # remove/handle popups here if needed
                # find job cards -- selector must be adapted
                cards = page.query_selector_all(".job-card, .job-primary, li.job-item")
                if not cards:
                    print("No cards found on page", page_number)
                    break

                rows = []
                for c in cards:
                    rec = extract_from_card(c)
                    if rec:
                        rows.append(rec)

                if not rows:
                    break

                with out_path.open("a", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=DEFAULT_HEADERS)
                    for r in rows:
                        writer.writerow(r)
                        collected += 1
                        pbar.update(1)
                        if collected >= target_count:
                            break

                page_number += 1
                # polite delay
                time.sleep(0.8 + (0.6 * (page_number % 3)))
            except Exception as e:
                print("page error:", e)
                time.sleep(5)
                page_number += 1
                continue
        pbar.close()
        browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', type=str, default='Python')
    parser.add_argument('--city', type=str, default='北京')
    parser.add_argument('--target', type=int, default=1000)
    parser.add_argument('--out', type=str, default='data/raw_jobs.csv')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    scrape(args.keyword, args.city, args.target, Path(args.out), headless=args.headless)
