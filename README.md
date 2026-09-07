# Jobs data pipeline and Streamlit dashboard

This project collects, processes and visualizes IT job data (互联网岗位) and produces monthly metrics per your spec.

Key user choices applied in this branch:
- Priority site: Boss 直聘 (zhipin)
- Target volume: 100,000 rows (per site/overall as configured)
- Output fields / monthly metrics required:
  - date (YYYY-MM)
  - job_total (当月互联网岗位总量)
  - salary_avg (月均薪资, RMB/month)
  - job_type (岗位类别)
  - supply_demand_ratio (供需比 = 岗位数 / 求职人数) — requires job-seeker counts input (see ETL join)
- Output format: CSV
- Visualization: Streamlit interactive dashboard

What I committed on branch feat/jobs-data-system:
- Project scaffold and templates:
  - src/scraper/scrape_template.py  (Playwright-based starter scraper)
  - src/etl/transform.py           (aggregation and metric computation)
  - app/streamlit_app.py           (Streamlit dashboard)
  - src/analysis/compute_metrics.py (helper script)
  - requirements.txt, Dockerfile, .gitignore, README.md

Next steps (I can continue after you review):
1. Iterate and harden the Boss/51Job scraper selectors and anti-bot handling.
2. Run an initial crawl to produce sample data (you must run the scraper locally or provide credentials / proxy infra).
3. Tune ETL (salary parsing, job type taxonomy) and deploy dashboard.

Important: scrapers may trigger anti-bot protections. Please follow site Terms of Service and use proxies / rate limits for large-scale collection.
