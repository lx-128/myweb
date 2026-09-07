"""ETL transform helpers: aggregate raw job rows into monthly metrics per user spec.

Input: raw CSV(s) with columns including: posted_date (free text), salary (raw), job_type
Output: monthly CSV with columns: date, job_total, salary_avg, job_type (category counts as separate file or wide format), supply_demand_ratio

Note: supply_demand_ratio requires a complementary file with job-seeker counts per month (applicants.csv)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re
from dateutil import parser


def parse_posted_month(s):
    if pd.isna(s) or not str(s).strip():
        return None
    try:
        # try to parse a date and return YYYY-MM
        dt = parser.parse(s, fuzzy=True)
        return dt.strftime('%Y-%m')
    except Exception:
        # common patterns like '2024-08' or '2024年08月'
        m = re.search(r'(20\d{2})[\-年/](0?[1-9]|1[0-2])', str(s))
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}"
    return None


def salary_to_monthly_number(s):
    """Convert salary string like '15k-30k/月' or '20k/月' to midpoint monthly RMB.
    Return np.nan if cannot parse."""
    if pd.isna(s) or not str(s).strip():
        return np.nan
    text = str(s).lower().replace('\u00a0', ' ')
    # remove '/月' '/年'
    per_year = False
    if '/年' in text or '年' in text:
        per_year = True
    nums = re.findall(r'(\d+\.?\d*)k', text)
    if nums:
        nums = [float(n) * 1000 for n in nums]
    else:
        nums = re.findall(r'(\d+\.?\d*)', text)
        nums = [float(n) for n in nums]
    if not nums:
        return np.nan
    if len(nums) == 1:
        val = nums[0]
    else:
        val = sum(nums) / len(nums)
    if per_year:
        val = val / 12.0
    return val


def aggregate_monthly(input_csv: str, applicants_csv: str = None, out_csv: str = 'data/monthly_metrics.csv'):
    df = pd.read_csv(input_csv, encoding='utf-8')
    # normalize posted_date -> YYYY-MM
    df['posted_month'] = df.get('posted_date').apply(parse_posted_month) if 'posted_date' in df.columns else None
    # fallback: use crawl_timestamp
    if df['posted_month'].isnull().all():
        if 'crawl_timestamp' in df.columns:
            df['posted_month'] = pd.to_datetime(df['crawl_timestamp']).dt.strftime('%Y-%m')
        else:
            df['posted_month'] = pd.Timestamp.utcnow().strftime('%Y-%m')

    # salary numeric
    df['salary_num'] = df.get('salary').apply(salary_to_monthly_number) if 'salary' in df.columns else np.nan

    # compute job_type: if job_type exists keep, else try to infer from job_title
    if 'job_type' not in df.columns:
        df['job_type'] = df.get('job_title', '').fillna('').apply(lambda t: infer_job_type(t))

    # aggregate
    grouped = df.groupby('posted_month')
    rows = []
    for month, g in grouped:
        job_total = len(g)
        salary_avg = float(g['salary_num'].dropna().mean()) if not g['salary_num'].dropna().empty else np.nan
        # top job types
        job_type_counts = g['job_type'].value_counts().to_dict()
        rows.append({
            'date': month,
            'job_total': job_total,
            'salary_avg': round(salary_avg, 2) if not np.isnan(salary_avg) else None,
            'job_type_breakdown': job_type_counts
        })

    metrics = pd.DataFrame(rows).sort_values('date')

    # supply/demand: join applicants if provided
    if applicants_csv and Path(applicants_csv).exists():
        app = pd.read_csv(applicants_csv)
        # applicants CSV expected columns: date(YYYY-MM), applicants
        metrics = metrics.merge(app, left_on='date', right_on='date', how='left')
        metrics['supply_demand_ratio'] = metrics['job_total'] / metrics['applicants']
    else:
        metrics['supply_demand_ratio'] = None

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_csv, index=False, encoding='utf-8-sig')
    return metrics


def infer_job_type(title: str):
    t = str(title).lower()
    mapping = {
        'backend': ['后端', 'backend', 'java', 'python', 'go', 'golang', 'node'],
        'frontend': ['前端', 'frontend', 'react', 'vue', 'angular'],
        'devops': ['运维', 'devops', '云', 'docker', 'k8s', 'kubernetes'],
        'data': ['数据', 'data', 'ml', '机器学习', '算法'],
        'mobile': ['android', 'ios', '移动', 'flutter'],
        'qa': ['测试', 'qa', 'quality']
    }
    for cat, keys in mapping.items():
        for k in keys:
            if k in t:
                return cat
    return 'other'


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('usage: python transform.py data/raw_jobs.csv [applicants.csv]')
        sys.exit(1)
    input_csv = sys.argv[1]
    applicants = sys.argv[2] if len(sys.argv) > 2 else None
    metrics = aggregate_monthly(input_csv, applicants, out_csv='data/monthly_metrics.csv')
    print(metrics.head())
