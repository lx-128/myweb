"""Small helper to compute final CSVs and normalize schema."""
import pandas as pd
from pathlib import Path


def standardize_schema(input_csv, out_csv='data/standard_jobs.csv'):
    df = pd.read_csv(input_csv, encoding='utf-8')
    # ensure columns exist
    cols = ['source','job_title','company','salary','city','district','experience','education','job_type','posted_date','job_url','description','crawl_timestamp']
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    return out_csv


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('usage: python compute_metrics.py raw.csv')
    else:
        print('writing standardized CSV to data/standard_jobs.csv')
        standardize_schema(sys.argv[1])
