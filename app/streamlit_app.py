"""Streamlit dashboard for monthly IT job metrics.
Run:
  streamlit run app/streamlit_app.py --server.port 8501

The app expects a CSV at data/monthly_metrics.csv (created by ETL) and optionally data/standard_jobs.csv for detailed views.
"""
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title='IT Jobs Dashboard', layout='wide')

st.title('互联网 IT 岗位数据仪表盘')

DATA_DIR = Path('data')

uploaded = st.file_uploader('上传 monthly_metrics CSV（可选）', type=['csv'])
if uploaded:
    metrics = pd.read_csv(uploaded)
else:
    if (DATA_DIR / 'monthly_metrics.csv').exists():
        metrics = pd.read_csv(DATA_DIR / 'monthly_metrics.csv')
    else:
        st.warning('未找到 data/monthly_metrics.csv，请先运行 ETL 并生成该文件或上传 CSV。')
        st.stop()

# basic cleaning
metrics['date'] = pd.to_datetime(metrics['date'], format='%Y-%m')
metrics = metrics.sort_values('date')

st.sidebar.header('筛选')
start = st.sidebar.date_input('开始日期', metrics['date'].min())
end = st.sidebar.date_input('结束日期', metrics['date'].max())
mask = (metrics['date'] >= pd.to_datetime(start)) & (metrics['date'] <= pd.to_datetime(end))
filtered = metrics[mask]

col1, col2 = st.columns(2)
with col1:
    st.metric('当月岗位总量（最近）', int(filtered['job_total'].iloc[-1]) if not filtered.empty else 0)
    st.metric('月均薪资（最近）', f"{filtered['salary_avg'].iloc[-1]:.0f} 元" if not filtered.empty and pd.notna(filtered['salary_avg'].iloc[-1]) else 'N/A')
with col2:
    if 'supply_demand_ratio' in filtered.columns:
        val = filtered['supply_demand_ratio'].iloc[-1]
        st.metric('供需比（最近）', f"{val:.3f}" if pd.notna(val) else 'N/A')

st.markdown('---')

# time series
chart_df = filtered[['date','job_total','salary_avg']].melt(id_vars='date', value_vars=['job_total','salary_avg'], var_name='metric', value_name='value')
line = alt.Chart(chart_df).mark_line(point=True).encode(x='date:T', y='value:Q', color='metric:N')
st.altair_chart(line.interactive(), use_container_width=True)

st.markdown('### 岗位类别 TOP')
# job_type breakdown column is a dict string in ETL; try to parse
if 'job_type_breakdown' in metrics.columns:
    # expand the latest month
    latest = metrics.iloc[-1]['job_type_breakdown']
    try:
        import ast
        d = ast.literal_eval(latest) if isinstance(latest, str) else (latest if isinstance(latest, dict) else {})
        df_types = pd.DataFrame(list(d.items()), columns=['job_type','count']).sort_values('count', ascending=False)
        st.bar_chart(df_types.set_index('job_type'))
    except Exception:
        st.write('无法解析 job_type_breakdown 字段')

st.markdown('### 原始样本数据（可下载）')
if (DATA_DIR / 'standard_jobs.csv').exists():
    sj = pd.read_csv(DATA_DIR / 'standard_jobs.csv')
    st.dataframe(sj.head(200))
    csv = sj.to_csv(index=False, encoding='utf-8-sig')
    st.download_button('下载标准化原始数据 CSV', csv, file_name='standard_jobs.csv', mime='text/csv')
else:
    st.info('data/standard_jobs.csv 不存在；运行标准化脚本以生成。')
