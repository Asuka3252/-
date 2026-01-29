import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import scipy.stats as stats
import json
import numpy as np

# --- 1. 全局配置 ---
st.set_page_config(page_title="传染病疫情智能研判系统 (Executive)", layout="wide", page_icon="📝")

# 绘图风格：学术论文风
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300 

# --- 2. 核心组件：法定传染病分类 ---
DISEASE_CLASS = {
    '甲类': ['鼠疫', '霍乱'],
    '乙类': ['传染性非典型肺炎', '艾滋病', '病毒性肝炎', '脊髓灰质炎', '人感染高致病性禽流感', '麻疹', '流行性出血热', '狂犬病', '流行性乙型脑炎', '登革热', '炭疽', '细菌性痢疾', '阿米巴性痢疾', '肺结核', '伤寒', '副伤寒', '流行性脑脊髓膜炎', '百日咳', '白喉', '新生儿破伤风', '猩红热', '布鲁氏菌病', '淋病', '梅毒', '钩端螺旋体病', '血吸虫病', '疟疾', '新型冠状病毒感染', '新冠病毒感染', '人感染H7N9禽流感', '猴痘'],
    '丙类': ['流行性感冒', '流行性腮腺炎', '风疹', '急性出血性结膜炎', '麻风病', '流行性斑疹伤寒', '地方性斑疹伤寒', '黑热病', '包虫病', '丝虫病', '除霍乱、细菌性痢疾、伤寒和副伤寒以外的感染性腹泻病', '手足口病', '其它感染性腹泻病', '其他感染性腹泻病']
}

def get_disease_class(name):
    name = str(name).replace(' ', '').strip()
    if name in DISEASE_CLASS['甲类']: return '甲类'
    if name in DISEASE_CLASS['乙类'] or any(x in name for x in ['肝炎', '梅毒', '炭疽', '艾滋']): return '乙类'
    if name in DISEASE_CLASS['丙类'] or any(x in name for x in ['腹泻', '斑疹']): return '丙类'
    return '其他'

# --- 3. 统计学引擎 ---
def format_p_value(p):
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"

def generate_three_line_table_html(df, title=""):
    """生成标准三线表 (HTML)"""
    html = f"""
    <style>
        .three-line-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-family: 'Times New Roman', 'SimSun', serif; font-size: 14px; text-align: center; }}
        .three-line-table thead th {{ border-top: 2px solid #000; border-bottom: 1px solid #000; padding: 8px; font-weight: bold; }}
        .three-line-table tbody td {{ padding: 6px; border: none; }}
        .three-line-table tbody tr:last-child td {{ border-bottom: 2px solid #000; }}
        .caption {{ font-weight: bold; margin-bottom: 5px; text-align: center; }}
    </style>
    <div class="caption">{title}</div>
    <table class="three-line-table">
        <thead><tr>{''.join(f'<th>{col}</th>' for col in df.columns)}</tr></thead>
        <tbody>{''.join('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in df.values)}</tbody>
    </table>
    """
    return html

def process_stats_table(df, row_name_col):
    """处理统计表：添加构成比、卡方值、P值"""
    try:
        stats_df = df.copy()
        cols = stats_df.columns
        val_cols = cols[1:] 
        for c in val_cols: stats_df[c] = pd.to_numeric(stats_df[c], errors='coerce').fillna(0)
        stats_df['合计'] = stats_df[val_cols].sum(axis=1)
        total_sum = stats_df['合计'].sum()
        stats_df['构成比(%)'] = (stats_df['合计'] / total_sum * 100).round(2)
        
        if len(val_cols) >= 2:
            obs = stats_df[val_cols].values
            obs_clean = obs[~np.all(obs == 0, axis=1)]
            if obs_clean.sum() > 0 and obs_clean.shape[0] > 1:
                chi2, p, dof, ex = stats.chi2_contingency(obs_clean)
                stats_df['χ²值'] = ''
                stats_df['P值'] = ''
                stats_df.iloc[0, stats_df.columns.get_loc('χ²值')] = f"{chi2:.2f}"
                stats_df.iloc[0, stats_df.columns.get_loc('P值')] = format_p_value(p)
        return stats_df
    except: return df

# --- 4. 研判报告生成引擎 (Executive Generator) ---
class ReportGenerator:
    def __init__(self, data_map):
        self.d = data_map

    def fmt_trend(self, val):
        """格式化涨跌幅: 上升XX% / 下降XX%"""
        try:
            v = float(val)
            if v > 0: return f"上升{v:.2f}%"
            elif v < 0: return f"下降{abs(v):.2f}%"
            return "持平"
        except: return "持平"

    def get_top_diseases_text(self, df_sub, total_cases):
        """生成 Top N 病种描述文本"""
        if df_sub.empty: return "无报告病例。"
        
        # 排序
        df_sub = df_sub.sort_values('本期发病数', ascending=False)
        top_list = []
        
        # 遍历前几位 (默认前3，如果少于3则全部)
        for idx, row in df_sub.head(3).iterrows():
            if row['本期发病数'] <= 0: continue
            
            name = row.iloc[0] # 病种名
            cases = int(row['本期发病数'])
            percent = (cases / total_cases * 100)
            
            # 获取环比同比 (假设列名固定，需增强鲁棒性)
            mom = row.get('与上期比（%）', 0)
            yoy = row.get('与去年同期比（%）', 0)
            
            # 格式：病名（病例数，占比，与上月比...，与去年同期比...）
            desc = f"{name}（{cases}例，占比{percent:.2f}%，较上月{self.fmt_trend(mom)}，较去年同期{self.fmt_trend(yoy)}）"
            top_list.append(desc)
            
        return "、".join(top_list) if top_list else "无报告病例。"

    def generate_full_report(self):
        if self.d['summary'] is None: return "⚠️ 缺失疫情分析报表，无法生成概况。"
        
        df = self.d['summary'].copy()
        
        # 1. 总体概况
        # 尝试提取合计行
        total_row = df[df.iloc[:,0].astype(str).str.contains('合计')].iloc[0]
        total_cases = int(total_row['本期发病数'])
        total_mom = total_row.get('与上期比（%）', 0)
        total_yoy = total_row.get('与去年同期比（%）', 0)
        
        # 统计有病例的病种数
        df_detail = df[~df.iloc[:,0].astype(str).str.contains('合计')].copy()
        reported_count = len(df_detail[df_detail['本期发病数'] > 0])
        
        section_1 = f"""
### 一、 近期概况
**(一) 传染病报告信息管理系统**
1. **传染病疫情**：本月我区共报告法定传染病 **{reported_count}** 种 **{total_cases}** 例。
   与上月相比{self.fmt_trend(total_mom)}；与去年同期相比{self.fmt_trend(total_yoy)}。
        """

        # 2. 乙类分析
        df_detail['Class'] = df_detail.iloc[:,0].apply(get_disease_class)
        df_b = df_detail[df_detail['Class'] == '乙类']
        
        if not df_b.empty:
            b_cases = df_b['本期发病数'].sum()
            b_count = len(df_b[df_b['本期发病数'] > 0])
            b_text = self.get_top_diseases_text(df_b, b_cases) if b_cases > 0 else "无"
            
            section_2 = f"""
2. **乙类传染病**：本月报告 **{b_count}** 种，合计 **{int(b_cases)}** 例。
   发病数居前几位的病种为：**{b_text}**。
            """
        else: section_2 = "\n2. **乙类传染病**：无报告。\n"

        # 3. 丙类分析
        df_c = df_detail[df_detail['Class'] == '丙类']
        
        if not df_c.empty:
            c_cases = df_c['本期发病数'].sum()
            c_count = len(df_c[df_c['本期发病数'] > 0])
            c_text = self.get_top_diseases_text(df_c, c_cases) if c_cases > 0 else "无"
            
            section_3 = f"""
3. **丙类传染病**：本月报告 **{c_count}** 种，合计 **{int(c_cases)}** 例。
   主要流行病种为：**{c_text}**。
            """
        else: section_3 = "\n3. **丙类传染病**：无报告。\n"

        return section_1 + section_2 + section_3

# --- 5. 数据解析 (Advanced Parser) ---
class AdvancedParser:
    def __init__(self):
        self.data = {'summary': None, 'time': None, 'age': None, 'pop': None, 'area': None}
        self.geojson = None

    def bin_ages(self, df):
        """5岁年龄组分箱"""
        age_col = next((c for c in df.columns if '年龄' in str(c) and '组' not in str(c)), None)
        if age_col:
            try:
                df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
                bins = range(0, 101, 5)
                labels = [f"{i}-{i+4}" for i in range(0, 96, 5)] + ["100+"]
                labels = labels[:len(bins)-1]
                df['年龄组'] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
                if any('性' in c for c in df.columns):
                    sex_col = next(c for c in df.columns if '性' in c)
                    grouped = df.groupby(['年龄组', sex_col]).size().unstack(fill_value=0)
                    grouped.reset_index(inplace=True)
                    return grouped
                else:
                    grouped = df['年龄组'].value_counts().sort_index().reset_index()
                    grouped.columns = ['年龄组', '发病数']
                    return grouped
            except: pass
        return df

    def parse_files(self, files):
        logs = []
        for f in files:
            try:
                fname = f.name
                if fname.endswith('.json') or fname.endswith('.geojson'):
                    self.geojson = gpd.GeoDataFrame.from_features(json.load(f)["features"])
                    logs.append(f"🗺️ 地图: {fname}")
                    continue

                if fname.endswith('.csv'):
                    try: df = pd.read_csv(f, header=0, encoding='utf-8')
                    except: df = pd.read_csv(f, header=0, encoding='gbk')
                else: df = pd.read_excel(f)
                
                cols = "".join(df.columns.astype(str))
                
                if '报表' in fname or ('病种' in cols and '本期' in cols):
                    # 关键清洗：确保数值列为数字
                    for c in df.columns: 
                        if any(k in str(c) for k in ['数', '比']):
                            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.replace('-', '0'), errors='coerce').fillna(0)
                    self.data['summary'] = df
                    logs.append(f"✅ 汇总报表: {fname}")
                elif '时间' in fname or ('时间' in cols and '发病' in cols):
                    self.data['time'] = df
                    logs.append(f"✅ 时间分布: {fname}")
                elif '年龄' in fname or '男' in cols:
                    df = self.bin_ages(df)
                    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
                    self.data['age'] = df
                    logs.append(f"✅ 年龄分布: {fname}")
                elif '人群' in fname or '职业' in cols:
                    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
                    self.data['pop'] = df
                    logs.append(f"✅ 人群分布: {fname}")
                elif '地区' in fname or '乡镇' in cols or '街道' in cols:
                    if df.shape[1] >= 2:
                        df = df.iloc[:, [0, 1]] 
                        df.columns = ['Name', 'Cases']
                        df['Cases'] = pd.to_numeric(df['Cases'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
                        self.data['area'] = df
                        logs.append(f"✅ 地区分布: {fname}")
            except Exception as e: logs.append(f"❌ 解析异常 {f.name}: {e}")
        return logs

# --- 6. 绘图引擎 ---
def plot_geo_heatmap(df_area, gdf_map):
    """GIS热力图 (修复 Length Mismatch)"""
    try:
        data = df_area[~df_area['Name'].str.contains('合计|总计')].copy()
        gdf = gdf_map.copy()
        name_col = next((c for c in gdf.columns if c.lower() in ['name', 'town']), gdf.select_dtypes(include=['object']).columns[0])
        gdf[name_col] = gdf[name_col].astype(str).str.strip()
        data['Name'] = data['Name'].astype(str).str.strip()
        merged = gdf.merge(data, left_on=name_col, right_on='Name', how='left').fillna(0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        merged.plot(column='Cases', cmap='Blues', linewidth=0.8, edgecolor='0.6', legend=True, legend_kwds={'shrink': 0.6}, ax=ax)
        for idx, row in merged.iterrows():
            if row.geometry and row['Cases'] > 0:
                ax.text(row.geometry.centroid.x, row.geometry.centroid.y, f"{row[name_col]}\n{int(row['Cases'])}", fontsize=8, ha='center', color='black')
        ax.axis('off')
        return fig
    except: return None

# --- 7. 主程序 ---
def main():
    st.title("🛡️ 传染病疫情智能研判系统 (Executive)")
    with st.sidebar:
        st.header("📂 数据中心")
        st.info("💡 请按住 Ctrl 批量上传：\n1. 疫情分析报表.xlsx\n2. 三间分布表\n3. yixiu.json")
        files = st.file_uploader("文件上传", accept_multiple_files=True)
        parser = AdvancedParser()
        if files:
            logs = parser.parse_files(files)
            for l in logs: st.caption(l)

    tab1, tab2, tab3 = st.tabs(["📄 智能研判报告", "📊 统计附表", "🗺️ 可视化图表"])
    
    with tab1:
        if parser.data['summary'] is not None:
            gen = ReportGenerator(parser.data)
            report = gen.generate_full_report()
            st.markdown(report)
            st.download_button("📥 导出报告文本", report, "report.txt")
        else: st.info("请上传[疫情分析报表]以生成概况。")

    with tab2:
        st.subheader("流行病学特征统计表")
        if parser.data['age'] is not None:
            st.markdown("**表1 不同年龄组发病情况及性别分布**")
            html = generate_three_line_table_html(process_stats_table(parser.data['age'], '年龄'))
            st.markdown(html, unsafe_allow_html=True)
        if parser.data['pop'] is not None:
            st.markdown("**表2 重点职业人群发病情况**")
            html = generate_three_line_table_html(process_stats_table(parser.data['pop'], '人群'))
            st.markdown(html, unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 图1：地区分布热力图")
            if parser.data['area'] is not None and parser.geojson is not None:
                fig = plot_geo_heatmap(parser.data['area'], parser.geojson)
                if fig: st.pyplot(fig)
        with col2:
            st.markdown("#### 图2：时间分布趋势")
            if parser.data['time'] is not None:
                df = parser.data['time']
                df = df[~df.iloc[:,0].astype(str).str.contains('合计')]
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(df.iloc[:,0].astype(str), df.iloc[:,1], marker='o')
                plt.xticks(rotation=45)
                st.pyplot(fig)

if __name__ == "__main__":
    main()