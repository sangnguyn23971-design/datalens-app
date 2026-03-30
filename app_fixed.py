import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataLens — AI Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #131619; border-right: 1px solid #2e3740; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.metric-card { background:#1c2028; border:1px solid #2e3740; border-radius:10px; padding:16px 20px; margin-bottom:8px; }
.metric-label { font-size:11px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:#8a9ab0; margin-bottom:4px; }
.metric-value { font-size:26px; font-weight:700; color:#e2e8f0; font-variant-numeric:tabular-nums; }
.metric-delta-up { color:#34d399; font-size:12px; font-weight:500; }
.metric-delta-down { color:#f87171; font-size:12px; font-weight:500; }
.insight-box { background:#1d2228; border-left:3px solid #2dd4bf; border-radius:8px; padding:14px 16px; margin-bottom:8px; }
.insight-title { font-size:12px; font-weight:700; color:#2dd4bf; margin-bottom:4px; }
.insight-text { font-size:12px; color:#8a9ab0; line-height:1.6; }
.scp-box { background:#1d2228; border-radius:10px; padding:16px; }
.scp-title { font-size:11px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:#2dd4bf; margin-bottom:6px; }
.scp-text { font-size:12px; color:#8a9ab0; line-height:1.6; }
.chat-user { background:#2dd4bf22; border:1px solid #2dd4bf44; border-radius:10px; padding:10px 14px; margin:6px 0; font-size:13px; }
.chat-ai { background:#1d2228; border:1px solid #2e3740; border-radius:10px; padding:10px 14px; margin:6px 0; font-size:13px; color:#e2e8f0; }
div[data-testid="stHorizontalBlock"] { gap: 12px; }
</style>
""", unsafe_allow_html=True)

TEAL = "#2dd4bf"
COLORS = ["#2dd4bf","#60a5fa","#f59e0b","#34d399","#f87171","#a78bfa","#fb923c","#e879f9","#4ade80","#facc15"]
BG = "#0d0f12"
SURFACE = "#1c2028"
GRID = "rgba(255,255,255,0.05)"
TEXT_COLOR = "#8a9ab0"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family="Inter, sans-serif", size=11, color=TEXT_COLOR),
    margin=dict(l=12, r=12, t=36, b=12),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=10)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=10)),
    colorway=COLORS,
)
LEGEND_DEFAULT = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", font=dict(size=10))
LEGEND_TOP = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", font=dict(size=10),
                  orientation="h", y=1.05, x=0.5, xanchor="center", yanchor="bottom")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(n):
    if n is None or (isinstance(n, float) and np.isnan(n)): return "—"
    a = abs(n)
    if a >= 1e12: return f"{n/1e12:.2f}T"
    if a >= 1e9:  return f"{n/1e9:.2f}B"
    if a >= 1e6:  return f"{n/1e6:.2f}M"
    if a >= 1e3:  return f"{n/1e3:.1f}K"
    return f"{n:,.2f}"

def detect_cols(df):
    nums, cats = [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            nums.append(c)
        else:
            try:
                pd.to_numeric(df[c].astype(str).str.replace(",","").str.replace("%",""))
                nums.append(c)
            except:
                cats.append(c)
    return nums, cats

def card(label, value, delta=None, delta_up=True):
    delta_html = ""
    if delta:
        cls = "metric-delta-up" if delta_up else "metric-delta-down"
        arrow = "▲" if delta_up else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    return f'''<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>'''

def plotly_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT, legend=LEGEND_DEFAULT)
    return fig

def trend_label(vals):
    if len(vals) < 2: return "ổn định"
    h1 = np.mean(vals[:len(vals)//2])
    h2 = np.mean(vals[len(vals)//2:])
    if h2 > h1*1.05: return "📈 tăng"
    if h2 < h1*0.95: return "📉 giảm"
    return "➡️ ổn định"

def pearson(x, y):
    n = min(len(x), len(y))
    if n < 2: return 0
    xd = np.array(x[:n]) - np.mean(x[:n])
    yd = np.array(y[:n]) - np.mean(y[:n])
    d = np.sqrt((xd**2).sum() * (yd**2).sum())
    return float(xd @ yd / d) if d else 0

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role":"ai","content":"👋 Tôi là **DataLens AI** — analyst theo tư duy MBB.\n\nUpload dữ liệu ở sidebar, sau đó hỏi tôi:\n• Phân tích tài chính / biên lợi nhuận\n• Cạnh tranh — HHI, CR4, SCP\n• Vận hành — chi phí/đơn, năng suất\n• So what & khuyến nghị hành động"}
    ]
if "df" not in st.session_state:
    st.session_state.df = None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 DataLens")
    st.markdown("<div style='color:#8a9ab0;font-size:12px;margin-bottom:16px;'>AI Consulting Analyst</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload dữ liệu (CSV / Excel)", type=["csv","xlsx","xls"], label_visibility="collapsed")

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded, encoding="utf-8", on_bad_lines="skip")
            else:
                df = pd.read_excel(uploaded)
            df.columns = [str(c).strip() for c in df.columns]
            st.session_state.df = df
            st.success(f"✅ {len(df):,} dòng • {len(df.columns)} cột")
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")

    st.markdown("---")
    if st.session_state.df is not None:
        df = st.session_state.df
        nums, cats = detect_cols(df)
        st.markdown(f"<div style='font-size:11px;color:#8a9ab0;'>Cột số: {len(nums)}<br>Cột nhóm: {len(cats)}</div>", unsafe_allow_html=True)
        if nums:
            st.markdown(f"<div style='font-size:10px;color:#4e5d6e;margin-top:4px;'>{', '.join(nums[:5])}</div>", unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
df = st.session_state.df

TAB_NAMES = ["📊 Tổng quan","💰 Tài chính","🏆 Cạnh tranh","⚙️ Vận hành","📋 Dữ liệu thô","🤖 AI Chat"]
tabs = st.tabs(TAB_NAMES)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TỔNG QUAN
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    if df is None:
        st.info("⬆️ Upload file CSV hoặc Excel ở sidebar để bắt đầu phân tích.")
    else:
        nums, cats = detect_cols(df)
        st.markdown(f"**File:** `{uploaded.name if uploaded else 'đã tải'}` • **{len(df):,} dòng** • **{len(df.columns)} cột**")
        st.markdown("---")

        # KPI cards
        kpi_cols = nums[:4]
        if kpi_cols:
            cols = st.columns(len(kpi_cols))
            for i, col_name in enumerate(kpi_cols):
                vals = pd.to_numeric(df[col_name].astype(str).str.replace(",",""), errors="coerce").dropna()
                total = vals.sum()
                avg = vals.mean()
                h1 = vals.iloc[:len(vals)//2].mean() if len(vals) > 1 else avg
                h2 = vals.iloc[len(vals)//2:].mean() if len(vals) > 1 else avg
                growth = (h2-h1)/abs(h1)*100 if h1 else 0
                with cols[i]:
                    st.markdown(card(col_name, fmt(total), f"{abs(growth):.1f}% vs nửa đầu", growth >= 0), unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            if len(nums) >= 1:
                n = nums[0]
                vals = pd.to_numeric(df[n].astype(str).str.replace(",",""), errors="coerce")
                labels = df[cats[0]].astype(str) if cats else df.index.astype(str)
                fig = px.bar(x=labels[:50], y=vals[:50], title=f"Phân phối — {n}")
                fig.update_traces(marker_color="rgba(45,212,191,0.53)", marker_line_color=TEAL, marker_line_width=1)
                st.plotly_chart(plotly_fig(fig), use_container_width=True)

        with c2:
            if len(nums) >= 2:
                n = nums[1]
                vals = pd.to_numeric(df[n].astype(str).str.replace(",",""), errors="coerce")
                labels = df[cats[0]].astype(str) if cats else df.index.astype(str)
                fig = px.line(x=labels[:50], y=vals[:50], title=f"Xu hướng — {n}")
                fig.update_traces(line_color=COLORS[1], line_width=2)
                fig.update_traces(fill="tozeroy", fillcolor="rgba(96,165,250,0.13)")
                st.plotly_chart(plotly_fig(fig), use_container_width=True)

        # Auto insights
        st.markdown("### 💡 Auto Insights")
        ins_cols = st.columns(2)
        insight_idx = 0
        for n in nums[:3]:
            vals = pd.to_numeric(df[n].astype(str).str.replace(",",""), errors="coerce").dropna()
            if vals.empty: continue
            avg = vals.mean()
            max_v = vals.max()
            max_idx = vals.idxmax()
            max_label = str(df.loc[max_idx, cats[0]]) if cats and max_idx in df.index else f"Hàng {max_idx}"
            trend = trend_label(vals.tolist())
            pct = (max_v/avg - 1)*100
            with ins_cols[insight_idx % 2]:
                st.markdown(f'''<div class="insight-box">
                    <div class="insight-title">📌 {n}</div>
                    <div class="insight-text"><b>Fact:</b> "{max_label}" đạt {fmt(max_v)} — cao hơn TB {pct:.1f}%. Xu hướng tổng thể: {trend}.<br>
                    <b>So what:</b> {"Đây là điểm ngoại lệ cần phân tích sâu hơn về nguyên nhân." if pct > 50 else "Phân phối tương đối đồng đều, không có outlier nguy hiểm."}</div>
                </div>''', unsafe_allow_html=True)
            insight_idx += 1

        if cats:
            col = cats[0]
            freq = df[col].value_counts().head(5)
            top_share = freq.iloc[0]/len(df)*100
            with ins_cols[insight_idx % 2]:
                st.markdown(f'''<div class="insight-box">
                    <div class="insight-title">👥 Nhóm "{col}"</div>
                    <div class="insight-text"><b>Fact:</b> "{freq.index[0]}" chiếm {top_share:.1f}% ({freq.iloc[0]} dòng).<br>
                    <b>So what:</b> {"Rủi ro phụ thuộc vào 1 nhóm — cần diversify." if top_share > 50 else "Phân phối tương đối cân bằng — cạnh tranh lành mạnh."}</div>
                </div>''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TÀI CHÍNH
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    if df is None:
        st.info("⬆️ Upload dữ liệu để phân tích tài chính.")
    else:
        nums, cats = detect_cols(df)
        if not nums:
            st.warning("Không tìm thấy cột số trong dữ liệu.")
        else:
            kw = lambda kws: next((c for c in nums if any(k in c.lower() for k in kws)), nums[0])
            rev_col = kw(["revenue","doanh thu","doanh_thu","dt","sales","net_sales","oanh"])
            cost_col = kw(["cost","chi phi","chi_phi","cp","expense","cogs","ost"])
            profit_col = kw(["profit","lợi nhuận","loi nhuan","net income","net_income","income","ln","rofit"])
            labels = df[cats[0]].astype(str) if cats else pd.Series(range(len(df))).astype(str)

            def get_num(col):
                return pd.to_numeric(df[col].astype(str).str.replace(",","").str.replace("%",""), errors="coerce")

            rev_vals = get_num(rev_col).dropna()
            profit_vals = get_num(profit_col)
            rev_all = get_num(rev_col)
            margin = (profit_vals / rev_all * 100).replace([np.inf, -np.inf], np.nan)

            avg_margin = margin.dropna().mean()
            avg_rev = rev_vals.mean()
            total_rev = rev_vals.sum()

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(card("Tổng doanh thu", fmt(total_rev)), unsafe_allow_html=True)
            with c2: st.markdown(card("TB / kỳ", fmt(avg_rev)), unsafe_allow_html=True)
            with c3:
                m_cls = "up" if avg_margin > 15 else ("neutral" if avg_margin > 0 else "down")
                m_icon = "✅" if avg_margin > 15 else ("⚡" if avg_margin > 0 else "🔴")
                st.markdown(card("Biên LN trung bình", f"{avg_margin:.1f}%", f"{m_icon} {'Tốt' if avg_margin>15 else 'Mỏng' if avg_margin>0 else 'Âm'}", avg_margin > 0), unsafe_allow_html=True)
            with c4: st.markdown(card("Xu hướng doanh thu", trend_label(rev_vals.tolist())), unsafe_allow_html=True)

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(x=labels[:40], y=get_num(rev_col)[:40], title=f"Doanh thu ({rev_col})")
                fig.update_traces(marker_color="rgba(45,212,191,0.4)", marker_line_color=TEAL, marker_line_width=1.5)
                st.plotly_chart(plotly_fig(fig), use_container_width=True)
            with c2:
                fig = px.line(x=labels[:40], y=margin[:40], title="Biên lợi nhuận (%)")
                fig.update_traces(line_color=COLORS[3], line_width=2, fill="tozeroy", fillcolor="rgba(52,211,153,0.13)")
                fig.add_hline(y=0, line_dash="dash", line_color="#f87171", line_width=1)
                st.plotly_chart(plotly_fig(fig), use_container_width=True)

            # Multi-metric radar
            st.markdown("##### 📡 Radar — Hiệu suất đa chỉ số (chuẩn hóa)")
            radar_cols = nums[:5]
            radar_vals = []
            for c in radar_cols:
                v = get_num(c).dropna()
                mx = v.max()
                radar_vals.append(round(v.mean()/mx*100, 1) if mx else 0)
            fig = go.Figure(go.Scatterpolar(r=radar_vals+[radar_vals[0]], theta=radar_cols+[radar_cols[0]],
                fill="toself", fillcolor="rgba(45,212,191,0.2)", line_color=TEAL))
            fig.update_layout(**{**PLOTLY_LAYOUT, "polar": dict(
                radialaxis=dict(visible=True, range=[0,100], gridcolor=GRID, tickfont=dict(size=9), color=TEXT_COLOR),
                angularaxis=dict(gridcolor=GRID, tickfont=dict(size=10), color=TEXT_COLOR),
                bgcolor=SURFACE
            )})
            st.plotly_chart(fig, use_container_width=True)

            # Fact → Insight → So what
            st.markdown("---")
            st.markdown(f"""<div class="insight-box">
                <div class="insight-title">💡 Fact → Insight → So what</div>
                <div class="insight-text">
                <b>Fact:</b> {rev_col} trung bình {fmt(avg_rev)}, xu hướng {trend_label(rev_vals.tolist())}. Biên LN TB {avg_margin:.1f}%.<br>
                <b>Insight:</b> {"Biên lợi nhuận tốt → mô hình kinh doanh bền vững, có room tái đầu tư." if avg_margin > 15 else "Biên mỏng → doanh nghiệp dễ bị squeeze nếu chi phí tăng hoặc giá giảm áp lực."}<br>
                <b>So what:</b> {"Ưu tiên bảo vệ biên bằng cách kiểm soát chi phí biến đổi và tối ưu mix sản phẩm." if avg_margin > 0 else "Khẩn cấp rà soát cấu trúc chi phí — biên âm không bền vững trung hạn."}
                </div></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CẠNH TRANH (SCP)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    if df is None:
        st.info("⬆️ Upload dữ liệu để phân tích cạnh tranh.")
    else:
        nums, cats = detect_cols(df)
        if not cats or not nums:
            st.warning("Cần ít nhất 1 cột văn bản (tên nhóm/công ty) và 1 cột số.")
        else:
            group_col = cats[0]
            metric_col = nums[0]

            def get_num2(col):
                return pd.to_numeric(df[col].astype(str).str.replace(",",""), errors="coerce")

            grouped = df.groupby(group_col)[metric_col].apply(
                lambda x: pd.to_numeric(x.astype(str).str.replace(",",""), errors="coerce").sum()
            ).sort_values(ascending=False).head(10)

            total = grouped.sum()
            shares = (grouped/total*100).round(2)
            hhi = float((shares**2).sum())
            cr4 = float(shares.head(4).sum())
            leader = grouped.index[0]
            leader_share = float(shares.iloc[0])

            hhi_label = "Tập trung cao (>2500)" if hhi > 2500 else ("Tập trung vừa" if hhi > 1500 else "Cạnh tranh lành mạnh")
            hhi_icon  = "🔴" if hhi > 2500 else ("🟡" if hhi > 1500 else "🟢")

            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown(card("Leader", str(leader)[:18], f"{leader_share:.1f}% thị phần"), unsafe_allow_html=True)
            with c2: st.markdown(card("HHI Index", f"{hhi:.0f}", f"{hhi_icon} {hhi_label}", hhi < 2500), unsafe_allow_html=True)
            with c3: st.markdown(card("CR4", f"{cr4:.1f}%", "Top 4 cộng lại"), unsafe_allow_html=True)
            with c4: st.markdown(card("Số nhóm", str(len(grouped)), group_col), unsafe_allow_html=True)

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(values=grouped.values, names=grouped.index, title=f"Thị phần theo {group_col}", hole=0.4)
                fig.update_traces(textposition="inside", textinfo="percent+label", marker=dict(colors=COLORS[:len(grouped)]))
                st.plotly_chart(plotly_fig(fig), use_container_width=True)
            with c2:
                fig = px.bar(x=grouped.values[::-1], y=grouped.index[::-1], orientation="h", title=f"{metric_col} theo nhóm")
                fig.update_traces(marker_color=[COLORS[i%len(COLORS)] for i in range(len(grouped))])
                st.plotly_chart(plotly_fig(fig), use_container_width=True)

            # SCP Cards
            st.markdown("##### 🔍 SCP Framework")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown(f'''<div class="scp-box">
                    <div class="scp-title">🏗 Structure</div>
                    <div class="scp-text">HHI = {hhi:.0f} → {hhi_label}<br>CR4 = {cr4:.1f}%<br>{len(grouped)} nhóm cạnh tranh<br><b>So what:</b> {"Thị trường tập trung — cần giám sát hành vi định giá" if hhi > 1500 else "Cạnh tranh đủ — khó có giá độc quyền"}</div>
                </div>''', unsafe_allow_html=True)
            with sc2:
                st.markdown(f'''<div class="scp-box">
                    <div class="scp-title">⚡ Conduct</div>
                    <div class="scp-text">Leader "{leader}" kiểm soát {leader_share:.1f}%<br>{"→ Có khả năng dẫn dắt giá thị trường" if leader_share > 40 else "→ Không có dominance rõ ràng"}<br><b>So what:</b> {"Theo dõi hành vi trợ giá, tích hợp dọc" if leader_share > 40 else "Differentiation là key win"}</div>
                </div>''', unsafe_allow_html=True)
            with sc3:
                st.markdown(f'''<div class="scp-box">
                    <div class="scp-title">📈 Performance</div>
                    <div class="scp-text">Tổng {metric_col}: {fmt(total)}<br>TB/nhóm: {fmt(total/len(grouped))}<br><b>So what:</b> {"Phân bổ không đều — top nhóm hút phần lớn giá trị" if leader_share > 30 else "Giá trị phân bổ tương đối đều giữa các nhóm"}</div>
                </div>''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VẬN HÀNH
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    if df is None:
        st.info("⬆️ Upload dữ liệu để phân tích vận hành.")
    else:
        nums, cats = detect_cols(df)
        if not nums:
            st.warning("Cần ít nhất 1 cột số.")
        else:
            def get_num3(col):
                return pd.to_numeric(df[col].astype(str).str.replace(",",""), errors="coerce")

            # Stats
            st.markdown("##### 📐 Thống kê mô tả")
            stats = []
            for c in nums[:6]:
                v = get_num3(c).dropna()
                if v.empty: continue
                avg = v.mean(); std = v.std(); cv = std/abs(avg)*100 if avg else 0
                stats.append({"Chỉ số": c, "Trung bình": fmt(avg), "Max": fmt(v.max()),
                               "Min": fmt(v.min()), "Std": fmt(std), "CV (%)": f"{cv:.1f}%",
                               "Xu hướng": trend_label(v.tolist())})
            if stats:
                st.dataframe(pd.DataFrame(stats).set_index("Chỉ số"), use_container_width=True)

            st.markdown("---")
            labels = df[cats[0]].astype(str) if cats else pd.Series(range(len(df))).astype(str)

            # Multi-series trend
            fig = go.Figure()
            for i, c in enumerate(nums[:4]):
                v = get_num3(c)[:40]
                fig.add_trace(go.Scatter(x=labels[:40], y=v, mode="lines", name=c,
                    line=dict(color=COLORS[i], width=2)))
            fig.update_layout(**PLOTLY_LAYOUT, legend=LEGEND_TOP, title="Xu hướng đa chỉ số", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            # Scatter correlation
            if len(nums) >= 2:
                c1, c2 = st.columns(2)
                with c1:
                    v1 = get_num3(nums[0]); v2 = get_num3(nums[1])
                    corr_df = pd.DataFrame({nums[0]: v1, nums[1]: v2}).dropna()
                    r = pearson(corr_df[nums[0]].tolist(), corr_df[nums[1]].tolist())
                    fig = px.scatter(corr_df, x=nums[0], y=nums[1],
    title=f"Tương quan r={r:.2f}",
    color_discrete_sequence=[TEAL])
x_vals = corr_df[nums[0]].dropna().values
y_vals = corr_df[nums[1]].dropna().values
if len(x_vals) > 1:
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
    fig.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode="lines",
        name="Trendline",
        line=dict(color="#f59e0b", width=2, dash="dash")))
                    st.plotly_chart(plotly_fig(fig), use_container_width=True)
                with c2:
                    # CV bar chart
                    cv_data = []
                    for c in nums[:6]:
                        v = get_num3(c).dropna()
                        if v.empty or v.mean() == 0: continue
                        cv_data.append({"Chỉ số": c, "CV (%)": v.std()/abs(v.mean())*100})
                    if cv_data:
                        cv_df = pd.DataFrame(cv_data)
                        fig = px.bar(cv_df, x="Chỉ số", y="CV (%)",
                            title="Mức độ biến động (CV)",
                            color="CV (%)", color_continuous_scale=["#34d399","#f59e0b","#f87171"])
                        st.plotly_chart(plotly_fig(fig), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DỮ LIỆU THÔ
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    if df is None:
        st.info("⬆️ Upload dữ liệu để xem bảng.")
    else:
        nums, cats = detect_cols(df)
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            search = st.text_input("🔍 Tìm kiếm", placeholder="Nhập từ khóa...", label_visibility="collapsed")
        with c2:
            sort_col = st.selectbox("Sắp xếp theo", ["(mặc định)"] + list(df.columns), label_visibility="collapsed")
        with c3:
            sort_asc = st.radio("Thứ tự", ["↑ Tăng","↓ Giảm"], horizontal=True, label_visibility="collapsed") == "↑ Tăng"

        display_df = df.copy()
        if search:
            mask = display_df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
            display_df = display_df[mask]
        if sort_col != "(mặc định)":
            try:
                display_df = display_df.sort_values(sort_col, ascending=sort_asc)
            except: pass

        st.caption(f"Hiển thị {min(500, len(display_df)):,} / {len(display_df):,} dòng")
        st.dataframe(display_df.head(500), use_container_width=True, height=500)

        # Export
        csv = display_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("⬇️ Tải CSV", csv, "export.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    # Display history
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(f'<div class="chat-user">👤 {content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 {content}</div>', unsafe_allow_html=True)

    # Hint buttons
    hint_cols = st.columns(4)
    hints = ["📊 Tổng quan dữ liệu","💰 Phân tích biên lợi nhuận","🏆 Ai dẫn đầu thị trường?","⚠️ Rủi ro chính?"]
    for i, h in enumerate(hints):
        if hint_cols[i].button(h, key=f"hint_{i}"):
            st.session_state._hint = h

    q = st.chat_input("Hỏi AI Analyst về dữ liệu của anh...")
    if hasattr(st.session_state, "_hint"):
        q = st.session_state._hint
        del st.session_state._hint

    if q:
        st.session_state.chat_history.append({"role":"user","content":q})

        def ai_reply(q):
            if df is None:
                return "⚠️ Anh chưa upload dữ liệu. Hãy upload CSV/Excel ở sidebar trước."
            nums, cats = detect_cols(df)
            ql = q.lower()

            def get_v(col):
                return pd.to_numeric(df[col].astype(str).str.replace(",",""), errors="coerce").dropna()

            def stats_str(col):
                v = get_v(col)
                return f"TB={fmt(v.mean())}, Max={fmt(v.max())}, Xu hướng: {trend_label(v.tolist())}"

            if any(k in ql for k in ["tổng quan","tóm tắt","summary","mô tả","overview"]):
                lines = [f"📊 **Tổng quan dữ liệu**\n\n• {len(df):,} dòng × {len(df.columns)} cột"]
                for c in nums[:3]:
                    lines.append(f"• **{c}**: {stats_str(c)}")
                if cats:
                    freq = df[cats[0]].value_counts()
                    lines.append(f"• **Nhóm `{cats[0]}`**: Top 3 là {', '.join(f'{k}({v})' for k,v in freq.head(3).items())}")
                lines.append(f"\n💡 **So what:** Dataset {'lớn — nên lọc nhóm trước khi phân tích sâu.' if len(df)>1000 else 'nhỏ — phù hợp phân tích toàn bộ.'}")
                return "\n".join(lines)

            if any(k in ql for k in ["biên","lợi nhuận","margin","tài chính","profit","doanh thu"]):
                kw = lambda kws: next((c for c in nums if any(k in c.lower() for k in kws)), nums[0] if nums else None)
                rev = kw(["revenue","doanh","sales"])
                prf = kw(["profit","ln","loi","income"])
                r = [f"💰 **Phân tích tài chính**\n"]
                if rev:
                    v = get_v(rev)
                    r.append(f"**{rev}:** Tổng={fmt(v.sum())}, TB={fmt(v.mean())}, Xu hướng: {trend_label(v.tolist())}")
                if prf and rev:
                    vp = get_v(prf); vr = get_v(rev)
                    avg_m = (vp.mean()/vr.mean()*100) if vr.mean() else 0
                    icon = "✅" if avg_m>15 else ("⚡" if avg_m>0 else "🔴")
                    r.append(f"\n**Biên LN:** {avg_m:.1f}% {icon}")
                r.append(f"\n💡 **So what:** {'Biên tốt — bảo vệ bằng kiểm soát chi phí.' if (prf and vp.mean()>0) else 'Cần rà soát cấu trúc chi phí ngay.'}")
                return "\n".join(r)

            if any(k in ql for k in ["thị phần","cạnh tranh","hhi","scp","dẫn đầu","market","leader"]):
                if not cats:
                    return "Cần cột văn bản (tên công ty/nhóm) để phân tích cạnh tranh."
                grouped = df.groupby(cats[0])[nums[0]].apply(lambda x: get_v(nums[0]).reindex(x.index).sum()).sort_values(ascending=False)
                total = grouped.sum()
                shares = grouped/total*100
                hhi = float((shares**2).sum())
                cr4 = float(shares.head(4).sum())
                r = [f"🏆 **Phân tích cạnh tranh — SCP**\n"]
                r.append(f"**Structure:** HHI={hhi:.0f} ({'Tập trung cao' if hhi>2500 else 'Tập trung vừa' if hhi>1500 else 'Cạnh tranh'}), CR4={cr4:.1f}%")
                r.append(f"**Conduct:** Leader \"{grouped.index[0]}\" chiếm {shares.iloc[0]:.1f}%")
                r.append(f"**Performance:** Tổng {nums[0]}={fmt(total)}, TB/nhóm={fmt(total/len(grouped))}")
                r.append(f"\n💡 **So what:** {'Leader đang dominate — rủi ro phụ thuộc nếu là đối tác, cơ hội nếu là ta.' if shares.iloc[0]>40 else 'Thị trường còn cạnh tranh — differentiation và hiệu suất là key.'}")
                return "\n".join(r)

            if any(k in ql for k in ["rủi ro","risk","nguy","cảnh báo"]):
                risks = []
                for c in nums[:4]:
                    v = get_v(c)
                    if trend_label(v.tolist()) == "📉 giảm":
                        risks.append(f"⚠️ **{c}** đang có xu hướng giảm")
                    if v.std() > v.mean()*0.5:
                        risks.append(f"🔴 **{c}** biến động cao (CV={v.std()/v.mean()*100:.0f}%) — không ổn định")
                if cats:
                    freq = df[cats[0]].value_counts()
                    if freq.iloc[0]/len(df) > 0.5:
                        risks.append(f"🟡 Tập trung vào nhóm \"{freq.index[0]}\" ({freq.iloc[0]/len(df)*100:.0f}%) — rủi ro phụ thuộc")
                if not risks:
                    risks = ["✅ Không phát hiện rủi ro rõ ràng từ cấu trúc dữ liệu hiện tại."]
                return f"⚠️ **Rủi ro phát hiện:**\n\n" + "\n".join(risks) + "\n\n💡 **Ưu tiên xử lý:** xu hướng giảm > tập trung nhóm > biến động cao."

            if any(k in ql for k in ["so what","khuyến nghị","nên làm","hành động","now what"]):
                r = ["🎯 **So what & Now what**\n"]
                for c in nums[:2]:
                    v = get_v(c)
                    r.append(f"• **{c}**: {trend_label(v.tolist())} — {'Tiếp tục đầu tư vào driver tăng trưởng' if 'tăng' in trend_label(v.tolist()) else 'Ưu tiên kiểm soát chi phí và tìm nguồn tăng trưởng mới'}")
                r.append(f"\n📌 **3 bước tiếp theo:**")
                r.append(f"1. Phân tích chi tiết nhóm \"{cats[0] if cats else 'chính'}\" để tìm segment hiệu quả nhất")
                r.append(f"2. Tập trung vào {nums[0] if nums else 'chỉ số chính'} — đây là driver quan trọng")
                r.append(f"3. Kiểm tra outlier — có systematic không hay chỉ one-off?")
                return "\n".join(r)

            # Generic
            r = [f"📋 **Dữ liệu \"{uploaded.name if uploaded else 'của anh'}\":**\n"]
            for c in nums[:2]:
                r.append(f"• **{c}**: {stats_str(c)}")
            r.append(f"\n💬 Hỏi cụ thể hơn để tôi phân tích sâu, ví dụ:\n• \"Phân tích biên lợi nhuận\"\n• \"Nhóm nào dẫn đầu?\"\n• \"Rủi ro chính là gì?\"")
            return "\n".join(r)

        reply = ai_reply(q)
        st.session_state.chat_history.append({"role":"ai","content":reply})
        st.rerun()
