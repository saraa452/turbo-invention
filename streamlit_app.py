from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

AUTHOR_NAME: Final = "Sara Oliveira Guimarães Nascimento"
AUTHOR_ROLE: Final = "Analista de Negócios"
AUTHOR_COMPANY: Final = "Dalumia Consultoria ME"

# Paleta corporativa
COLORS: Final = ["#1B4F72", "#2E86C1", "#48C9B0", "#F39C12", "#E74C3C", "#8E44AD", "#2C3E50", "#27AE60"]
COLOR_PRIMARY: Final = "#1B4F72"
COLOR_SECONDARY: Final = "#2E86C1"
COLOR_SUCCESS: Final = "#27AE60"
COLOR_WARNING: Final = "#F39C12"
COLOR_DANGER: Final = "#E74C3C"

PLOTLY_LAYOUT = dict(
    font=dict(family="Segoe UI, Roboto, sans-serif", size=13, color="#2C3E50"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor="white", font_size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

BASE_DIR = Path(__file__).resolve().parent
ARTICLE_DIR = BASE_DIR / "portfolio_articles"

SALES_FILE = BASE_DIR / "sales_analysis_project" / "data" / "sales_transactions.csv"
PRODUCTS_FILE = BASE_DIR / "sales_analysis_project" / "data" / "products.csv"
STORES_FILE = BASE_DIR / "sales_analysis_project" / "data" / "stores.csv"
CUSTOMER_FILE = BASE_DIR / "customer_segmentation" / "dataset" / "customer_behavior.csv"
KPI_FILE = BASE_DIR / "business_dashboard" / "datasets" / "business_kpis.csv"
CHANNEL_FILE = BASE_DIR / "business_dashboard" / "datasets" / "channel_performance.csv"
ETL_SCRIPT = BASE_DIR / "etl_pipeline_project" / "scripts" / "etl_pipeline.py"
PIPELINE_CONFIG = BASE_DIR / "etl_pipeline_project" / "pipeline" / "pipeline_config.yaml"
ETL_OUTPUT_DIR = BASE_DIR / "etl_pipeline_project" / "pipeline" / "output"

ARTICLE_FILES = {
    "Sales Analysis": "sales_analysis_article.html",
    "Customer Segmentation": "customer_segmentation_article.html",
    "Business Dashboard": "business_dashboard_article.html",
    "ETL Pipeline": "etl_pipeline_article.html",
}


def format_currency(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def calculate_delta(current: float, previous: float | None, suffix: str = "") -> str | None:
    if previous is None or previous == 0:
        return None
    variation = (current - previous) / previous
    sign = "+" if variation >= 0 else ""
    suffix_text = f" {suffix}" if suffix else ""
    return f"{sign}{variation * 100:.1f}%{suffix_text}"


def styled_plotly(fig: go.Figure, height: int = 420) -> go.Figure:
    """Aplica layout corporativo padrão a qualquer figura Plotly."""
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#F0F0F0")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#F0F0F0")
    return fig


@st.cache_data(show_spinner=False)
def load_sales_model() -> pd.DataFrame:
    sales = pd.read_csv(SALES_FILE)
    products = pd.read_csv(PRODUCTS_FILE)
    stores = pd.read_csv(STORES_FILE)

    sales["date"] = pd.to_datetime(sales["date"])
    sales["gross_revenue"] = sales["quantity"] * sales["unit_price"]
    sales["net_revenue"] = sales["gross_revenue"] * (1 - sales["discount_pct"])
    sales["discount_value"] = sales["gross_revenue"] - sales["net_revenue"]
    sales["month"] = sales["date"].dt.to_period("M").astype(str)
    sales["day_of_week"] = sales["date"].dt.day_name()

    model = (
        sales.merge(
            products[["product_id", "product_name", "category", "cost_price"]],
            on="product_id",
            how="left",
        )
        .merge(
            stores[["store_id", "store_name", "city", "state", "region"]],
            on="store_id",
            how="left",
        )
        .sort_values("date")
    )

    model["cost_price"] = model["cost_price"].fillna(0.0)
    model["total_cost"] = model["quantity"] * model["cost_price"]
    model["gross_margin"] = model["net_revenue"] - model["total_cost"]
    model["margin_pct"] = (model["gross_margin"] / model["net_revenue"]).fillna(0.0)
    return model


@st.cache_data(show_spinner=False)
def load_customer_data() -> pd.DataFrame:
    df = pd.read_csv(CUSTOMER_FILE)
    df["ltv_estimate"] = df["avg_order_value"] * df["frequency_12m"] * (df["tenure_months"] / 12)
    return df


@st.cache_data(show_spinner=False)
def load_kpi_data() -> pd.DataFrame:
    kpis = pd.read_csv(KPI_FILE)
    kpis["month"] = pd.to_datetime(kpis["month"])
    kpis["month_label"] = kpis["month"].dt.strftime("%Y-%m")
    kpis["profit_margin"] = (kpis["profit"] / kpis["revenue"]).fillna(0.0)
    kpis["expense_ratio"] = (kpis["expenses"] / kpis["revenue"]).fillna(0.0)
    if len(kpis) > 1:
        kpis["revenue_growth"] = kpis["revenue"].pct_change()
        kpis["profit_growth"] = kpis["profit"].pct_change()
    else:
        kpis["revenue_growth"] = 0.0
        kpis["profit_growth"] = 0.0
    return kpis.sort_values("month")


@st.cache_data(show_spinner=False)
def load_channel_data() -> pd.DataFrame:
    channels = pd.read_csv(CHANNEL_FILE)
    channels["month"] = pd.to_datetime(channels["month"])
    channels["month_label"] = channels["month"].dt.strftime("%Y-%m")
    channels["conversion_rate"] = (channels["conversions"] / channels["sessions"]).fillna(0.0)
    channels["estimated_acquisition_cost"] = channels["conversions"] * channels["cac"]
    channels["roi_proxy"] = (
        channels["revenue"] / channels["estimated_acquisition_cost"].replace(0, pd.NA)
    ).fillna(0.0)
    return channels.sort_values("month")


def render_html_article(article_filename: str, height: int = 900) -> None:
    article_path = ARTICLE_DIR / article_filename
    if not article_path.exists():
        st.error(f"Artigo não encontrado: {article_filename}")
        return
    html_content = article_path.read_text(encoding="utf-8")
    components.html(html_content, height=height, scrolling=True)


def apply_ui_style() -> None:
    st.markdown(
        """
<style>
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
    border: 1px solid #d1d9e6;
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    font-size: 0.82rem;
    color: #5a6a7e;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700;
    color: #1B4F72;
}
[data-testid="stSidebar"] {
    border-right: 2px solid #1B4F72;
    background: linear-gradient(180deg, #fafbfc 0%, #f0f4f8 100%);
}
div.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
div.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 600;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_profile() -> None:
    st.sidebar.markdown("### Profissional Responsável")
    st.sidebar.markdown(f"**{AUTHOR_NAME}**")
    st.sidebar.caption(AUTHOR_ROLE)
    st.sidebar.caption(AUTHOR_COMPANY)
    st.sidebar.markdown("---")


def render_author_signature() -> None:
    st.markdown("---")
    st.caption(f"Autora: {AUTHOR_NAME} · {AUTHOR_ROLE} · {AUTHOR_COMPANY}")


def show_home_page() -> None:
    st.title("Portfólio Executivo de Dados")
    st.subheader("Dashboards e relatórios orientados a decisão de negócio")

    sales = load_sales_model()
    customers = load_customer_data()
    kpis = load_kpi_data()

    latest_kpi = kpis.iloc[-1]
    prev_kpi = kpis.iloc[-2] if len(kpis) > 1 else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Receita acumulada",
        format_currency(float(sales["net_revenue"].sum())),
    )
    col2.metric("Clientes monitorados", f"{customers['customer_id'].nunique()}")
    col3.metric(
        "Lucro (último mês)",
        format_currency(float(latest_kpi["profit"])),
        calculate_delta(
            float(latest_kpi["profit"]),
            float(prev_kpi["profit"]) if prev_kpi is not None else None,
        ),
    )
    col4.metric(
        "NPS (último mês)",
        f"{int(latest_kpi['nps'])}",
        calculate_delta(
            float(latest_kpi["nps"]),
            float(prev_kpi["nps"]) if prev_kpi is not None else None,
        ),
    )

    # Mini trend charts
    home_col1, home_col2 = st.columns(2)
    with home_col1:
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(
            x=kpis["month_label"], y=kpis["revenue"],
            fill="tozeroy", fillcolor="rgba(46,134,193,0.15)",
            line=dict(color=COLOR_SECONDARY, width=3),
            name="Receita",
        ))
        fig_rev.add_trace(go.Scatter(
            x=kpis["month_label"], y=kpis["profit"],
            fill="tozeroy", fillcolor="rgba(39,174,96,0.12)",
            line=dict(color=COLOR_SUCCESS, width=3),
            name="Lucro",
        ))
        styled_plotly(fig_rev, 300).update_layout(title="Tendência Receita & Lucro")
        st.plotly_chart(fig_rev, use_container_width=True)

    with home_col2:
        seg_counts = customers["segment_hint"].value_counts()
        fig_seg = go.Figure(go.Pie(
            labels=seg_counts.index, values=seg_counts.values,
            hole=0.55, marker_colors=COLORS,
            textinfo="label+percent", textposition="outside",
        ))
        styled_plotly(fig_seg, 300).update_layout(
            title="Composição da Base de Clientes",
            showlegend=False,
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown(
        """
### Posicionamento do portfólio

Este ambiente demonstra competências de análise e gestão de indicadores em quatro frentes:
- **Performance comercial** com recortes de receita, margem e mix de produtos;
- **Segmentação de clientes** com leitura de comportamento, risco e LTV;
- **Dashboard executivo** com acompanhamento de aquisição, retenção e P&L;
- **Pipeline ETL** para governança e disponibilidade dos dados analíticos.
        """
    )

    st.info("Use o menu lateral para abrir dashboards, relatórios executivos e artigos técnicos dos projetos.")
    render_author_signature()


def show_executive_report_page() -> None:
    st.title("Relatório Executivo Integrado")
    st.write("Resumo consolidado para tomada de decisão com foco em crescimento sustentável.")

    sales = load_sales_model()
    customers = load_customer_data()
    kpis = load_kpi_data()

    latest_kpi = kpis.iloc[-1]
    top_category = (
        sales.groupby("category", as_index=False)["net_revenue"]
        .sum()
        .sort_values("net_revenue", ascending=False)
        .iloc[0]
    )
    risk_clients = int((customers["recency_days"] >= 60).sum())

    # Gauge indicators
    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
    with gauge_col1:
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(latest_kpi["profit_margin"]) * 100,
            title={"text": "Margem de Lucro (%)"},
            gauge=dict(
                axis=dict(range=[0, 40]),
                bar=dict(color=COLOR_PRIMARY),
                steps=[
                    dict(range=[0, 15], color="#FADBD8"),
                    dict(range=[15, 25], color="#FEF9E7"),
                    dict(range=[25, 40], color="#D5F5E3"),
                ],
                threshold=dict(line=dict(color=COLOR_DANGER, width=3), thickness=0.8, value=20),
            ),
            number=dict(suffix="%"),
        ))
        styled_plotly(fig_g1, 280)
        st.plotly_chart(fig_g1, use_container_width=True)

    with gauge_col2:
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(latest_kpi["nps"]),
            title={"text": "NPS"},
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color=COLOR_SECONDARY),
                steps=[
                    dict(range=[0, 30], color="#FADBD8"),
                    dict(range=[30, 50], color="#FEF9E7"),
                    dict(range=[50, 100], color="#D5F5E3"),
                ],
                threshold=dict(line=dict(color=COLOR_SUCCESS, width=3), thickness=0.8, value=50),
            ),
        ))
        styled_plotly(fig_g2, 280)
        st.plotly_chart(fig_g2, use_container_width=True)

    with gauge_col3:
        fig_g3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(latest_kpi["churn_rate"]) * 100,
            title={"text": "Churn Rate (%)"},
            gauge=dict(
                axis=dict(range=[0, 10]),
                bar=dict(color=COLOR_WARNING),
                steps=[
                    dict(range=[0, 3], color="#D5F5E3"),
                    dict(range=[3, 5], color="#FEF9E7"),
                    dict(range=[5, 10], color="#FADBD8"),
                ],
                threshold=dict(line=dict(color=COLOR_DANGER, width=3), thickness=0.8, value=5),
            ),
            number=dict(suffix="%"),
        ))
        styled_plotly(fig_g3, 280)
        st.plotly_chart(fig_g3, use_container_width=True)

    # P&L Waterfall
    st.markdown("### Decomposição de Resultado (último mês)")
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Receita", "Despesas", "Lucro"],
        y=[float(latest_kpi["revenue"]), -float(latest_kpi["expenses"]), float(latest_kpi["profit"])],
        text=[format_currency(float(latest_kpi["revenue"])),
              format_currency(-float(latest_kpi["expenses"])),
              format_currency(float(latest_kpi["profit"]))],
        textposition="outside",
        connector=dict(line=dict(color="#CBD5E0", width=1)),
        increasing=dict(marker=dict(color=COLOR_SUCCESS)),
        decreasing=dict(marker=dict(color=COLOR_DANGER)),
        totals=dict(marker=dict(color=COLOR_PRIMARY)),
    ))
    styled_plotly(fig_wf, 380).update_layout(title="Waterfall P&L", showlegend=False)
    st.plotly_chart(fig_wf, use_container_width=True)

    # Summary table
    summary_df = pd.DataFrame(
        [
            {
                "Indicador": "Receita líquida total (sales project)",
                "Valor": format_currency(float(sales["net_revenue"].sum())),
                "Leitura executiva": "Volume comercial consolidado do portfólio de vendas.",
            },
            {
                "Indicador": "Categoria líder de receita",
                "Valor": f"{top_category['category']} ({format_currency(float(top_category['net_revenue']))})",
                "Leitura executiva": "Categoria com maior contribuição para crescimento.",
            },
            {
                "Indicador": "Margem de lucro (último mês)",
                "Valor": format_percent(float(latest_kpi["profit_margin"])),
                "Leitura executiva": "Eficiência operacional no período mais recente.",
            },
            {
                "Indicador": "Clientes em risco (recência >= 60 dias)",
                "Valor": str(risk_clients),
                "Leitura executiva": "Base prioritária para iniciativas de retenção.",
            },
        ]
    )
    st.dataframe(summary_df, width="stretch", hide_index=True)

    # Growth trends
    st.markdown("### Crescimento Mês a Mês")
    growth_df = kpis.dropna(subset=["revenue_growth"]).copy()
    if not growth_df.empty:
        fig_growth = go.Figure()
        fig_growth.add_trace(go.Bar(
            x=growth_df["month_label"], y=growth_df["revenue_growth"] * 100,
            name="Receita (%)", marker_color=COLOR_SECONDARY,
            text=growth_df["revenue_growth"].apply(lambda v: f"{v*100:+.1f}%"),
            textposition="outside",
        ))
        fig_growth.add_trace(go.Bar(
            x=growth_df["month_label"], y=growth_df["profit_growth"] * 100,
            name="Lucro (%)", marker_color=COLOR_SUCCESS,
            text=growth_df["profit_growth"].apply(lambda v: f"{v*100:+.1f}%"),
            textposition="outside",
        ))
        styled_plotly(fig_growth, 350).update_layout(barmode="group", yaxis_title="Crescimento (%)")
        st.plotly_chart(fig_growth, use_container_width=True)

    recommendations = pd.DataFrame(
        [
            {
                "Prioridade": "Alta",
                "Ação": "Plano de retenção para clientes at_risk/churn_risk",
                "Impacto esperado": "Redução de churn e recuperação de receita recorrente.",
            },
            {
                "Prioridade": "Alta",
                "Ação": "Revisão de investimento em canais com CAC elevado",
                "Impacto esperado": "Melhor equilíbrio entre aquisição e rentabilidade.",
            },
            {
                "Prioridade": "Média",
                "Ação": "Acelerar campanhas sobre categoria líder",
                "Impacto esperado": "Aumento de receita com menor esforço de conversão.",
            },
            {
                "Prioridade": "Média",
                "Ação": "Industrializar execução ETL com agenda recorrente",
                "Impacto esperado": "Melhor governança e atualização contínua dos indicadores.",
            },
        ]
    )
    st.markdown("### Plano de ação recomendado")
    st.dataframe(recommendations, width="stretch", hide_index=True)
    render_author_signature()


def show_sales_page() -> None:
    st.title("Sales Analysis · Visão Comercial")
    sales = load_sales_model()

    available_months = sorted(sales["month"].unique())
    selected_months = st.multiselect(
        "Período de análise (mês)",
        available_months,
        default=available_months,
    )

    filtered = sales[sales["month"].isin(selected_months)]
    if filtered.empty:
        st.warning("Selecione ao menos um mês para visualizar os indicadores.")
        return

    monthly_view = (
        filtered.groupby("month", as_index=False)
        .agg(
            net_revenue=("net_revenue", "sum"),
            gross_margin=("gross_margin", "sum"),
            discount_value=("discount_value", "sum"),
            active_customers=("customer_id", "nunique"),
            transactions=("transaction_id", "count"),
        )
        .sort_values("month")
    )
    monthly_view["margin_pct"] = (
        monthly_view["gross_margin"] / monthly_view["net_revenue"].replace(0, pd.NA)
    ).fillna(0.0)
    monthly_view["ticket_medio"] = monthly_view["net_revenue"] / monthly_view["transactions"]

    latest = monthly_view.iloc[-1]
    previous = monthly_view.iloc[-2] if len(monthly_view) > 1 else None

    revenue_delta = calculate_delta(
        float(latest["net_revenue"]),
        float(previous["net_revenue"]) if previous is not None else None,
        "vs mês anterior",
    )
    margin_delta = calculate_delta(
        float(latest["gross_margin"]),
        float(previous["gross_margin"]) if previous is not None else None,
        "vs mês anterior",
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Receita líquida", format_currency(float(latest["net_revenue"])), revenue_delta)
    col2.metric("Margem bruta", format_currency(float(latest["gross_margin"])), margin_delta)
    col3.metric("Margem (%)", format_percent(float(latest["margin_pct"])))
    col4.metric("Ticket médio", format_currency(float(latest["ticket_medio"])))
    col5.metric("Clientes ativos", f"{int(filtered['customer_id'].nunique())}")

    tab1, tab2, tab3 = st.tabs(["Tendências", "Mix de Produto", "Análise Detalhada"])

    with tab1:
        # Revenue & Margin area chart
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly_view["month"], y=monthly_view["net_revenue"],
            fill="tozeroy", fillcolor="rgba(46,134,193,0.15)",
            line=dict(color=COLOR_SECONDARY, width=3), name="Receita Líquida",
            hovertemplate="Receita: R$ %{y:,.2f}<extra></extra>",
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_view["month"], y=monthly_view["gross_margin"],
            fill="tozeroy", fillcolor="rgba(39,174,96,0.12)",
            line=dict(color=COLOR_SUCCESS, width=3), name="Margem Bruta",
            hovertemplate="Margem: R$ %{y:,.2f}<extra></extra>",
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_view["month"], y=monthly_view["discount_value"],
            line=dict(color=COLOR_WARNING, width=2, dash="dash"), name="Descontos",
            hovertemplate="Descontos: R$ %{y:,.2f}<extra></extra>",
        ))
        styled_plotly(fig_trend).update_layout(title="Evolução Mensal: Receita, Margem e Descontos")
        st.plotly_chart(fig_trend, use_container_width=True)

        # Revenue by region map-style horizontal bar
        region_summary = (
            filtered.groupby("region", as_index=False)
            .agg(net_revenue=("net_revenue", "sum"), transactions=("transaction_id", "count"),
                 gross_margin=("gross_margin", "sum"))
            .sort_values("net_revenue", ascending=True)
        )
        region_summary["margin_pct"] = (region_summary["gross_margin"] / region_summary["net_revenue"]).fillna(0.0)

        fig_region = go.Figure()
        fig_region.add_trace(go.Bar(
            y=region_summary["region"], x=region_summary["net_revenue"],
            orientation="h", marker_color=COLOR_SECONDARY, name="Receita",
            text=region_summary["net_revenue"].apply(lambda v: format_currency(v)),
            textposition="auto",
            hovertemplate="%{y}: R$ %{x:,.2f}<extra></extra>",
        ))
        styled_plotly(fig_region, 320).update_layout(title="Receita por Região", xaxis_title="Receita (R$)")
        st.plotly_chart(fig_region, use_container_width=True)

    with tab2:
        category_summary = (
            filtered.groupby("category", as_index=False)
            .agg(net_revenue=("net_revenue", "sum"), gross_margin=("gross_margin", "sum"),
                 transactions=("transaction_id", "count"), quantity=("quantity", "sum"))
            .sort_values("net_revenue", ascending=False)
        )
        category_summary["margin_pct"] = (
            category_summary["gross_margin"] / category_summary["net_revenue"]
        ).fillna(0.0)

        cat_col1, cat_col2 = st.columns(2)
        with cat_col1:
            # Treemap by category
            product_tree = (
                filtered.groupby(["category", "product_name"], as_index=False)["net_revenue"].sum()
            )
            fig_tree = px.treemap(
                product_tree, path=["category", "product_name"], values="net_revenue",
                color="net_revenue", color_continuous_scale=["#D6EAF8", "#1B4F72"],
                title="Composição de Receita (Treemap)",
            )
            styled_plotly(fig_tree, 420)
            st.plotly_chart(fig_tree, use_container_width=True)

        with cat_col2:
            # Category donut
            fig_cat_pie = go.Figure(go.Pie(
                labels=category_summary["category"],
                values=category_summary["net_revenue"],
                hole=0.5, marker_colors=COLORS,
                textinfo="label+percent", textposition="outside",
                hovertemplate="%{label}: R$ %{value:,.2f}<extra></extra>",
            ))
            styled_plotly(fig_cat_pie, 420).update_layout(
                title="Participação por Categoria",
                showlegend=False,
            )
            st.plotly_chart(fig_cat_pie, use_container_width=True)

        # Product profitability scatter
        product_prof = (
            filtered.groupby("product_name", as_index=False)
            .agg(net_revenue=("net_revenue", "sum"), gross_margin=("gross_margin", "sum"),
                 quantity=("quantity", "sum"), category=("category", "first"))
        )
        product_prof["margin_pct"] = (product_prof["gross_margin"] / product_prof["net_revenue"]).fillna(0.0)

        fig_scatter = px.scatter(
            product_prof, x="net_revenue", y="margin_pct",
            size="quantity", color="category", color_discrete_sequence=COLORS,
            hover_name="product_name",
            labels={"net_revenue": "Receita Líquida (R$)", "margin_pct": "Margem (%)", "quantity": "Qtd vendida"},
            title="Rentabilidade por Produto (Receita × Margem × Volume)",
        )
        fig_scatter.update_traces(
            marker=dict(line=dict(width=1, color="#2C3E50")),
            hovertemplate="<b>%{hovertext}</b><br>Receita: R$ %{x:,.2f}<br>Margem: %{y:.1%}<br>Qtd: %{marker.size}<extra></extra>",
        )
        styled_plotly(fig_scatter, 420).update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        det_col1, det_col2 = st.columns(2)
        with det_col1:
            # Payment method analysis
            payment_agg = (
                filtered.groupby("payment_method", as_index=False)
                .agg(net_revenue=("net_revenue", "sum"), count=("transaction_id", "count"))
            )
            fig_pay = go.Figure(go.Pie(
                labels=payment_agg["payment_method"],
                values=payment_agg["net_revenue"],
                hole=0.45, marker_colors=COLORS,
                textinfo="label+percent",
                hovertemplate="%{label}: R$ %{value:,.2f} (%{percent})<extra></extra>",
            ))
            styled_plotly(fig_pay, 370).update_layout(
                title="Receita por Forma de Pagamento", showlegend=False,
            )
            st.plotly_chart(fig_pay, use_container_width=True)

        with det_col2:
            # Store performance
            store_perf = (
                filtered.groupby("store_name", as_index=False)
                .agg(net_revenue=("net_revenue", "sum"), gross_margin=("gross_margin", "sum"),
                     transactions=("transaction_id", "count"))
                .sort_values("net_revenue", ascending=True)
            )
            fig_store = go.Figure()
            fig_store.add_trace(go.Bar(
                y=store_perf["store_name"], x=store_perf["net_revenue"],
                orientation="h", marker_color=COLOR_SECONDARY, name="Receita",
                text=store_perf["net_revenue"].apply(lambda v: format_currency(v)),
                textposition="auto",
            ))
            fig_store.add_trace(go.Bar(
                y=store_perf["store_name"], x=store_perf["gross_margin"],
                orientation="h", marker_color=COLOR_SUCCESS, name="Margem",
                text=store_perf["gross_margin"].apply(lambda v: format_currency(v)),
                textposition="auto",
            ))
            styled_plotly(fig_store, 370).update_layout(
                title="Performance por Loja", barmode="group",
            )
            st.plotly_chart(fig_store, use_container_width=True)

        # Pareto analysis (80/20)
        st.markdown("### Análise de Pareto (Curva ABC)")
        pareto = (
            filtered.groupby("product_name", as_index=False)["net_revenue"]
            .sum()
            .sort_values("net_revenue", ascending=False)
        )
        total_rev = pareto["net_revenue"].sum()
        pareto["pct"] = pareto["net_revenue"] / total_rev
        pareto["cumulative_pct"] = pareto["pct"].cumsum()
        pareto["class"] = pareto["cumulative_pct"].apply(
            lambda v: "A" if v <= 0.8 else ("B" if v <= 0.95 else "C")
        )
        color_map = {"A": COLOR_PRIMARY, "B": COLOR_WARNING, "C": "#BDC3C7"}

        fig_pareto = go.Figure()
        for cls in ["A", "B", "C"]:
            subset = pareto[pareto["class"] == cls]
            if not subset.empty:
                fig_pareto.add_trace(go.Bar(
                    x=subset["product_name"], y=subset["net_revenue"],
                    name=f"Classe {cls}", marker_color=color_map[cls],
                    hovertemplate="%{x}: R$ %{y:,.2f}<extra></extra>",
                ))
        fig_pareto.add_trace(go.Scatter(
            x=pareto["product_name"], y=pareto["cumulative_pct"] * total_rev,
            yaxis="y2", name="% Acumulado",
            line=dict(color=COLOR_DANGER, width=2),
            hovertemplate="%{x}: %{text}<extra></extra>",
            text=pareto["cumulative_pct"].apply(lambda v: f"{v:.0%}"),
        ))
        styled_plotly(fig_pareto, 420).update_layout(
            title="Curva ABC de Produtos",
            yaxis=dict(title="Receita (R$)"),
            yaxis2=dict(title="% Acumulado", overlaying="y", side="right", range=[0, total_rev * 1.1]),
            barmode="stack",
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

        # Top clients
        top_clients = (
            filtered.groupby("customer_id", as_index=False)
            .agg(
                receita_liquida=("net_revenue", "sum"),
                transacoes=("transaction_id", "count"),
                ticket_medio=("net_revenue", "mean"),
                margem=("gross_margin", "sum"),
            )
            .sort_values("receita_liquida", ascending=False)
            .head(10)
        )
        top_clients["receita_liquida"] = top_clients["receita_liquida"].map(format_currency)
        top_clients["ticket_medio"] = top_clients["ticket_medio"].map(format_currency)
        top_clients["margem"] = top_clients["margem"].map(format_currency)

        st.markdown("### Top 10 Clientes por Receita Líquida")
        st.dataframe(top_clients, width="stretch", hide_index=True)

    category_summary = (
        filtered.groupby("category", as_index=False)
        .agg(net_revenue=("net_revenue", "sum"))
        .sort_values("net_revenue", ascending=False)
    )
    region_summary = (
        filtered.groupby("region", as_index=False)
        .agg(net_revenue=("net_revenue", "sum"))
        .sort_values("net_revenue", ascending=False)
    )
    st.markdown("### Relatório executivo")
    st.markdown(
        f"""
- Categoria com melhor desempenho no período: **{category_summary.iloc[0]['category']}**.
- Região líder de faturamento: **{region_summary.iloc[0]['region']}**.
- Ticket médio no último mês: **{format_currency(float(latest['ticket_medio']))}**.
- Recomenda-se priorizar estratégia de mix com foco em produtos Classe A (Pareto).
        """
    )

    with st.expander("Artigo do projeto"):
        render_html_article(ARTICLE_FILES["Sales Analysis"])
    render_author_signature()


def show_customer_page() -> None:
    st.title("Customer Segmentation · Visão de Base")
    customers = load_customer_data()

    available_segments = sorted(customers["segment_hint"].unique())
    available_channels = sorted(customers["preferred_channel"].unique())

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_segments = st.multiselect("Segmentos", available_segments, default=available_segments)
    with filter_col2:
        selected_channels = st.multiselect("Canais preferidos", available_channels, default=available_channels)

    filtered = customers[
        customers["segment_hint"].isin(selected_segments)
        & customers["preferred_channel"].isin(selected_channels)
    ]

    if filtered.empty:
        st.warning("A combinação de filtros não retornou clientes.")
        return

    churn_risk_share = (filtered["recency_days"] >= 60).mean()
    total_ltv = filtered["ltv_estimate"].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Clientes", f"{int(filtered['customer_id'].nunique())}")
    col2.metric("Gasto médio 12M", format_currency(float(filtered["total_spent_12m"].mean())))
    col3.metric("Frequência média", f"{filtered['frequency_12m'].mean():.1f}")
    col4.metric("LTV estimado total", format_currency(float(total_ltv)))
    col5.metric("Risco de churn", format_percent(float(churn_risk_share)))

    tab1, tab2, tab3 = st.tabs(["Segmentação", "Comportamento", "Análise RFM"])

    with tab1:
        seg_col1, seg_col2 = st.columns(2)
        with seg_col1:
            # Segment donut
            seg_counts = filtered["segment_hint"].value_counts()
            segment_color_map = {
                "champion": "#1B4F72", "high_value": "#2E86C1", "loyal": "#48C9B0",
                "potential": "#F39C12", "new_customer": "#27AE60",
                "at_risk": "#E74C3C", "churn_risk": "#8E44AD",
            }
            seg_colors = [segment_color_map.get(s, "#BDC3C7") for s in seg_counts.index]
            fig_seg = go.Figure(go.Pie(
                labels=seg_counts.index, values=seg_counts.values,
                hole=0.55, marker_colors=seg_colors,
                textinfo="label+value+percent", textposition="outside",
            ))
            styled_plotly(fig_seg, 400).update_layout(
                title="Distribuição por Segmento", showlegend=False,
            )
            st.plotly_chart(fig_seg, use_container_width=True)

        with seg_col2:
            # Channel × Segment stacked bar
            cross = (
                filtered.groupby(["preferred_channel", "segment_hint"], as_index=False)["customer_id"]
                .count().rename(columns={"customer_id": "clientes"})
            )
            fig_cross = px.bar(
                cross, x="preferred_channel", y="clientes", color="segment_hint",
                color_discrete_map=segment_color_map,
                title="Canal × Segmento", barmode="stack",
                labels={"preferred_channel": "Canal", "clientes": "Clientes", "segment_hint": "Segmento"},
            )
            styled_plotly(fig_cross, 400)
            st.plotly_chart(fig_cross, use_container_width=True)

        # Segment metrics radar
        seg_metrics = (
            filtered.groupby("segment_hint", as_index=False)
            .agg(
                freq_media=("frequency_12m", "mean"),
                gasto_medio=("total_spent_12m", "mean"),
                recencia_media=("recency_days", "mean"),
                tenure_medio=("tenure_months", "mean"),
                ltv_medio=("ltv_estimate", "mean"),
            )
        )
        if len(seg_metrics) >= 2:
            fig_radar = go.Figure()
            categories_r = ["Frequência", "Gasto Médio", "Tenure", "LTV"]
            for _, row in seg_metrics.iterrows():
                max_gasto = seg_metrics["gasto_medio"].max() or 1
                max_ltv = seg_metrics["ltv_medio"].max() or 1
                max_tenure = seg_metrics["tenure_medio"].max() or 1
                max_freq = seg_metrics["freq_media"].max() or 1
                vals = [
                    row["freq_media"] / max_freq * 100,
                    row["gasto_medio"] / max_gasto * 100,
                    row["tenure_medio"] / max_tenure * 100,
                    row["ltv_medio"] / max_ltv * 100,
                ]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=categories_r + [categories_r[0]],
                    fill="toself", name=row["segment_hint"],
                    line=dict(color=segment_color_map.get(row["segment_hint"], "#BDC3C7")),
                ))
            styled_plotly(fig_radar, 450).update_layout(
                title="Perfil Comparativo por Segmento (Radar)",
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        beh_col1, beh_col2 = st.columns(2)
        with beh_col1:
            # Age distribution by segment
            fig_age = px.histogram(
                filtered, x="age", color="segment_hint", nbins=12,
                color_discrete_map=segment_color_map,
                title="Distribuição Etária por Segmento",
                labels={"age": "Idade", "segment_hint": "Segmento"},
                barmode="overlay", opacity=0.7,
            )
            styled_plotly(fig_age, 380)
            st.plotly_chart(fig_age, use_container_width=True)

        with beh_col2:
            # Tenure vs Spending
            fig_tenure = px.scatter(
                filtered, x="tenure_months", y="total_spent_12m",
                color="segment_hint", size="frequency_12m",
                color_discrete_map=segment_color_map,
                hover_name="customer_id",
                title="Tempo de Relacionamento × Gasto",
                labels={"tenure_months": "Tenure (meses)", "total_spent_12m": "Gasto 12M (R$)",
                        "frequency_12m": "Frequência"},
            )
            fig_tenure.update_traces(marker=dict(line=dict(width=1, color="#2C3E50")))
            styled_plotly(fig_tenure, 380)
            st.plotly_chart(fig_tenure, use_container_width=True)

        # LTV by segment box plot
        fig_ltv = px.box(
            filtered, x="segment_hint", y="ltv_estimate",
            color="segment_hint", color_discrete_map=segment_color_map,
            title="Distribuição de LTV Estimado por Segmento",
            labels={"segment_hint": "Segmento", "ltv_estimate": "LTV Estimado (R$)"},
        )
        fig_ltv.update_layout(showlegend=False)
        styled_plotly(fig_ltv, 380)
        st.plotly_chart(fig_ltv, use_container_width=True)

        # Gender breakdown
        gender_seg = (
            filtered.groupby(["gender", "segment_hint"], as_index=False)["customer_id"]
            .count().rename(columns={"customer_id": "clientes"})
        )
        fig_gender = px.sunburst(
            gender_seg, path=["gender", "segment_hint"], values="clientes",
            color_discrete_sequence=COLORS,
            title="Composição: Gênero → Segmento",
        )
        styled_plotly(fig_gender, 420)
        st.plotly_chart(fig_gender, use_container_width=True)

    with tab3:
        # RFM Bubble Chart
        st.markdown("### Matriz RFM (Recência × Frequência × Gasto)")
        fig_rfm = px.scatter(
            filtered, x="recency_days", y="frequency_12m",
            size="total_spent_12m", color="segment_hint",
            color_discrete_map=segment_color_map,
            hover_name="customer_id",
            hover_data={"avg_order_value": ":.2f", "city": True},
            labels={"recency_days": "Recência (dias)", "frequency_12m": "Frequência 12M",
                    "total_spent_12m": "Gasto 12M (R$)", "segment_hint": "Segmento"},
            title="Mapa de Clientes: Proximidade × Engajamento × Valor",
        )
        fig_rfm.update_traces(marker=dict(line=dict(width=1, color="#2C3E50"), opacity=0.85))
        fig_rfm.add_vline(x=60, line_dash="dash", line_color=COLOR_DANGER, annotation_text="Zona de Risco")
        styled_plotly(fig_rfm, 480)
        st.plotly_chart(fig_rfm, use_container_width=True)

        # Top customers table
        top_customers = filtered.sort_values("total_spent_12m", ascending=False).head(10).copy()
        display_cols = ["customer_id", "city", "segment_hint", "frequency_12m",
                        "recency_days", "avg_order_value", "total_spent_12m", "ltv_estimate"]
        top_display = top_customers[display_cols].copy()
        top_display["total_spent_12m"] = top_display["total_spent_12m"].map(format_currency)
        top_display["avg_order_value"] = top_display["avg_order_value"].map(format_currency)
        top_display["ltv_estimate"] = top_display["ltv_estimate"].map(format_currency)

        st.markdown("### Top 10 Clientes por Gasto")
        st.dataframe(top_display, width="stretch", hide_index=True)

    recommended_actions = pd.DataFrame(
        [
            {"Segmento": "champion/high_value", "Estratégia": "Programa de fidelização premium",
             "Objetivo": "Elevar retenção e ticket médio."},
            {"Segmento": "loyal", "Estratégia": "Upsell e cross-sell direcionado",
             "Objetivo": "Aumentar frequência e LTV."},
            {"Segmento": "potential", "Estratégia": "Campanha de progressão de frequência",
             "Objetivo": "Converter em clientes leais."},
            {"Segmento": "new_customer", "Estratégia": "Onboarding e incentivo à segunda compra",
             "Objetivo": "Reduzir abandono no início da jornada."},
            {"Segmento": "at_risk/churn_risk", "Estratégia": "Ação de win-back com oferta personalizada",
             "Objetivo": "Reduzir churn e recuperar receita."},
        ]
    )
    st.markdown("### Recomendação de Negócios por Segmento")
    st.dataframe(recommended_actions, width="stretch", hide_index=True)

    with st.expander("Artigo do projeto"):
        render_html_article(ARTICLE_FILES["Customer Segmentation"])
    render_author_signature()


def show_business_dashboard_page() -> None:
    st.title("Business Dashboard · Gestão Executiva")
    kpis = load_kpi_data()
    channels = load_channel_data()

    available_months = kpis["month_label"].tolist()
    selected_months = st.multiselect("Período de análise", available_months, default=available_months)

    filtered_kpis = kpis[kpis["month_label"].isin(selected_months)]
    filtered_channels = channels[channels["month_label"].isin(selected_months)]

    if filtered_kpis.empty:
        st.warning("Selecione ao menos um mês para visualizar o dashboard.")
        return

    latest = filtered_kpis.iloc[-1]
    previous = filtered_kpis.iloc[-2] if len(filtered_kpis) > 1 else None

    revenue_delta = calculate_delta(
        float(latest["revenue"]),
        float(previous["revenue"]) if previous is not None else None,
    )
    profit_delta = calculate_delta(
        float(latest["profit"]),
        float(previous["profit"]) if previous is not None else None,
    )
    churn_delta = calculate_delta(
        float(latest["churn_rate"]),
        float(previous["churn_rate"]) if previous is not None else None,
    )
    nps_delta = calculate_delta(
        float(latest["nps"]),
        float(previous["nps"]) if previous is not None else None,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Receita", format_currency(float(latest["revenue"])), revenue_delta)
    col2.metric("Lucro", format_currency(float(latest["profit"])), profit_delta)
    col3.metric("Margem", format_percent(float(latest["profit_margin"])))
    col4.metric("Churn", format_percent(float(latest["churn_rate"])), churn_delta, delta_color="inverse")
    col5.metric("NPS", f"{int(latest['nps'])}", nps_delta)

    tab1, tab2, tab3 = st.tabs(["P&L e Tendências", "Canais de Aquisição", "Análise Avançada"])

    with tab1:
        # Stacked area: Revenue, Expenses, Profit
        fig_pl = go.Figure()
        fig_pl.add_trace(go.Scatter(
            x=filtered_kpis["month_label"], y=filtered_kpis["revenue"],
            fill="tozeroy", fillcolor="rgba(46,134,193,0.2)",
            line=dict(color=COLOR_SECONDARY, width=3), name="Receita",
            hovertemplate="Receita: R$ %{y:,.0f}<extra></extra>",
        ))
        fig_pl.add_trace(go.Scatter(
            x=filtered_kpis["month_label"], y=filtered_kpis["expenses"],
            fill="tozeroy", fillcolor="rgba(231,76,60,0.12)",
            line=dict(color=COLOR_DANGER, width=2, dash="dash"), name="Despesas",
            hovertemplate="Despesas: R$ %{y:,.0f}<extra></extra>",
        ))
        fig_pl.add_trace(go.Scatter(
            x=filtered_kpis["month_label"], y=filtered_kpis["profit"],
            fill="tozeroy", fillcolor="rgba(39,174,96,0.15)",
            line=dict(color=COLOR_SUCCESS, width=3), name="Lucro",
            hovertemplate="Lucro: R$ %{y:,.0f}<extra></extra>",
        ))
        styled_plotly(fig_pl).update_layout(title="P&L — Receita, Despesas e Lucro")
        st.plotly_chart(fig_pl, use_container_width=True)

        # Dual axis: Margin + NPS
        qual_col1, qual_col2 = st.columns(2)
        with qual_col1:
            fig_margin = go.Figure()
            fig_margin.add_trace(go.Bar(
                x=filtered_kpis["month_label"],
                y=filtered_kpis["profit_margin"] * 100,
                marker_color=COLOR_PRIMARY, name="Margem (%)",
                text=filtered_kpis["profit_margin"].apply(lambda v: f"{v*100:.1f}%"),
                textposition="outside",
            ))
            fig_margin.add_trace(go.Scatter(
                x=filtered_kpis["month_label"],
                y=filtered_kpis["expense_ratio"] * 100,
                line=dict(color=COLOR_DANGER, width=2), name="Razão Despesas (%)",
                yaxis="y2",
            ))
            styled_plotly(fig_margin, 380).update_layout(
                title="Margem de Lucro & Razão de Despesas",
                yaxis=dict(title="Margem (%)"),
                yaxis2=dict(title="Despesas/Receita (%)", overlaying="y", side="right"),
            )
            st.plotly_chart(fig_margin, use_container_width=True)

        with qual_col2:
            fig_quality = go.Figure()
            fig_quality.add_trace(go.Scatter(
                x=filtered_kpis["month_label"], y=filtered_kpis["nps"],
                line=dict(color=COLOR_SECONDARY, width=3), name="NPS",
                mode="lines+markers", marker=dict(size=10),
            ))
            fig_quality.add_trace(go.Scatter(
                x=filtered_kpis["month_label"],
                y=filtered_kpis["churn_rate"] * 100,
                line=dict(color=COLOR_DANGER, width=2, dash="dot"), name="Churn (%)",
                yaxis="y2", mode="lines+markers", marker=dict(size=8),
            ))
            styled_plotly(fig_quality, 380).update_layout(
                title="NPS & Churn Rate",
                yaxis=dict(title="NPS"),
                yaxis2=dict(title="Churn (%)", overlaying="y", side="right"),
            )
            st.plotly_chart(fig_quality, use_container_width=True)

    with tab2:
        channel_summary = (
            filtered_channels.groupby("channel", as_index=False)
            .agg(
                receita_total=("revenue", "sum"),
                sessions_total=("sessions", "sum"),
                conversions_total=("conversions", "sum"),
                cac_medio=("cac", "mean"),
                taxa_conversao_media=("conversion_rate", "mean"),
                roi_proxy_medio=("roi_proxy", "mean"),
            )
            .sort_values("receita_total", ascending=False)
        )

        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            # Revenue by channel bar
            fig_ch_rev = go.Figure(go.Bar(
                x=channel_summary["channel"], y=channel_summary["receita_total"],
                marker_color=[COLOR_PRIMARY, COLOR_SECONDARY, COLOR_WARNING],
                text=channel_summary["receita_total"].apply(lambda v: format_currency(v)),
                textposition="outside",
            ))
            styled_plotly(fig_ch_rev, 380).update_layout(title="Receita Total por Canal")
            st.plotly_chart(fig_ch_rev, use_container_width=True)

        with ch_col2:
            # CAC vs Conversion scatter
            fig_cac = px.scatter(
                channel_summary, x="cac_medio", y="taxa_conversao_media",
                size="receita_total", color="channel",
                color_discrete_sequence=COLORS,
                hover_data={"roi_proxy_medio": ":.2f"},
                labels={"cac_medio": "CAC Médio (R$)", "taxa_conversao_media": "Taxa de Conversão",
                        "receita_total": "Receita Total"},
                title="CAC × Conversão × Receita",
            )
            fig_cac.update_traces(marker=dict(line=dict(width=1, color="#2C3E50")))
            styled_plotly(fig_cac, 380).update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_cac, use_container_width=True)

        # Funnel analysis
        st.markdown("### Funil de Conversão por Canal")
        for _, row in channel_summary.iterrows():
            fig_funnel = go.Figure(go.Funnel(
                y=["Sessões", "Conversões"],
                x=[row["sessions_total"], row["conversions_total"]],
                textinfo="value+percent initial",
                marker_color=[COLOR_SECONDARY, COLOR_SUCCESS],
            ))
            styled_plotly(fig_funnel, 200).update_layout(
                title=f"Canal: {row['channel']}",
                margin=dict(l=30, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_funnel, use_container_width=True)

        # Channel evolution over time
        fig_ch_trend = px.line(
            filtered_channels, x="month_label", y="revenue",
            color="channel", markers=True,
            color_discrete_sequence=COLORS,
            labels={"month_label": "Mês", "revenue": "Receita (R$)", "channel": "Canal"},
            title="Evolução da Receita por Canal",
        )
        fig_ch_trend.update_traces(line=dict(width=3))
        styled_plotly(fig_ch_trend, 380)
        st.plotly_chart(fig_ch_trend, use_container_width=True)

        # Summary table
        display_channel = channel_summary.copy()
        display_channel["receita_total"] = display_channel["receita_total"].map(format_currency)
        display_channel["cac_medio"] = display_channel["cac_medio"].map(format_currency)
        display_channel["taxa_conversao_media"] = display_channel["taxa_conversao_media"].map(format_percent)
        display_channel["roi_proxy_medio"] = display_channel["roi_proxy_medio"].map(lambda v: f"{v:.2f}x")
        st.markdown("### Resumo Executivo por Canal")
        st.dataframe(display_channel, width="stretch", hide_index=True)

    with tab3:
        # New customers trend
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            fig_new_cust = go.Figure()
            fig_new_cust.add_trace(go.Bar(
                x=filtered_kpis["month_label"], y=filtered_kpis["new_customers"],
                marker_color=COLOR_SECONDARY, name="Novos Clientes",
                text=filtered_kpis["new_customers"].astype(int).astype(str),
                textposition="outside",
            ))
            styled_plotly(fig_new_cust, 380).update_layout(title="Aquisição de Novos Clientes")
            st.plotly_chart(fig_new_cust, use_container_width=True)

        with adv_col2:
            # Revenue per new customer
            rev_per_cust = filtered_kpis.copy()
            rev_per_cust["rev_per_new"] = rev_per_cust["revenue"] / rev_per_cust["new_customers"]
            fig_rpc = go.Figure(go.Scatter(
                x=rev_per_cust["month_label"], y=rev_per_cust["rev_per_new"],
                mode="lines+markers+text",
                text=rev_per_cust["rev_per_new"].apply(lambda v: format_currency(v)),
                textposition="top center",
                line=dict(color=COLOR_PRIMARY, width=3),
                marker=dict(size=10),
            ))
            styled_plotly(fig_rpc, 380).update_layout(
                title="Receita por Novo Cliente",
                yaxis_title="R$/cliente",
            )
            st.plotly_chart(fig_rpc, use_container_width=True)

        # Waterfall P&L
        fig_wf_dash = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Receita", "Despesas", "Lucro"],
            y=[float(latest["revenue"]), -float(latest["expenses"]), float(latest["profit"])],
            text=[format_currency(float(latest["revenue"])),
                  format_currency(-float(latest["expenses"])),
                  format_currency(float(latest["profit"]))],
            textposition="outside",
            connector=dict(line=dict(color="#CBD5E0")),
            increasing=dict(marker=dict(color=COLOR_SUCCESS)),
            decreasing=dict(marker=dict(color=COLOR_DANGER)),
            totals=dict(marker=dict(color=COLOR_PRIMARY)),
        ))
        styled_plotly(fig_wf_dash, 380).update_layout(title="Decomposição P&L (Mês Atual)")
        st.plotly_chart(fig_wf_dash, use_container_width=True)

    st.markdown("### Recomendações Gerenciais")
    best_channel = channel_summary.iloc[0] if not filtered_channels.empty else None
    if best_channel is not None:
        highest_cac_channel = channel_summary.sort_values("cac_medio", ascending=False).iloc[0]
        best_roi_channel = channel_summary.sort_values("roi_proxy_medio", ascending=False).iloc[0]
        st.markdown(
            f"""
- Canal com maior receita: **{best_channel['channel']}**.
- Canal com maior CAC: **{highest_cac_channel['channel']}** (R$ {highest_cac_channel['cac_medio']:.2f}/conversão).
- Melhor ROI proxy: **{best_roi_channel['channel']}** ({best_roi_channel['roi_proxy_medio']:.2f}x).
- Priorizar rebalanceamento de orçamento com base no ROI por canal.
            """
        )

    with st.expander("Artigo do projeto"):
        render_html_article(ARTICLE_FILES["Business Dashboard"])
    render_author_signature()


def show_etl_page() -> None:
    st.title("ETL Pipeline · Governança de Dados")
    st.markdown(
        """
Pipeline de dados para consolidar fontes transacionais e de clientes com padrão analítico reutilizável.
        """
    )

    st.markdown("### Execução")
    st.code(
        "pip install -r requirements.txt\npython etl_pipeline_project/scripts/etl_pipeline.py",
        language="bash",
    )

    quality_rows: list[dict[str, str | int]] = []
    if SALES_FILE.exists():
        sales_rows = len(pd.read_csv(SALES_FILE))
        quality_rows.append({"dataset": "sales_transactions.csv", "linhas": sales_rows, "status": "fonte"})
    if CUSTOMER_FILE.exists():
        customer_rows = len(pd.read_csv(CUSTOMER_FILE))
        quality_rows.append({"dataset": "customer_behavior.csv", "linhas": customer_rows, "status": "fonte"})

    output_files = ["sales_enriched.csv", "monthly_metrics.csv", "customer_metrics.csv"]
    for output_file in output_files:
        output_path = ETL_OUTPUT_DIR / output_file
        if output_path.exists():
            output_rows = len(pd.read_csv(output_path))
            quality_rows.append({"dataset": output_file, "linhas": output_rows, "status": "output"})
        else:
            quality_rows.append({"dataset": output_file, "linhas": 0, "status": "output pendente"})

    st.markdown("### Controle de artefatos")
    st.dataframe(pd.DataFrame(quality_rows), width="stretch", hide_index=True)

    if PIPELINE_CONFIG.exists():
        with st.expander("Configuração do pipeline"):
            st.code(PIPELINE_CONFIG.read_text(encoding="utf-8"), language="yaml")

    if ETL_SCRIPT.exists():
        with st.expander("Script ETL"):
            st.code(ETL_SCRIPT.read_text(encoding="utf-8"), language="python")

    st.markdown("### Artigo do projeto")
    render_html_article(ARTICLE_FILES["ETL Pipeline"])
    render_author_signature()


def show_articles_page() -> None:
    st.title("Artigos Executivos em HTML")
    st.write("Selecione um projeto para leitura detalhada do contexto, abordagem e recomendações.")

    article_name = st.selectbox("Projeto", list(ARTICLE_FILES.keys()))
    render_html_article(ARTICLE_FILES[article_name], height=1100)
    render_author_signature()


def main() -> None:
    st.set_page_config(
        page_title="Data Analytics Portfolio",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_ui_style()

    menu = st.sidebar.selectbox(
        "Navegação",
        [
            "🏠 Home",
            "📊 Sales Analysis",
            "👥 Customer Segmentation",
            "📈 Business Dashboard",
            "⚙️ ETL Pipeline",
            "📚 Artigos",
            "👤 Sobre"
        ]
    )

    if menu == "🏠 Home":

        st.title("📊 Data Analytics Portfolio")

        st.markdown(
        """
        Plataforma de demonstração de projetos de **Data Analytics e Business Intelligence**.

        Este portfólio apresenta projetos práticos envolvendo:

        • Análise de dados de vendas  
        • Segmentação de clientes  
        • Dashboards de indicadores de negócio  
        • Pipeline de engenharia de dados  

        ### Tecnologias utilizadas

        - Python
        - Pandas
        - SQL
        - Plotly
        - Streamlit
        - ETL Pipelines

        ### Objetivo

        Demonstrar aplicações reais de **análise de dados para tomada de decisão empresarial**.
        """
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric("Projetos Analíticos", "4")
        col2.metric("Dashboards Criados", "6+")
        col3.metric("Pipelines de Dados", "1")

        st.divider()

        st.subheader("Projetos Disponíveis")

        st.markdown(
        """
        **Sales Analysis**  
        Análise de transações de vendas para identificar padrões de receita.

        **Customer Segmentation**  
        Segmentação de clientes baseada em comportamento de compra.

        **Business Dashboard**  
        Dashboard executivo com indicadores financeiros.

        **ETL Pipeline**  
        Pipeline de dados automatizado para processamento analítico.
        """
        )

    if menu == "📊 Sales Analysis":

        st.title("📊 Sales Performance Analysis")

        df = load_sales_model()

        st.subheader("Visão Geral")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Receita Total",
            format_currency(df["net_revenue"].sum())
        )

        col2.metric(
            "Pedidos",
            df.shape[0]
        )

        col3.metric(
            "Margem Média",
            format_percent(df["margin_pct"].mean())
        )

        col4.metric(
            "Ticket Médio",
            format_currency(df["net_revenue"].mean())
        )

        st.divider()

        st.subheader("Receita por Categoria")

        category_sales = (
            df.groupby("category")["net_revenue"]
            .sum()
            .reset_index()
            .sort_values("net_revenue", ascending=False)
        )

        fig = px.bar(
            category_sales,
            x="category",
            y="net_revenue",
            color="category",
            title="Faturamento por Categoria",
            color_discrete_sequence=COLORS
        )

        st.plotly_chart(styled_plotly(fig), use_container_width=True)

    if menu == "👥 Customer Segmentation":

        st.title("👥 Customer Segmentation")

        df = load_customer_data()

        st.subheader("Distribuição de Clientes")

        fig = px.histogram(
            df,
            x="avg_order_value",
            nbins=30,
            title="Distribuição do Ticket Médio"
        )

        st.plotly_chart(styled_plotly(fig), use_container_width=True)

        st.subheader("Valor Estimado do Cliente (LTV)")

        fig2 = px.scatter(
            df,
            x="frequency_12m",
            y="ltv_estimate",
            color="tenure_months",
            title="Lifetime Value por Frequência"
        )

        st.plotly_chart(styled_plotly(fig2), use_container_width=True)


if __name__ == "__main__":
    main()
