from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


AUTHOR_NAME: Final = "Sara Oliveira Guimarães Nascimento"
AUTHOR_ROLE: Final = "Analista de Negócios"
AUTHOR_COMPANY: Final = "Dalumia Consultoria ME"

BASE_DIR = Path(__file__).resolve().parents[2]
SALES_FILE = BASE_DIR / "sales_analysis_project" / "data" / "sales_transactions.csv"
CUSTOMER_FILE = BASE_DIR / "customer_segmentation" / "dataset" / "customer_behavior.csv"
OUTPUT_DIR = BASE_DIR / "etl_pipeline_project" / "pipeline" / "output"

SALES_REQUIRED_COLUMNS: Final[set[str]] = {
    "transaction_id",
    "date",
    "customer_id",
    "quantity",
    "unit_price",
    "discount_pct",
}
CUSTOMER_REQUIRED_COLUMNS: Final[set[str]] = {"customer_id", "recency_days", "frequency_12m", "total_spent_12m"}
DIAGNOSTIC_OUTPUT = OUTPUT_DIR / "diagnostic_report.csv"

LOGGER = logging.getLogger("etl_pipeline")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def validate_columns(dataframe: pd.DataFrame, required_columns: set[str], dataset_name: str) -> None:
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            f"Dataset '{dataset_name}' inválido. Colunas ausentes: {', '.join(missing_columns)}"
        )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Carregando fontes de dados")
    sales_df = pd.read_csv(SALES_FILE)
    customer_df = pd.read_csv(CUSTOMER_FILE)

    validate_columns(sales_df, SALES_REQUIRED_COLUMNS, SALES_FILE.name)
    validate_columns(customer_df, CUSTOMER_REQUIRED_COLUMNS, CUSTOMER_FILE.name)

    LOGGER.info("Fontes carregadas: sales=%s linhas | customers=%s linhas", len(sales_df), len(customer_df))
    return sales_df, customer_df


# ---------------------------------------------------------------------------
# Limpeza e tratamento de dados
# ---------------------------------------------------------------------------

def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza e tratamento do dataset de vendas."""
    LOGGER.info("Iniciando limpeza de vendas (%s linhas)", len(df))
    cleaned = df.copy()

    # Remover espaços em branco de colunas texto
    for col in cleaned.select_dtypes(include="object").columns:
        cleaned[col] = cleaned[col].str.strip()

    # Converter data e remover registros com data inválida
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    invalid_dates = cleaned["date"].isna().sum()
    if invalid_dates:
        LOGGER.warning("Removidas %s linhas com datas inválidas", invalid_dates)
    cleaned = cleaned.dropna(subset=["date"])

    # Garantir tipos numéricos
    cleaned["quantity"] = pd.to_numeric(cleaned["quantity"], errors="coerce")
    cleaned["unit_price"] = pd.to_numeric(cleaned["unit_price"], errors="coerce")
    cleaned["discount_pct"] = pd.to_numeric(cleaned["discount_pct"], errors="coerce").fillna(0.0)

    # Regras de negócio: valores devem ser positivos e desconto entre 0 e 1
    rows_before = len(cleaned)
    cleaned = cleaned[
        (cleaned["quantity"] > 0)
        & (cleaned["unit_price"] > 0)
        & (cleaned["discount_pct"] >= 0)
        & (cleaned["discount_pct"] <= 1)
    ]
    removed = rows_before - len(cleaned)
    if removed:
        LOGGER.warning("Removidas %s linhas com valores fora das regras de negócio", removed)

    # Remover duplicatas por transaction_id
    dupl = cleaned.duplicated(subset=["transaction_id"]).sum()
    if dupl:
        LOGGER.warning("Removidas %s transações duplicadas", dupl)
        cleaned = cleaned.drop_duplicates(subset=["transaction_id"], keep="first")

    # Padronizar payment_method
    if "payment_method" in cleaned.columns:
        cleaned["payment_method"] = cleaned["payment_method"].str.lower().str.replace(" ", "_")

    LOGGER.info("Limpeza de vendas concluída: %s linhas retidas", len(cleaned))
    return cleaned.reset_index(drop=True)


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza e tratamento do dataset de clientes."""
    LOGGER.info("Iniciando limpeza de clientes (%s linhas)", len(df))
    cleaned = df.copy()

    # Remover espaços em branco de colunas texto
    for col in cleaned.select_dtypes(include="object").columns:
        cleaned[col] = cleaned[col].str.strip()

    # Garantir tipos numéricos
    for col in ["age", "tenure_months", "recency_days", "frequency_12m"]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    for col in ["avg_order_value", "total_spent_12m"]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Remover linhas com campos-chave nulos
    key_cols = ["customer_id", "recency_days", "frequency_12m", "total_spent_12m"]
    present = [c for c in key_cols if c in cleaned.columns]
    nulls_before = cleaned[present].isna().any(axis=1).sum()
    if nulls_before:
        LOGGER.warning("Removidas %s linhas com campos-chave nulos", nulls_before)
    cleaned = cleaned.dropna(subset=present)

    # Regras de negócio
    rows_before = len(cleaned)
    if "age" in cleaned.columns:
        cleaned = cleaned[(cleaned["age"] >= 18) & (cleaned["age"] <= 120)]
    if "frequency_12m" in cleaned.columns:
        cleaned = cleaned[cleaned["frequency_12m"] >= 0]
    if "total_spent_12m" in cleaned.columns:
        cleaned = cleaned[cleaned["total_spent_12m"] >= 0]
    removed = rows_before - len(cleaned)
    if removed:
        LOGGER.warning("Removidas %s linhas fora das regras de negócio", removed)

    # Remover duplicatas por customer_id
    dupl = cleaned.duplicated(subset=["customer_id"]).sum()
    if dupl:
        LOGGER.warning("Removidas %s clientes duplicados", dupl)
        cleaned = cleaned.drop_duplicates(subset=["customer_id"], keep="first")

    # Padronizar gênero e canal preferido
    if "gender" in cleaned.columns:
        cleaned["gender"] = cleaned["gender"].str.upper()
    if "preferred_channel" in cleaned.columns:
        cleaned["preferred_channel"] = cleaned["preferred_channel"].str.lower().str.replace(" ", "_")

    LOGGER.info("Limpeza de clientes concluída: %s linhas retidas", len(cleaned))
    return cleaned.reset_index(drop=True)


def transform_sales(sales_df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Aplicando transformações de vendas")
    transformed = sales_df.copy()

    transformed["gross_revenue"] = transformed["quantity"] * transformed["unit_price"]
    transformed["net_revenue"] = transformed["gross_revenue"] * (1 - transformed["discount_pct"])
    transformed["discount_value"] = transformed["gross_revenue"] - transformed["net_revenue"]
    transformed["month"] = transformed["date"].dt.to_period("M").astype(str)

    numeric_columns = ["gross_revenue", "net_revenue", "discount_value"]
    transformed[numeric_columns] = transformed[numeric_columns].round(2)
    return transformed


def build_monthly_metrics(sales_df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Consolidando métricas mensais")
    monthly_metrics = (
        sales_df.groupby("month", as_index=False)
        .agg(
            total_transactions=("transaction_id", "count"),
            total_units=("quantity", "sum"),
            gross_revenue=("gross_revenue", "sum"),
            net_revenue=("net_revenue", "sum"),
            total_discount=("discount_value", "sum"),
            active_customers=("customer_id", "nunique"),
        )
        .sort_values("month")
    )
    return monthly_metrics.round(2)


def build_customer_metrics(sales_df: pd.DataFrame, customer_df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Consolidando métricas por cliente")
    reference_date = sales_df["date"].max() + pd.Timedelta(days=1)

    customer_metrics = (
        sales_df.groupby("customer_id", as_index=False)
        .agg(
            frequency=("transaction_id", "count"),
            monetary=("net_revenue", "sum"),
            last_purchase=("date", "max"),
        )
        .sort_values("monetary", ascending=False)
    )

    customer_metrics["recency"] = (reference_date - customer_metrics["last_purchase"]).dt.days
    customer_metrics = customer_metrics.drop(columns=["last_purchase"])

    merged = customer_metrics.merge(customer_df, on="customer_id", how="left")
    merged["data_completeness"] = merged.notna().all(axis=1).map({True: "complete", False: "partial"})
    return merged.round(2)


# ---------------------------------------------------------------------------
# Análise diagnóstica
# ---------------------------------------------------------------------------

def diagnostic_analysis(
    sales_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    customer_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Gera análise diagnóstica com indicadores e possíveis causas."""
    LOGGER.info("Gerando análise diagnóstica")
    diagnostics: list[dict[str, object]] = []

    # 1. Variação de receita mês a mês (MoM)
    if len(monthly_df) >= 2:
        monthly_sorted = monthly_df.sort_values("month")
        monthly_sorted["revenue_mom_pct"] = (
            monthly_sorted["net_revenue"].pct_change() * 100
        ).round(2)
        for _, row in monthly_sorted.dropna(subset=["revenue_mom_pct"]).iterrows():
            direction = "crescimento" if row["revenue_mom_pct"] > 0 else "queda"
            diagnostics.append({
                "categoria": "Variação Receita MoM",
                "indicador": f"{row['month']}",
                "valor": f"{row['revenue_mom_pct']:+.2f}%",
                "diagnostico": (
                    f"{direction.title()} de {abs(row['revenue_mom_pct']):.1f}% na receita líquida. "
                    f"Transações: {int(row['total_transactions'])} | Clientes ativos: {int(row['active_customers'])}"
                ),
            })

    # 2. Impacto de descontos
    total_gross = sales_df["gross_revenue"].sum()
    total_discount = sales_df["discount_value"].sum()
    discount_share = (total_discount / total_gross * 100) if total_gross else 0
    diagnostics.append({
        "categoria": "Impacto de Descontos",
        "indicador": "Desconto / Receita Bruta",
        "valor": f"{discount_share:.2f}%",
        "diagnostico": (
            f"R$ {total_discount:,.2f} concedidos em descontos sobre R$ {total_gross:,.2f} bruto. "
            + ("Nível de desconto elevado — avaliar política de descontos." if discount_share > 10 else "Nível de desconto dentro do esperado.")
        ),
    })

    # 3. Concentração de receita por cliente (top 20%)
    if len(customer_metrics) > 0 and "monetary" in customer_metrics.columns:
        sorted_cust = customer_metrics.sort_values("monetary", ascending=False)
        top_n = max(1, int(len(sorted_cust) * 0.2))
        top_revenue = sorted_cust.head(top_n)["monetary"].sum()
        total_revenue = sorted_cust["monetary"].sum()
        concentration = (top_revenue / total_revenue * 100) if total_revenue else 0
        diagnostics.append({
            "categoria": "Concentração de Clientes",
            "indicador": f"Top {top_n} clientes (20%)",
            "valor": f"{concentration:.1f}% da receita",
            "diagnostico": (
                f"Os {top_n} maiores clientes representam {concentration:.1f}% da receita total. "
                + ("Alta concentração — risco de dependência." if concentration > 60 else "Concentração saudável.")
            ),
        })

    # 4. Ticket médio e dispersão
    avg_ticket = sales_df["net_revenue"].mean()
    std_ticket = sales_df["net_revenue"].std()
    cv_ticket = (std_ticket / avg_ticket * 100) if avg_ticket else 0
    diagnostics.append({
        "categoria": "Ticket Médio",
        "indicador": "Média / Desvio Padrão",
        "valor": f"R$ {avg_ticket:,.2f} ± R$ {std_ticket:,.2f}",
        "diagnostico": (
            f"Coeficiente de variação de {cv_ticket:.1f}%. "
            + ("Alta dispersão — mix de produtos muito heterogêneo." if cv_ticket > 80 else "Dispersão moderada no valor das transações.")
        ),
    })

    # 5. Frequência de compra: clientes inativos vs ativos
    if "recency" in customer_metrics.columns:
        inactive = (customer_metrics["recency"] > 60).sum()
        active = (customer_metrics["recency"] <= 60).sum()
        total_cust = len(customer_metrics)
        inactive_pct = (inactive / total_cust * 100) if total_cust else 0
        diagnostics.append({
            "categoria": "Atividade de Clientes",
            "indicador": "Inativos (recência > 60 dias)",
            "valor": f"{inactive} de {total_cust} ({inactive_pct:.1f}%)",
            "diagnostico": (
                f"{active} clientes ativos vs {inactive} inativos. "
                + ("Alto índice de inatividade — avaliar campanha de reativação." if inactive_pct > 30 else "Nível de inatividade controlado.")
            ),
        })

    # 6. Método de pagamento predominante
    if "payment_method" in sales_df.columns:
        pay_dist = sales_df["payment_method"].value_counts(normalize=True) * 100
        top_method = pay_dist.index[0]
        top_pct = pay_dist.iloc[0]
        diagnostics.append({
            "categoria": "Método de Pagamento",
            "indicador": f"Mais utilizado: {top_method}",
            "valor": f"{top_pct:.1f}%",
            "diagnostico": (
                f"{top_method} representa {top_pct:.1f}% das transações. "
                f"Distribuição: {', '.join(f'{m}={v:.0f}%' for m, v in pay_dist.items())}"
            ),
        })

    # 7. Performance por loja
    if "store_id" in sales_df.columns:
        store_perf = (
            sales_df.groupby("store_id", as_index=False)
            .agg(receita=( "net_revenue", "sum"), transacoes=("transaction_id", "count"))
            .sort_values("receita", ascending=False)
        )
        best = store_perf.iloc[0]
        worst = store_perf.iloc[-1]
        diagnostics.append({
            "categoria": "Performance por Loja",
            "indicador": f"Melhor: {best['store_id']} | Pior: {worst['store_id']}",
            "valor": f"R$ {best['receita']:,.2f} vs R$ {worst['receita']:,.2f}",
            "diagnostico": (
                f"Diferença de R$ {best['receita'] - worst['receita']:,.2f} entre melhor e pior loja. "
                f"Transações: {int(best['transacoes'])} vs {int(worst['transacoes'])}."
            ),
        })

    report = pd.DataFrame(diagnostics)
    LOGGER.info("Análise diagnóstica concluída: %s indicadores gerados", len(report))
    return report


def save_outputs(
    sales_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    customer_df: pd.DataFrame,
    diagnostic_df: pd.DataFrame,
) -> dict[str, Path]:
    LOGGER.info("Gravando saídas do pipeline")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "sales_enriched": OUTPUT_DIR / "sales_enriched.csv",
        "monthly_metrics": OUTPUT_DIR / "monthly_metrics.csv",
        "customer_metrics": OUTPUT_DIR / "customer_metrics.csv",
        "diagnostic_report": DIAGNOSTIC_OUTPUT,
    }

    sales_df.to_csv(output_paths["sales_enriched"], index=False)
    monthly_df.to_csv(output_paths["monthly_metrics"], index=False)
    customer_df.to_csv(output_paths["customer_metrics"], index=False)
    diagnostic_df.to_csv(output_paths["diagnostic_report"], index=False)
    return output_paths


def run_pipeline() -> dict[str, Path]:
    # Extract
    sales_raw, customer_raw = load_data()

    # Limpeza e tratamento
    sales_clean = clean_sales(sales_raw)
    customer_clean = clean_customers(customer_raw)

    # Transform
    sales_enriched = transform_sales(sales_clean)
    monthly_metrics = build_monthly_metrics(sales_enriched)
    customer_metrics = build_customer_metrics(sales_enriched, customer_clean)

    # Análise diagnóstica
    diagnostic_df = diagnostic_analysis(sales_enriched, monthly_metrics, customer_metrics)

    # Load
    return save_outputs(sales_enriched, monthly_metrics, customer_metrics, diagnostic_df)


def main() -> int:
    configure_logging()
    LOGGER.info("Iniciando ETL | %s | %s | %s", AUTHOR_NAME, AUTHOR_ROLE, AUTHOR_COMPANY)
    try:
        output_paths = run_pipeline()
    except Exception:
        LOGGER.exception("Falha na execução do pipeline ETL")
        return 1

    LOGGER.info("Pipeline concluído com sucesso")
    for output_name, output_path in output_paths.items():
        LOGGER.info("Output %s => %s", output_name, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
