from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

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


def transform_sales(sales_df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Aplicando transformações de vendas")
    transformed = sales_df.copy()
    transformed["date"] = pd.to_datetime(transformed["date"], errors="coerce")
    transformed = transformed.dropna(subset=["date"])

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


def save_outputs(
    sales_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    customer_df: pd.DataFrame,
) -> dict[str, Path]:
    LOGGER.info("Gravando saídas do pipeline")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "sales_enriched": OUTPUT_DIR / "sales_enriched.csv",
        "monthly_metrics": OUTPUT_DIR / "monthly_metrics.csv",
        "customer_metrics": OUTPUT_DIR / "customer_metrics.csv",
    }

    sales_df.to_csv(output_paths["sales_enriched"], index=False)
    monthly_df.to_csv(output_paths["monthly_metrics"], index=False)
    customer_df.to_csv(output_paths["customer_metrics"], index=False)
    return output_paths


def run_pipeline() -> dict[str, Path]:
    sales_raw, customer_raw = load_data()
    sales_enriched = transform_sales(sales_raw)
    monthly_metrics = build_monthly_metrics(sales_enriched)
    customer_metrics = build_customer_metrics(sales_enriched, customer_raw)
    return save_outputs(sales_enriched, monthly_metrics, customer_metrics)


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
