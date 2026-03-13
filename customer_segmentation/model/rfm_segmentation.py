from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd


AUTHOR_NAME: Final = "Sara Oliveira Guimarães Nasciemnto"
AUTHOR_ROLE: Final = "Analista de Negócios"
AUTHOR_COMPANY: Final = "Dalumia Consultoria ME"

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "customer_segmentation" / "dataset" / "customer_behavior.csv"
OUTPUT_FILE = BASE_DIR / "customer_segmentation" / "model" / "rfm_segments.csv"

REQUIRED_COLUMNS: Final[set[str]] = {
    "customer_id",
    "recency_days",
    "frequency_12m",
    "total_spent_12m",
    "preferred_channel",
}

LOGGER = logging.getLogger("rfm_segmentation")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def validate_columns(dataframe: pd.DataFrame) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Dataset inválido. Colunas ausentes: {', '.join(missing_columns)}")


def score_series(series: pd.Series, reverse: bool = False) -> pd.Series:
    ranked = series.rank(method="first")
    bins = pd.qcut(ranked, q=4, labels=[1, 2, 3, 4]).astype(int)
    return 5 - bins if reverse else bins


def assign_segment(score: int) -> str:
    if score >= 10:
        return "champions"
    if score >= 8:
        return "loyal"
    if score >= 6:
        return "potential"
    return "at_risk"


def assign_recommended_action(segment: str) -> str:
    segment_actions = {
        "champions": "Ofertas exclusivas e programa de embaixadores",
        "loyal": "Cross-sell com produtos complementares",
        "potential": "Campanhas de aumento de frequência",
        "at_risk": "Ações de reativação com incentivo direcionado",
    }
    return segment_actions.get(segment, "Revisar estratégia")


def run() -> Path:
    LOGGER.info("Iniciando segmentação RFM | %s | %s | %s", AUTHOR_NAME, AUTHOR_ROLE, AUTHOR_COMPANY)
    dataframe = pd.read_csv(INPUT_FILE)
    validate_columns(dataframe)

    dataframe["r_score"] = score_series(dataframe["recency_days"], reverse=True)
    dataframe["f_score"] = score_series(dataframe["frequency_12m"])
    dataframe["m_score"] = score_series(dataframe["total_spent_12m"])
    dataframe["rfm_score"] = dataframe[["r_score", "f_score", "m_score"]].sum(axis=1)
    dataframe["segment"] = dataframe["rfm_score"].apply(assign_segment)
    dataframe["recommended_action"] = dataframe["segment"].apply(assign_recommended_action)

    output_df = dataframe.sort_values(["segment", "rfm_score", "total_spent_12m"], ascending=[True, False, False])
    output_df.to_csv(OUTPUT_FILE, index=False)
    LOGGER.info("Segmentação gerada em: %s", OUTPUT_FILE)
    return OUTPUT_FILE


def main() -> int:
    configure_logging()
    try:
        run()
    except Exception:
        LOGGER.exception("Falha ao gerar segmentação RFM")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
