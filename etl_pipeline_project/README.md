# ETL Pipeline Project

Pipeline de extração e transformação para consolidar dados de vendas e clientes.

## Estrutura

- `scripts/etl_pipeline.py` — pipeline principal
- `scripts/requirements.txt` — dependências
- `pipeline/pipeline_config.yaml` — configuração da execução

## Como rodar

```bash
pip install -r etl_pipeline_project/scripts/requirements.txt
python etl_pipeline_project/scripts/etl_pipeline.py
```

## Saídas

Os arquivos são gravados em `etl_pipeline_project/pipeline/output/`:

- `sales_enriched.csv`
- `monthly_metrics.csv`
- `customer_metrics.csv`
