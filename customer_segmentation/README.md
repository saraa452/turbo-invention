# Customer Segmentation

Projeto focado em segmentação de clientes com base em comportamento de compra.

## Pastas

- `dataset/` — dados de clientes.
- `model/` — scripts de modelagem e saídas de segmentação.
- `notebooks/` — análises e validação de segmentos.

## Execução da segmentação RFM

```bash
python customer_segmentation/model/rfm_segmentation.py
```

Saída esperada:

- `customer_segmentation/model/rfm_segments.csv`

## Opcional: usar Kaggle

Substitua `dataset/customer_behavior.csv` por um dataset de clientes do Kaggle mantendo as colunas principais de frequência, recência e gasto.
