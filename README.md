# Data Analytics Portfolio

Portfólio com 4 projetos de análise de dados, BI e engenharia de dados.

## Autoria

- **Autora:** Sara Oliveira Guimarães Nascimento 
- **Cargo:** Analista de Negócios
- **Empresa:** Dalumia Consultoria ME

## Estrutura

- `sales_analysis_project/` — análise de vendas com SQL, dados transacionais e guia de dashboard.
- `customer_segmentation/` — segmentação de clientes com dataset pronto para modelagem.
- `business_dashboard/` — base de KPIs para construção de dashboard no Power BI.
- `etl_pipeline_project/` — pipeline ETL em Python para consolidar dados.

## Fonte dos dados

Este repositório já vem com **dados fictícios** para estudo e portfólio.

Se quiser usar Kaggle, você pode substituir os CSVs mantendo os mesmos nomes de colunas. Sugestões:

- Superstore Sales Dataset
- Online Retail Dataset
- Customer Segmentation Dataset

## Como começar

1. Crie e ative um ambiente virtual Python.
1. Instale dependências:

```bash
pip install -r requirements.txt
```

1. Execute o ETL:

```bash
python etl_pipeline_project/scripts/etl_pipeline.py
```

1. Use os arquivos gerados em `etl_pipeline_project/pipeline/output/` para notebooks e dashboards.

## Modo Streamlit (portfólio interativo)

Execute a aplicação:

```bash
streamlit run streamlit_app.py
```

O app inclui:

- Navegação por projeto com métricas e tabelas.
- Leitura de dados reais dos CSVs do repositório.
- Artigos em HTML descrevendo cada projeto e habilidades aplicadas.

## Próximos passos sugeridos

- Criar notebooks exploratórios nas pastas `notebooks/`.
- Construir dashboard no Power BI em `business_dashboard/powerbi/`.
- Adicionar métricas avançadas (LTV, cohort, churn por segmento).

