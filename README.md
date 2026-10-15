# sql-latency-ml

Estima latência aproximada de SQL com regressão sobre estimativas e características estruturais da consulta.

## Dados

O pipeline gera dados sintéticos determinísticos localmente, identificados como sintéticos em `pipeline.py`. Nenhum dado privado ou download externo é necessário. A divisão e a semente de avaliação são descritas junto às métricas abaixo.

## Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Métricas obtidas

Seed 42: holdout aleatório de 450 consultas sintéticas — MAE **17,1671 ms**, RMSE **21,8282 ms**, R² **0,9773**. A latência foi gerada por uma fórmula conhecida sobre linhas estimadas e estrutura SQL, por isso o resultado não representa um mecanismo de banco real. Sem `estimated_rows` no input ou comentário SQL, a extração usa 1.000 como valor padrão.

Limitação: dados e regras de geração simplificados; antes de uso real, substitua-os por dados com licença e proveniência adequadas e valide fora da amostra.
