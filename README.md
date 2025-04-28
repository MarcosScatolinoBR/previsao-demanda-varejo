# Previsão de Demanda para Varejo

Projeto de análise de dados e previsão de vendas, focado no setor de varejo, utilizando técnicas de Machine Learning e boas práticas de engenharia de dados.

O objetivo é construir um pipeline robusto que percorre todas as etapas de um projeto real: desde o pré-processamento dos dados, análise exploratória, engenharia de features, modelagem preditiva, até a geração de previsões automáticas.

---

## 📂 Estrutura de Pastas

```
previsao_demanda_retail/
├── data/               # Bases de dados de entrada
│   └─ retail_dataset.csv
├── models/             # Modelos treinados (.pkl)
│   └─ model_random_forest.pkl
├── outputs/            # Arquivos de previsões geradas
│   └─ previsoes_vendas.csv
├── src/                # Scripts do projeto
    ├─ preprocessing.py
    ├─ feature_engineering.py
    ├─ eda.py
    ├─ modeling.py
    ├─ predict.py
    └─ main.py
```

---

## 🚀 Tecnologias Utilizadas

- Python 3.12
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Holidays (feriados nacionais)

---

## 🔎 Etapas do Projeto

- **Pré-processamento**: limpeza e padronização dos dados
- **Análise exploratória**: identificação de padrões de vendas e comportamento de mercado
- **Engenharia de features**: criação de variáveis de calendário e tendência
- **Modelagem**: aplicação de Regressão Linear e Random Forest
- **Avaliação**: uso de métricas MAE, RMSE e R² para medir desempenho
- **Geração de previsões**: exportação de resultados em formato .csv

---

## 🗂️ Sobre a Base de Dados

Os dados utilizados neste projeto foram originalmente extraídos de fontes abertas de comportamento de vendas no varejo.
A base foi adaptada para o contexto de previsão de demanda, incluindo:

- Padronização de colunas
- Conversão e ajuste de datas
- Simulação de variáveis de promoção (~15% dos registros)
- Inclusão de marcação automática de feriados nacionais brasileiros

Essas adaptações foram necessárias para tornar o conjunto de dados mais aderente a cenários reais de negócios e permitir a aplicação de modelos de Machine Learning focados em previsão de vendas.

Fonte original dos dados:  
**Retail Sales Forecasting Dataset - Kaggle**

---

## ⚙️ Como Rodar o Projeto

1. Instale as dependências necessárias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib holidays
```

2. Para rodar o pipeline completo (treinamento do modelo):

```bash
python src/main.py
```

3. Para gerar previsões com o modelo já treinado:

```bash
python src/predict.py
```

As previsões serão salvas em `outputs/previsoes_vendas.csv`.

---

## 👨‍💻 Autor

Projeto desenvolvido por **Marcos Scatolino** — 2025.

---

## 📞 Contato

- Email: marcosscatolino@gmail.com
- LinkedIn: [Marcos Scatolino](https://www.linkedin.com/in/marcos-scatolino)

---
