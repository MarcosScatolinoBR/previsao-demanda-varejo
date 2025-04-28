"""
Módulo de pré-processamento: enriquecimento da base com variáveis realistas.

- Adiciona coluna de promoção
- Marca feriados nacionais brasileiros
- Padroniza nomes de colunas

Autor: Marcos Scatolino
Data: 2025-04-24
"""

# ================== IMPORTS ================== #
import pandas as pd
import numpy as np
import os
from datetime import datetime
import holidays


# ================== FUNÇÃO PRINCIPAL ================== #
def preparar_dados():
    """
    Enriquecimento da base original:
    - Simula promoções
    - Marca feriados nacionais
    - Retorna dataframe enriquecido
    """

    # Define caminho absoluto até a raiz do projeto
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    # Define caminho da base original
    caminho_base = os.path.join(BASE_DIR, "data", "retail_dataset.csv")

    # Lê o dataset
    df = pd.read_csv(caminho_base)

    # Exibe colunas originais (para debug - opcional)
    print("🧾 Colunas originais:", df.columns.tolist())

    # Renomeia colunas para padrão em inglês (com base real detectada)
    df.rename(columns={
        "data": "date",
        "venda": "sales",
        "estoque": "stock",
        "preco": "price"
    }, inplace=True)

    # Converte a coluna de data
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Marca feriados nacionais brasileiros
    feriados_br = holidays.Brazil()
    df["feriado"] = df["date"].apply(lambda x: x in feriados_br)

    # Simula promoções: 15% dos registros com promoção
    np.random.seed(42)
    df["promotion"] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])

    # Garante que 'sales' seja numérico antes de aplicar aumento
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0).astype(int)

    # Aumenta vendas em 40% nos dias de promoção
    df["sales"] = df.apply(
        lambda row: int(row["sales"] * 1.4) if row["promotion"] == 1 else row["sales"],
        axis=1
    )

    # Exibe colunas finais para conferência
    print("✅ Dados processados com sucesso! Colunas finais:")
    print(df.columns.tolist())

    return df
