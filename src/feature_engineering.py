"""
Módulo de engenharia de features para o projeto de Previsão de Demanda.

Cria variáveis derivadas de data para ajudar na modelagem preditiva.

Autor: Marcos Scatolino
Data: 2025-04-24
"""

# ================== IMPORTS ================== #
import pandas as pd


# ================== FUNÇÃO PRINCIPAL ================== #
def adicionar_features(df):
    """
    Enriquece o dataframe com variáveis de tempo e médias móveis.
    Retorna o DataFrame com novas colunas.
    """

    # Dia da semana (0=segunda, 6=domingo)
    df["day_of_week"] = df["date"].dt.dayofweek

    # Número da semana no ano
    df["week_of_year"] = df["date"].dt.isocalendar().week

    # Mês e ano
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # Se é final de semana
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    # Se é início ou fim do mês
    df["is_month_start"] = df["date"].dt.is_month_start
    df["is_month_end"] = df["date"].dt.is_month_end

    # Média móvel de 7 dias (por ano e mês para evitar mistura)
    df = df.sort_values("date")
    df["rolling_mean_7d"] = df["sales"].rolling(window=7, min_periods=1).mean()

    return df
