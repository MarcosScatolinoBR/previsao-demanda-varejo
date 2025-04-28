"""
Módulo de Análise Exploratória (EDA) do projeto de Previsão de Demanda.
Recebe o dataframe já processado e realiza análises iniciais e gráficos.

Autor: Marcos Scatolino
Data: 2025-04-24
"""

# ================== IMPORTS ================== #
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ================== FUNÇÃO PRINCIPAL ================== #
def executar_eda(df):
    """
    Executa a análise exploratória dos dados recebidos:
    - Visualizações iniciais
    - Estatísticas descritivas por tipo
    - Gráficos básicos com anotações
    """

    # ================== VISÃO GERAL ================== #
    print("📊 Primeiras Linhas do Dataset:")
    print(df.head())

    print("\nℹ️ Informações do Dataset:")
    print(df.info())

    # ================== ESTATÍSTICAS ================== #
    print("\n📈 Estatísticas Descritivas (Numéricas):")
    print(df.select_dtypes(include='number').describe())

    print("\n📆 Estatísticas da Coluna de Datas:")
    print("Data mínima:", df["date"].min())
    print("Data máxima:", df["date"].max())
    print("Total de dias únicos:", df["date"].nunique())

    # ================== GRÁFICO 1: Vendas ao longo do tempo ================== #
    plt.figure(num="📈 Vendas por Dia", figsize=(12, 5))
    df.groupby("date")["sales"].sum().plot()
    plt.title("📈 Vendas Totais por Dia")
    plt.xlabel("Data")
    plt.ylabel("Vendas")
    plt.tight_layout()
    plt.show()

    # ================== GRÁFICO 2: Vendas com e sem promoção ================== #
    plt.figure(num="🛍️ Promoção vs Sem Promoção", figsize=(8, 5))
    sns.barplot(data=df, x="promotion", y="sales", estimator=sum, errorbar=None)
    plt.title("🛍️ Vendas Totais: Promoção x Sem Promoção")
    plt.xticks([0, 1], ["Sem Promoção", "Com Promoção"])
    plt.ylabel("Vendas Totais")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

    # ================== GRÁFICO 3: Vendas em Feriados (com anotações) ================== #
    plt.figure(num="🎉 Vendas em Feriados", figsize=(8, 5))
    ax = sns.boxplot(data=df, x="feriado", y="sales")
    plt.title("🎉 Distribuição de Vendas: Dias Com e Sem Feriado")
    plt.xticks([0, 1], ["Dia Normal", "Feriado"])
    plt.ylabel("Vendas")
    plt.xlabel("")

    # Adiciona anotações no gráfico (mediana, Q1, Q3)
    grupos = df.groupby("feriado")["sales"]
    for i, (grupo, vendas) in enumerate(grupos):
        q1 = vendas.quantile(0.25)
        q2 = vendas.median()
        q3 = vendas.quantile(0.75)

        ax.text(i, q2, f"Mediana: {int(q2)}", ha='center', va='bottom', color='black', fontweight='bold')
        ax.text(i, q1, f"Q1: {int(q1)}", ha='center', va='top', color='gray')
        ax.text(i, q3, f"Q3: {int(q3)}", ha='center', va='bottom', color='gray')

    plt.tight_layout()
    plt.show()
