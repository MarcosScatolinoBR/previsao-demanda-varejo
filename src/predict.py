"""
Módulo de Previsão de Vendas usando o modelo treinado.

Carrega o modelo salvo, aplica as features no novo dataset,
gera previsões e exporta os resultados.

Autor: Marcos Scatolino
Data: 2025-04-24
"""

# ================== IMPORTS ================== #
import pandas as pd
import joblib
import os
import numpy as np
import holidays
from feature_engineering import adicionar_features


# ================== FUNÇÃO PRINCIPAL ================== #
def fazer_previsao():
    """
    Carrega o modelo salvo, aplica feature engineering,
    realiza previsões e salva o resultado em CSV.
    """

    # ================== CARREGAR MODELO ================== #
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    modelo_path = os.path.join(BASE_DIR, "models", "modelo_random_forest.pkl")
    modelo = joblib.load(modelo_path)

    print(f"✅ Modelo carregado de: {modelo_path}")

    # ================== CARREGAR NOVOS DADOS ================== #
    caminho_novos_dados = os.path.join(BASE_DIR, "data", "retail_dataset.csv")
    df_novos = pd.read_csv(caminho_novos_dados)

    # Conferência inicial
    print(f"📋 Novos dados carregados. Formato: {df_novos.shape}")

    # ================== AJUSTES INICIAIS ================== #
    # Renomeia as colunas para manter o padrão
    df_novos.rename(columns={
        "data": "date",
        "venda": "sales",
        "estoque": "stock",
        "preco": "price"
    }, inplace=True)

    # Conversão de datas
    df_novos["date"] = pd.to_datetime(df_novos["date"], errors="coerce")

    # ================== ADICIONAR FERIADOS ================== #
    feriados_br = holidays.Brazil()
    df_novos["feriado"] = df_novos["date"].apply(lambda x: x in feriados_br)

    # ================== ADICIONAR PROMOÇÕES ================== #
    np.random.seed(42)
    df_novos["promotion"] = np.random.choice([0, 1], size=len(df_novos), p=[0.85, 0.15])

    # ================== FEATURE ENGINEERING ================== #
    df_novos = adicionar_features(df_novos)

    # ================== PREPARAR PARA PREDIÇÃO ================== #
    features = [
        "day_of_week", "week_of_year", "month", "year",
        "is_weekend", "is_month_start", "is_month_end", "promotion", "feriado"
    ]

    # Garante que booleanos sejam inteiros
    df_novos["is_weekend"] = df_novos["is_weekend"].astype(int)
    df_novos["is_month_start"] = df_novos["is_month_start"].astype(int)
    df_novos["is_month_end"] = df_novos["is_month_end"].astype(int)
    df_novos["feriado"] = df_novos["feriado"].astype(int)

    # ================== REALIZAR PREVISÕES ================== #
    previsoes = modelo.predict(df_novos[features])

    # Adiciona a coluna de previsão
    df_novos["sales_predicted"] = previsoes

    # ================== EXPORTAR RESULTADOS ================== #
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    caminho_saida = os.path.join(outputs_dir, "previsoes_vendas.csv")
    df_novos.to_csv(caminho_saida, index=False)

    print(f"💾 Previsões salvas em: {caminho_saida}")

    return df_novos


# ================== EXECUÇÃO DIRETA ================== #
if __name__ == "__main__":
    df_resultado = fazer_previsao()
