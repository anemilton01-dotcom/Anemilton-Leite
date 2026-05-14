"""
data_loader.py
Carregamento e inspeção inicial dos dados.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import os


def load_data(save_csv: bool = True, data_dir: str = "data") -> pd.DataFrame:
    """
    Carrega o California Housing Dataset e retorna como DataFrame.

    Args:
        save_csv: Se True, salva os dados em data/housing.csv
        data_dir: Diretório onde o CSV será salvo

    Returns:
        DataFrame com os dados de habitação
    """
    print("📥 Carregando California Housing Dataset...")
    housing = fetch_california_housing(as_frame=True)

    df = housing.frame.copy()
    df.rename(columns={"MedHouseVal": "Price"}, inplace=True)

    # Price está em centenas de milhares de dólares; converter para dólares
    df["Price"] = df["Price"] * 100_000

    if save_csv:
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, "housing.csv")
        df.to_csv(path, index=False)
        print(f"✅ Dados salvos em '{path}'")

    print(f"   Shape: {df.shape}")
    print(f"   Colunas: {list(df.columns)}")
    return df


def basic_info(df: pd.DataFrame) -> None:
    """Exibe informações básicas sobre o dataset."""
    print("\n📋 Informações do Dataset")
    print("=" * 40)
    print(df.info())
    print("\n📊 Estatísticas descritivas:")
    print(df.describe().round(2))

    missing = df.isnull().sum()
    if missing.any():
        print("\n⚠️  Valores ausentes:")
        print(missing[missing > 0])
    else:
        print("\n✅ Nenhum valor ausente encontrado.")
