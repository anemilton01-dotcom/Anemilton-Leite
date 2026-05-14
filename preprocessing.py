"""
preprocessing.py
Pré-processamento, feature engineering e divisão dos dados.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features a partir das existentes.

    Args:
        df: DataFrame original

    Returns:
        DataFrame com novas features
    """
    df = df.copy()

    # Razão cômodos por domicílio
    df["RoomsPerHouse"] = df["AveRooms"] / df["HouseAge"].replace(0, 1)

    # Pessoas por cômodo
    df["PopPerRoom"] = df["Population"] / (df["AveRooms"] * df["AveOccup"]).replace(0, 1)

    # Distância ao centro de Los Angeles (aproximado)
    la_lat, la_lon = 34.05, -118.25
    df["DistToLA"] = np.sqrt(
        (df["Latitude"] - la_lat) ** 2 + (df["Longitude"] - la_lon) ** 2
    )

    # Distância a São Francisco
    sf_lat, sf_lon = 37.77, -122.42
    df["DistToSF"] = np.sqrt(
        (df["Latitude"] - sf_lat) ** 2 + (df["Longitude"] - sf_lon) ** 2
    )

    # Renda por cômodo
    df["IncomePerRoom"] = df["MedInc"] / df["AveRooms"].replace(0, 1)

    print(f"✅ Feature engineering concluído. Novas features: RoomsPerHouse, PopPerRoom, DistToLA, DistToSF, IncomePerRoom")
    return df


def split_data(df: pd.DataFrame, target: str = "Price", test_size: float = 0.2, random_state: int = 42):
    """
    Divide os dados em treino e teste.

    Args:
        df: DataFrame completo
        target: Nome da coluna alvo
        test_size: Proporção do conjunto de teste
        random_state: Semente aleatória

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\n✂️  Divisão dos dados:")
    print(f"   Treino : {X_train.shape[0]} amostras")
    print(f"   Teste  : {X_test.shape[0]} amostras")
    print(f"   Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def get_scaler() -> StandardScaler:
    """Retorna um StandardScaler pré-configurado."""
    return StandardScaler()
