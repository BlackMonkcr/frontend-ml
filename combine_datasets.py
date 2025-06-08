#!/usr/bin/env python3
"""
Script para combinar metadatos con características en un solo CSV
"""

import pandas as pd
import numpy as np

def combine_metadata_and_features():
    """Combina metadatos de películas con características extraídas"""
    try:
        # Cargar metadatos
        print("Cargando metadatos...")
        metadata = pd.read_csv('prim_reduced.csv')  # movieId, title, poster_url, poster_source
        print(f"Metadatos cargados: {len(metadata)} películas")

        # Cargar características
        print("Cargando características...")
        features = pd.read_csv('prim_reduced_features.csv')  # index, movieId, features 0-9
        print(f"Características cargadas: {len(features)} películas")

        # Combinar basándose en movieId
        print("Combinando datasets...")
        combined = pd.merge(metadata, features, on='movieId', how='inner')
        print(f"Dataset combinado: {len(combined)} películas")

        # Reorganizar columnas para mejor claridad
        cols = ['movieId', 'title', 'poster_url', 'poster_source'] + [str(i) for i in range(10)]
        combined = combined[cols]

        # Guardar resultado
        output_path = 'movie_data_complete.csv'
        combined.to_csv(output_path, index=False)
        print(f"Dataset combinado guardado en: {output_path}")

        # Mostrar estadísticas
        print("\n📊 Estadísticas del dataset combinado:")
        print(f"  - Total películas: {len(combined)}")
        print(f"  - Columnas: {list(combined.columns)}")
        print(f"  - Películas con poster_url: {len(combined[combined['poster_url'].notna()])}")

        # Mostrar muestra
        print("\n📋 Muestra del dataset:")
        print(combined.head())

        return combined

    except Exception as e:
        print(f"Error combinando datasets: {str(e)}")
        return None

if __name__ == "__main__":
    combined_df = combine_metadata_and_features()
