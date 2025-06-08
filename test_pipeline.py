#!/usr/bin/env python3
"""
Script de prueba del pipeline corregido para verificar consistencia
con el modelo entrenado
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# Importar funciones del app.py corregido
from app import (
    extract_all_features_original_pipeline,
    apply_preprocessing_pipeline,
    load_preprocessing_pipeline,
    download_img
)

def test_image_processing_pipeline():
    """
    Prueba el pipeline completo de procesamiento de imágenes
    """
    print("🧪 Iniciando prueba del pipeline corregido...")

    # Cargar una imagen de ejemplo
    test_image_urls = [
        "https://image.tmdb.org/t/p/w500/6CoRTJTmijhBLJTUNoVSUNxZMEI.jpg",  # Ejemplo de poster
        "https://via.placeholder.com/500x750/FF0000/FFFFFF?text=Test"  # Imagen de prueba simple
    ]

    results = []

    for i, url in enumerate(test_image_urls):
        print(f"\n📸 Procesando imagen {i+1}: {url[:50]}...")

        try:
            # Descargar imagen
            img_array = download_img(url)
            if img_array is None:
                print(f"❌ Error al descargar imagen {i+1}")
                continue

            print(f"✅ Imagen descargada: {img_array.shape}")

            # Extraer características usando pipeline original
            features = extract_all_features_original_pipeline(img_array)
            if features is None:
                print(f"❌ Error al extraer características de imagen {i+1}")
                continue

            print(f"✅ Características extraídas: {len(features)} features")
            print(f"   - HSV: ~96 features")
            print(f"   - LBP: ~26 features")
            print(f"   - Hu: 7 features")
            print(f"   - Total: {len(features)} features")

            # Aplicar preprocesamiento
            preprocessing_pipeline = load_preprocessing_pipeline()
            reduced_features = apply_preprocessing_pipeline(features, preprocessing_pipeline)

            if reduced_features is None:
                print(f"❌ Error en preprocesamiento de imagen {i+1}")
                continue

            print(f"✅ Preprocesamiento exitoso: {len(reduced_features)} features finales")
            print(f"   - Features finales: {reduced_features}")

            results.append({
                'image_id': i+1,
                'url': url,
                'original_features': len(features),
                'final_features': len(reduced_features),
                'feature_values': reduced_features
            })

        except Exception as e:
            print(f"❌ Error procesando imagen {i+1}: {str(e)}")
            continue

    return results

def compare_with_training_data():
    """
    Compara las características generadas con el dataset de entrenamiento
    """
    print("\n🔍 Comparando con dataset de entrenamiento...")

    try:
        # Cargar dataset de entrenamiento
        train_data = pd.read_csv('d:/ML_P2/prim_reduced.csv')
        train_features = train_data.iloc[:, 2:].values  # Features 0-9

        print(f"✅ Dataset de entrenamiento cargado:")
        print(f"   - Muestras: {len(train_data)}")
        print(f"   - Features por muestra: {train_features.shape[1]}")
        print(f"   - Rango de valores:")

        for i in range(train_features.shape[1]):
            feature_col = train_features[:, i]
            print(f"     Feature {i}: [{feature_col.min():.3f}, {feature_col.max():.3f}]")

        return train_features

    except Exception as e:
        print(f"❌ Error cargando dataset de entrenamiento: {str(e)}")
        return None

def validate_model_compatibility(test_results, train_features):
    """
    Valida que las características generadas sean compatibles con el modelo
    """
    print("\n✅ Validando compatibilidad con modelo...")

    if not test_results:
        print("❌ No hay resultados de prueba para validar")
        return False

    all_compatible = True

    for result in test_results:
        features = result['feature_values']

        print(f"\n🔬 Validando imagen {result['image_id']}:")

        # Verificar dimensiones
        if len(features) != 10:
            print(f"❌ Dimensiones incorrectas: {len(features)} (esperado: 10)")
            all_compatible = False
            continue
        else:
            print(f"✅ Dimensiones correctas: {len(features)} features")

        # Verificar rangos de valores comparando con datos de entrenamiento
        if train_features is not None:
            for i, feature_val in enumerate(features):
                train_min = train_features[:, i].min()
                train_max = train_features[:, i].max()

                if feature_val < train_min - 3 * np.std(train_features[:, i]) or \
                   feature_val > train_max + 3 * np.std(train_features[:, i]):
                    print(f"⚠️  Feature {i} fuera de rango esperado: {feature_val:.3f} (entrenamiento: [{train_min:.3f}, {train_max:.3f}])")
                else:
                    print(f"✅ Feature {i} en rango válido: {feature_val:.3f}")

    return all_compatible

def main():
    """
    Función principal de prueba
    """
    print("🎬 PRUEBA DEL PIPELINE CORREGIDO")
    print("="*50)

    # Prueba 1: Procesamiento de imágenes
    test_results = test_image_processing_pipeline()

    # Prueba 2: Comparación con datos de entrenamiento
    train_features = compare_with_training_data()

    # Prueba 3: Validación de compatibilidad
    is_compatible = validate_model_compatibility(test_results, train_features)

    # Resumen final
    print("\n" + "="*50)
    print("📊 RESUMEN DE PRUEBAS")
    print("="*50)

    print(f"🖼️  Imágenes procesadas exitosamente: {len(test_results)}")
    print(f"🔗 Compatibilidad con modelo: {'✅ SÍ' if is_compatible else '❌ NO'}")

    if test_results:
        print(f"📈 Características extraídas por imagen: {test_results[0]['original_features']}")
        print(f"🎯 Características finales: {test_results[0]['final_features']}")

    print("\n✅ Prueba completada!")

    if is_compatible:
        print("🎉 El pipeline está funcionando correctamente!")
        print("💡 Puedes proceder a usar la aplicación Streamlit.")
    else:
        print("⚠️  Hay problemas de compatibilidad que deben resolverse.")

    return is_compatible

if __name__ == "__main__":
    main()
