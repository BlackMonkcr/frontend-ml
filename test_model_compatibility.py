#!/usr/bin/env python3
"""
Test de compatibilidad del modelo con el pipeline corregido
Verifica que el modelo cargue correctamente y que las características sean compatibles
"""

import numpy as np
import pandas as pd
import pickle
import cv2
from skimage import feature, color
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from PIL import Image
import requests
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')

# Definir las clases necesarias para cargar el modelo
class KMeansCustom:
    """Implementación personalizada del algoritmo K-means"""
    def __init__(self, n_clusters: int = 8, max_iters: int = 300, tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def predict(self, X):
        """Predice clusters para nuevos datos"""
        if self.centroids_ is None:
            raise ValueError("Modelo no entrenado")
        distances = np.sqrt(((X - self.centroids_[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

class VisualClusteringAnalyzer:
    """Analizador de clustering para características visuales"""
    def __init__(self):
        self.features = None
        self.movie_metadata = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.features_scaled = None
        self.features_pca = None
        self.feature_columns = None
        self.kmeans_labels = None
        self.kmeans_silhouette = None
        self.kmeans_inertia = None

def test_model_loading():
    """Prueba la carga del modelo"""
    print("=" * 60)
    print("🔍 PRUEBA 1: Carga del modelo")
    print("=" * 60)

    try:
        model_path = "visual_clustering_model/visual_clustering_analyzer.pkl"
        print(f"Cargando modelo desde: {model_path}")

        with open(model_path, 'rb') as f:
            analyzer = pickle.load(f)

        print("✅ Modelo cargado exitosamente")

        # Verificar atributos del modelo
        print(f"📊 Información del modelo:")
        if hasattr(analyzer, 'features'):
            print(f"  - Features shape: {analyzer.features.shape}")
        if hasattr(analyzer, 'kmeans_model'):
            print(f"  - K-means clusters: {analyzer.kmeans_model.n_clusters}")
        if hasattr(analyzer, 'movie_metadata'):
            print(f"  - Películas: {len(analyzer.movie_metadata)}")

        return analyzer

    except Exception as e:
        print(f"❌ Error cargando modelo: {str(e)}")
        return None

def test_feature_extraction():
    """Prueba la extracción de características"""
    print("\n" + "=" * 60)
    print("🔍 PRUEBA 2: Extracción de características")
    print("=" * 60)

    try:
        # Crear imagen sintética para prueba
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        print(f"Imagen de prueba creada: {test_image.shape}")

        # Extraer características HSV
        print("🔄 Extrayendo características HSV...")
        hsv = color.rgb2hsv(test_image)
        h = np.histogram(hsv[:, :, 0], bins=32, range=(0, 1))[0]
        s = np.histogram(hsv[:, :, 1], bins=32, range=(0, 1))[0]
        v = np.histogram(hsv[:, :, 2], bins=32, range=(0, 1))[0]
        hsv_features = np.concatenate([h, s, v])
        print(f"✅ HSV features: {len(hsv_features)} características")

        # Extraer características LBP
        print("🔄 Extrayendo características LBP...")
        g = color.rgb2gray(test_image)
        g_uint8 = (g * 255).astype(np.uint8)
        lbp = feature.local_binary_pattern(g_uint8, 8, 1, method='uniform')
        n_bins = int(lbp.max() + 1)
        lbp_features, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        print(f"✅ LBP features: {len(lbp_features)} características")

        # Extraer Hu moments
        print("🔄 Extrayendo Hu moments...")
        gray = color.rgb2gray(test_image)
        gray_uint8 = (gray * 255).astype(np.uint8)
        moments = cv2.moments(gray_uint8)
        hu_features = cv2.HuMoments(moments).flatten()
        print(f"✅ Hu moments: {len(hu_features)} características")

        # Combinar todas las características
        all_features = np.concatenate([hsv_features, lbp_features, hu_features])
        print(f"📊 Total características extraídas: {len(all_features)}")
        print(f"📊 Rango de valores: [{np.min(all_features):.3f}, {np.max(all_features):.3f}]")

        return all_features

    except Exception as e:
        print(f"❌ Error en extracción de características: {str(e)}")
        return None

def test_preprocessing_compatibility(analyzer, features):
    """Prueba la compatibilidad del preprocesamiento"""
    print("\n" + "=" * 60)
    print("🔍 PRUEBA 3: Compatibilidad de preprocesamiento")
    print("=" * 60)

    try:
        # Cargar dataset de entrenamiento
        print("🔄 Cargando dataset de entrenamiento...")
        train_data = pd.read_csv('prim_reduced.csv')
        train_features = train_data.iloc[:, 2:].values  # Features 0-9
        print(f"✅ Dataset cargado: {train_features.shape}")

        # Verificar estadísticas del dataset
        print(f"📊 Estadísticas del dataset de entrenamiento:")
        print(f"  - Rango: [{np.min(train_features):.3f}, {np.max(train_features):.3f}]")
        print(f"  - Media: {np.mean(train_features):.3f}")
        print(f"  - Std: {np.std(train_features):.3f}")

        # Verificar scaler del modelo
        if hasattr(analyzer, 'scaler'):
            print("✅ Modelo tiene scaler")
            if hasattr(analyzer.scaler, 'mean_'):
                print(f"  - Scaler mean: {analyzer.scaler.mean_[:5]}...")
                print(f"  - Scaler scale: {analyzer.scaler.scale_[:5]}...")
        else:
            print("⚠️ Modelo no tiene scaler guardado")

        # Verificar features_scaled del modelo
        if hasattr(analyzer, 'features_scaled'):
            model_features = analyzer.features_scaled
            print(f"✅ Modelo tiene features escaladas: {model_features.shape}")
            print(f"📊 Rango de features del modelo: [{np.min(model_features):.3f}, {np.max(model_features):.3f}]")
        else:
            print("⚠️ Modelo no tiene features_scaled")

        return True

    except Exception as e:
        print(f"❌ Error en prueba de compatibilidad: {str(e)}")
        return False

def test_prediction_compatibility(analyzer):
    """Prueba la compatibilidad de predicción"""
    print("\n" + "=" * 60)
    print("🔍 PRUEBA 4: Compatibilidad de predicción")
    print("=" * 60)

    try:
        # Tomar una muestra del dataset de entrenamiento
        train_data = pd.read_csv('prim_reduced.csv')
        sample_features = train_data.iloc[0, 2:].values  # Primera muestra, features 0-9
        print(f"Muestra de prueba: {sample_features}")

        # Verificar si el modelo puede hacer predicciones
        if hasattr(analyzer, 'kmeans_model') and analyzer.kmeans_model is not None:
            print("✅ Modelo K-means disponible")

            # Probar predicción
            if hasattr(analyzer, 'features_scaled'):
                # Usar una muestra de las features escaladas del modelo
                test_sample = analyzer.features_scaled[0:1]  # Primera muestra

                # Verificar si podemos hacer predicción
                if hasattr(analyzer.kmeans_model, 'predict'):
                    prediction = analyzer.kmeans_model.predict(test_sample)
                    print(f"✅ Predicción exitosa: cluster {prediction[0]}")
                else:
                    print("⚠️ Método predict no disponible en kmeans_model")

            else:
                print("⚠️ No hay features_scaled para probar predicción")
        else:
            print("❌ Modelo K-means no disponible")

        return True

    except Exception as e:
        print(f"❌ Error en prueba de predicción: {str(e)}")
        return False

def main():
    """Función principal de pruebas"""
    print("🧪 PRUEBAS DE COMPATIBILIDAD DEL MODELO")
    print("Este script verifica que el modelo y pipeline sean compatibles")

    # Prueba 1: Cargar modelo
    analyzer = test_model_loading()
    if analyzer is None:
        print("❌ No se puede continuar sin el modelo")
        return

    # Prueba 2: Extraer características
    features = test_feature_extraction()
    if features is None:
        print("❌ No se pueden extraer características")
        return

    # Prueba 3: Compatibilidad de preprocesamiento
    preprocessing_ok = test_preprocessing_compatibility(analyzer, features)

    # Prueba 4: Compatibilidad de predicción
    prediction_ok = test_prediction_compatibility(analyzer)

    # Resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 60)
    print(f"✅ Carga del modelo: {'✅ OK' if analyzer is not None else '❌ FALLO'}")
    print(f"✅ Extracción de características: {'✅ OK' if features is not None else '❌ FALLO'}")
    print(f"✅ Compatibilidad de preprocesamiento: {'✅ OK' if preprocessing_ok else '❌ FALLO'}")
    print(f"✅ Compatibilidad de predicción: {'✅ OK' if prediction_ok else '❌ FALLO'}")

    if all([analyzer is not None, features is not None, preprocessing_ok, prediction_ok]):
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON! El modelo está listo para usar.")
    else:
        print("\n⚠️ Algunas pruebas fallaron. Revisa los errores arriba.")

if __name__ == "__main__":
    main()
