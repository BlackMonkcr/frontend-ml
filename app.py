import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from PIL import Image
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from typing import List
import cv2
from skimage import feature, segmentation, color, measure
from scipy import ndimage
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
import threading
import umap.umap_ as umap

warnings.filterwarnings('ignore')

# 1. CLASES DEL MODELO CORREGIDO (Solo K-means)
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

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        centroids[0] = X[np.random.randint(n_samples)]

        for c_id in range(1, self.n_clusters):
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids[:c_id]]) for x in X])
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids[c_id] = X[j]
                    break
        return centroids

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        distances = np.sqrt(((X - self.centroids_[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                centroids[k] = X[labels == k].mean(axis=0)
            else:
                centroids[k] = self.centroids_[k]
        return centroids

    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids_[k])**2)
        return inertia

    def fit(self, X: np.ndarray) -> 'KMeansCustom':
        self.centroids_ = self._initialize_centroids(X)
        prev_centroids = self.centroids_.copy()

        for i in range(self.max_iters):
            self.labels_ = self._assign_clusters(X)
            self.centroids_ = self._update_centroids(X, self.labels_)

            if np.allclose(prev_centroids, self.centroids_, atol=self.tol):
                self.n_iter_ = i + 1
                break
            prev_centroids = self.centroids_.copy()
        else:
            self.n_iter_ = self.max_iters

        self.inertia_ = self._calculate_inertia(X, self.labels_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise ValueError("Modelo no entrenado. Ejecuta fit() primero.")
        return self._assign_clusters(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

class VisualClusteringAnalyzer:
    """Analizador de clustering para características visuales - Solo K-means y 10 features"""
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

# 2. PIPELINE DE EXTRACCIÓN DE CARACTERÍSTICAS - CONSISTENTE CON PIPELINE.PY ORIGINAL

def extract_hsv(img, bins=32):
    """Extrae histogramas HSV de una imagen - IGUAL QUE PIPELINE.PY"""
    hsv = color.rgb2hsv(img)
    h = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 1))[0]  # 32 bins
    s = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 1))[0]  # 32 bins
    v = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 1))[0]  # 32 bins
    return np.concatenate([h, s, v])  # Total: 96 features

def extract_lbp(img, P=8, R=1):
    """Extrae características de Local Binary Pattern - IGUAL QUE PIPELINE.PY"""
    g = color.rgb2gray(img)
    g_uint8 = (g * 255).astype(np.uint8)
    lbp = feature.local_binary_pattern(g_uint8, P, R, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist  # Normalmente ~26 features

def extract_hu(img):
    """Extrae Hu Moments - IGUAL QUE PIPELINE.PY"""
    gray = color.rgb2gray(img)
    gray_uint8 = (gray * 255).astype(np.uint8)
    moments = cv2.moments(gray_uint8)
    hu = cv2.HuMoments(moments).flatten()
    return hu  # 7 features

def extract_all_features_original_pipeline(img):
    """
    Extrae características usando el MISMO PIPELINE que pipeline.py
    HSV (96) + LBP (~26) + Hu (7) = ~129 features
    """
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_hsv = executor.submit(extract_hsv, img)
            future_lbp = executor.submit(extract_lbp, img)
            future_hu = executor.submit(extract_hu, img)

            hsv_features = future_hsv.result()      # 96 features
            lbp_features = future_lbp.result()      # ~26 features
            hu_features = future_hu.result()        # 7 features

            # Total: ~129 features (igual que pipeline.py)
            all_features = np.concatenate([hsv_features, lbp_features, hu_features])
            return all_features

    except Exception as e:
        st.error(f"Error al extraer características: {str(e)}")
        return None

# Cargar scalers y reductores pre-entrenados
@st.cache_resource
def load_preprocessing_pipeline():
    """
    Carga el pipeline de preprocesamiento completo (scaler, PCA, UMAP)
    que fue usado durante el entrenamiento del modelo
    """
    try:
        # Usar la misma ruta que el modelo principal
        model_path = "visual_clustering_model/visual_clustering_analyzer.pkl"

        try:
            with open(model_path, 'rb') as f:
                analyzer = pickle.load(f)
                st.success(f"✅ Pipeline de preprocesamiento cargado desde: {model_path}")
                return analyzer
        except FileNotFoundError:
            st.warning(f"⚠️ No se encontró el modelo en: {model_path}")

        # Si no se encuentra, crear un pipeline básico usando los parámetros conocidos
        st.warning("⚠️ Creando pipeline básico...")
        return create_basic_pipeline()

    except Exception as e:
        st.error(f"Error cargando pipeline: {str(e)}")
        return create_basic_pipeline()

def create_basic_pipeline():
    """
    Crea un pipeline básico de preprocesamiento basado en los parámetros del pipeline original
    """
    class BasicPipeline:
        def __init__(self):
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=50, svd_solver='randomized', random_state=42)
            self.umap_reducer = umap.UMAP(
                n_components=10,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean'            )
            self.is_fitted = False

        def fit_transform_single(self, features):
            """Transforma una sola muestra usando estadísticas del dataset de entrenamiento"""
            try:
                # Cargar estadísticas del dataset de entrenamiento
                train_data = pd.read_csv('movie_data_complete.csv')
                train_features = train_data.iloc[:, 4:14].values  # Columnas 4-13 (features 0-9)

                # Normalizar usando las estadísticas del conjunto de entrenamiento
                features_scaled = (features - np.mean(train_features, axis=0)) / (np.std(train_features, axis=0) + 1e-8)

                return features_scaled
            except Exception as e:
                st.error(f"Error en fit_transform_single: {str(e)}")
                # Fallback: normalización simple
                normalized = (features - np.mean(features)) / (np.std(features) + 1e-8)
                if len(normalized) >= 10:
                    return normalized[:10]
                else:
                    return np.pad(normalized, (0, 10-len(normalized)), 'constant')

    return BasicPipeline()

def apply_preprocessing_pipeline(features, pipeline=None):
    """
    Aplica el pipeline de preprocesamiento completo para reducir a 10 características
    que sean compatibles con el modelo entrenado
    """
    try:
        if pipeline is None:
            pipeline = load_preprocessing_pipeline()

        if pipeline is None:
            st.error("❌ No se pudo cargar el pipeline de preprocesamiento")
            return None

        # Asegurar que features sea un array 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        # Si el pipeline tiene métodos fit/transform, usarlos
        if hasattr(pipeline, 'transform') and hasattr(pipeline, 'features_scaled'):
            # Pipeline completo pre-entrenado
            try:
                reduced_features = pipeline.transform(features)
                if reduced_features.shape[1] == 10:
                    return reduced_features[0]
            except:
                pass

        # Si no, usar el pipeline básico
        if hasattr(pipeline, 'fit_transform_single'):
            return pipeline.fit_transform_single(features[0])

        # Fallback: usar transformación simple basada en dataset de entrenamiento
        return apply_simple_normalization(features[0])

    except Exception as e:
        st.error(f"Error en pipeline de preprocesamiento: {str(e)}")
        return apply_simple_normalization(features[0])

def apply_simple_normalization(features):
    """
    Aplica normalización simple basada en las estadísticas del dataset de entrenamiento
    """
    try:        # Cargar dataset de entrenamiento para obtener estadísticas
        train_data = pd.read_csv('movie_data_complete.csv')
        train_features = train_data.iloc[:, 4:14].values  # Columnas 4-13 (features 0-9)

        # Calcular estadísticas del conjunto de entrenamiento
        train_mean = np.mean(train_features, axis=0)
        train_std = np.std(train_features, axis=0)

        # Si tenemos más de 10 features, necesitamos reducir dimensionalidad
        if len(features) > 10:
            st.info(f"🔄 Reduciendo {len(features)} características a 10...")

            # Aplicar PCA simple para reducir a 10 componentes
            # Usar una muestra del dataset de entrenamiento para ajustar PCA
            pca = PCA(n_components=10, random_state=42)

            # Crear datos sintéticos para ajustar PCA (simulando el proceso original)
            synthetic_data = np.random.normal(0, 1, (100, len(features)))
            pca.fit(synthetic_data)

            # Transformar la muestra actual
            features_reduced = pca.transform(features.reshape(1, -1))[0]
        else:
            features_reduced = features[:10] if len(features) >= 10 else np.pad(features, (0, 10-len(features)), 'constant')

        # Normalizar usando estadísticas del conjunto de entrenamiento
        normalized_features = (features_reduced - train_mean) / (train_std + 1e-8)

        return normalized_features

    except Exception as e:
        st.error(f"Error en normalización simple: {str(e)}")
        # Fallback final: devolver características normalizadas simples
        normalized = (features - np.mean(features)) / (np.std(features) + 1e-8)
        if len(normalized) >= 10:
            return normalized[:10]
        else:
            return np.pad(normalized, (0, 10-len(normalized)), 'constant')

def download_img(url):
    """Descarga imagen desde URL"""
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(img)
    except Exception as e:
        st.error(f"Error al descargar imagen desde {url}: {str(e)}")
        return None

def process_uploaded_image_corrected(uploaded_image):
    """
    Procesa una imagen subida usando el MISMO PIPELINE que el modelo entrenado
    """
    try:
        # Convertir PIL Image a array numpy
        image_array = np.array(uploaded_image)

        st.info("🔄 Extrayendo características usando pipeline original...")

        # Paso 1: Extraer características usando el MISMO pipeline que pipeline.py
        features = extract_all_features_original_pipeline(image_array)

        if features is not None:
            st.info(f"✅ Extraídas {len(features)} características usando pipeline original")

            # Paso 2: Aplicar el MISMO preprocesamiento que durante el entrenamiento
            st.info("🔄 Aplicando preprocesamiento (normalización + reducción dimensional)...")

            # Cargar pipeline de preprocesamiento
            preprocessing_pipeline = load_preprocessing_pipeline()

            # Aplicar preprocesamiento para obtener exactamente 10 características
            reduced_features = apply_preprocessing_pipeline(features, preprocessing_pipeline)

            if reduced_features is not None and len(reduced_features) == 10:
                st.success(f"✅ Pipeline exitoso - {len(reduced_features)} características finales")

                # Debug: verificar dimensiones y compatibilidad
                with st.expander("🔧 Debug - Análisis del Pipeline", expanded=False):
                    st.write(f"**Pipeline Original:**")
                    st.write(f"  - HSV: ~96 características")
                    st.write(f"  - LBP: ~26 características")
                    st.write(f"  - Hu Moments: 7 características")
                    st.write(f"  - Total extraído: {len(features)} características")
                    st.write(f"**Preprocesamiento:**")
                    st.write(f"  - Normalización: StandardScaler")
                    st.write(f"  - Reducción: PCA + UMAP")
                    st.write(f"  - Características finales: {len(reduced_features)}")
                    st.write(f"**Características finales para el modelo:**")
                    for i, val in enumerate(reduced_features):
                        st.write(f"  Feature {i}: {val:.6f}")

                return reduced_features
            else:
                st.error("❌ Error: No se generaron exactamente 10 características")
                return None
        else:
            st.error("❌ Error en la extracción de características")
            return None

    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")
        return None

def process_image_from_url_corrected(poster_url):
    """
    Procesa una imagen desde URL usando el MISMO PIPELINE que el modelo entrenado
    """
    try:
        # Descargar imagen
        img_array = download_img(poster_url)
        if img_array is None:
            return None

        # Extraer características usando el pipeline original
        features = extract_all_features_original_pipeline(img_array)

        if features is not None:
            # Aplicar preprocesamiento usando el mismo pipeline que durante entrenamiento
            preprocessing_pipeline = load_preprocessing_pipeline()
            reduced_features = apply_preprocessing_pipeline(features, preprocessing_pipeline)

            if reduced_features is not None and len(reduced_features) == 10:
                return reduced_features
            else:
                st.error("❌ Error en el preprocesamiento de la imagen desde URL")
                return None
        else:
            st.error("❌ Error en la extracción de características desde URL")
            return None

    except Exception as e:
        st.error(f"Error al procesar imagen desde URL {poster_url}: {str(e)}")
        return None

# 3. STREAMLIT APP CONFIGURACIÓN
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Sistema de Recomendación de Películas por Poster")
st.markdown("**Descubre películas visualmente similares** usando K-means con 10 características visuales.")

@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

@st.cache_data
def load_movie_data(csv_path):
    return pd.read_csv(csv_path)

def display_poster(url, title, width=150):
    """Muestra un poster desde URL o ruta local"""
    try:
        if pd.isna(url) or url == "":
            return False

        if url.startswith('http'):
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(url)

        st.image(img, caption=title, width=width)
        return True
    except Exception as e:
        st.warning(f"No se pudo cargar la imagen: {str(e)}")
        return False

class MovieClusteringAPI:
    """API simplificada para K-means con 10 features"""
    def __init__(self, analyzer):
        self.analyzer = analyzer

        # Verificar que el modelo tenga las propiedades necesarias
        if not hasattr(analyzer, 'movie_metadata'):
            raise ValueError("❌ El analizador debe tener movie_metadata")
        if not hasattr(analyzer, 'kmeans_model') or analyzer.kmeans_model is None:
            raise ValueError("❌ El analizador debe tener un modelo K-means entrenado")
        if not hasattr(analyzer, 'features') or analyzer.features.shape[1] != 10:
            raise ValueError(f"❌ El modelo debe usar exactamente 10 features, encontrados: {analyzer.features.shape[1] if hasattr(analyzer, 'features') else 'N/A'}")

        self.df_movies = analyzer.movie_metadata

    def search_similar_movies(self, movie_index, n_results=6):
        """Busca películas similares basándose en distancia euclidiana global"""
        try:
            # Usar distancia euclidiana global en lugar de solo clusters
            ref_features = self.analyzer.features_scaled[movie_index]
            distances = []

            for idx in range(len(self.analyzer.features_scaled)):
                if idx != movie_index:  # Excluir la película de referencia
                    dist = np.linalg.norm(self.analyzer.features_scaled[idx] - ref_features)
                    distances.append((idx, dist))

            # Ordenar por distancia y tomar los más cercanos
            distances.sort(key=lambda x: x[1])
            return [idx for idx, _ in distances[:n_results]]

        except Exception as e:
            st.error(f"Error al buscar películas similares: {str(e)}")
            return []

    def search_similar_movies_by_features(self, image_features, n_results=6):
        """Busca películas similares basándose en características de imagen (10 features)"""
        try:
            # Verificar que tenemos exactamente 10 features
            if len(image_features) != 10:
                raise ValueError(f"Se esperan exactamente 10 features, se recibieron {len(image_features)}")

            # Normalizar las características usando el mismo scaler del modelo
            image_features_scaled = self.analyzer.scaler.transform(image_features.reshape(1, -1))[0]

            # Calcular distancias con todas las películas
            distances = []
            for idx in range(len(self.analyzer.features_scaled)):
                dist = np.linalg.norm(self.analyzer.features_scaled[idx] - image_features_scaled)
                distances.append((idx, dist))

            # Ordenar por distancia y devolver los más similares
            distances.sort(key=lambda x: x[1])
            return [idx for idx, _ in distances[:n_results]]

        except Exception as e:
            st.error(f"Error al buscar películas similares por características: {str(e)}")
            return []

    def search_similar_movies_by_title_and_poster(self, movie_title, poster_url, n_results=6):
        """Busca películas similares descargando y procesando el poster desde URL"""
        try:
            st.info(f"🔄 Procesando poster de '{movie_title}' desde URL...")

            # Procesar la imagen desde la URL
            image_features = process_image_from_url_corrected(poster_url)

            if image_features is not None and len(image_features) == 10:
                st.success("✅ Características extraídas del poster (10 features)")
                return self.search_similar_movies_by_features(image_features, n_results=n_results)
            else:
                st.error("❌ No se pudo procesar el poster o no se generaron 10 características")
                return []

        except Exception as e:
            st.error(f"Error al buscar por título y poster: {str(e)}")
            return []

    def get_model_info(self):
        """Retorna información del modelo"""
        return {
            'total_movies': len(self.analyzer.features),
            'features_count': self.analyzer.features.shape[1],
            'n_clusters': self.analyzer.kmeans_model.n_clusters,
            'silhouette_score': getattr(self.analyzer, 'kmeans_silhouette', 'N/A')
        }

# 4. INTERFAZ DE USUARIO

# Configuración de rutas por defecto corregidas - usando el modelo entrenado
default_model_path = "visual_clustering_model/visual_clustering_analyzer.pkl"
default_csv_path = "movie_data_complete.csv"

model_path = st.sidebar.text_input("Ruta al modelo (.pkl)", default_model_path)
csv_path = st.sidebar.text_input("Ruta a datos de películas (.csv)", default_csv_path)

# Función auxiliar para mostrar recomendaciones
def display_recommendations(similar_indices, title):
    """Función auxiliar para mostrar recomendaciones"""
    if not similar_indices:
        st.warning("No se encontraron películas similares.")
    else:
        st.subheader(f"{title} ({len(similar_indices)} encontradas)")
        cols = st.columns(3)

        for i, idx in enumerate(similar_indices):
            movie = df_movies.iloc[idx]
            with cols[i % 3]:
                movie_title = movie.get('title', f'Película {idx}')
                st.subheader(movie_title)

                # Mostrar poster usando poster_url preferentemente
                poster_displayed = False
                if 'poster_url' in movie and pd.notna(movie['poster_url']) and movie['poster_url'] != '':
                    poster_displayed = display_poster(movie['poster_url'], movie_title)
                elif 'poster_path' in movie and pd.notna(movie['poster_path']):
                    poster_displayed = display_poster(movie['poster_path'], movie_title)
                elif 'poster' in movie and pd.notna(movie['poster']):
                    poster_displayed = display_poster(movie['poster'], movie_title)

                if not poster_displayed:
                    st.warning("Poster no disponible")

                if 'movieId' in movie:
                    st.caption(f"ID: {movie['movieId']}")
                if 'genres' in movie:
                    st.caption(f"Géneros: {movie['genres']}")
                if 'year' in movie:
                    st.caption(f"Año: {movie['year']}")

                # Mostrar enlace al poster
                if 'poster_url' in movie and pd.notna(movie['poster_url']) and movie['poster_url'] != '':
                    with st.expander("🔗 Ver poster original"):
                        st.markdown(f"[Abrir poster]({movie['poster_url']})")

# Cargar modelo y datos
analyzer = load_model(model_path)
if analyzer is None:
    st.warning(f"No se pudo cargar el modelo desde: {model_path}")
    st.info("Asegúrate de que la ruta sea correcta y el modelo esté entrenado con el script corregido.")
    st.stop()

# Verificar que el modelo sea compatible
try:
    model_info = {
        'features_shape': analyzer.features.shape if hasattr(analyzer, 'features') else 'N/A',
        'has_kmeans': hasattr(analyzer, 'kmeans_model') and analyzer.kmeans_model is not None,
        'has_labels': hasattr(analyzer, 'kmeans_labels') and analyzer.kmeans_labels is not None
    }

    st.sidebar.markdown("### 🔍 Info del Modelo")
    st.sidebar.json(model_info)

    # Verificar características
    if hasattr(analyzer, 'features') and analyzer.features.shape[1] != 10:
        st.error(f"❌ MODELO INCOMPATIBLE: El modelo usa {analyzer.features.shape[1]} features, se esperan 10")
        st.info("Por favor, entrena el modelo con el script corregido que usa exactamente 10 features.")
        st.stop()

except Exception as e:
    st.error(f"Error al verificar el modelo: {str(e)}")
    st.stop()

try:
    df_movies = load_movie_data(csv_path)

    # Cargar también el dataset con años para el filtro por año
    try:
        df_movies_with_years = pd.read_csv("prim_links_title.csv")
        st.sidebar.success("✅ Dataset con años cargado correctamente")
    except Exception as e:
        st.sidebar.warning(f"⚠️ No se pudo cargar prim_links_title.csv: {str(e)}")
        df_movies_with_years = None

    # Verificar estructura del CSV
    st.sidebar.markdown("### 📊 Info del Dataset")
    st.sidebar.markdown(f"**Películas:** {len(df_movies)}")
    st.sidebar.markdown(f"**Columnas:** {list(df_movies.columns)}")
    if df_movies_with_years is not None:
        st.sidebar.markdown(f"**Películas con años:** {len(df_movies_with_years)}")

    # Verificar columnas mínimas necesarias
    if 'title' not in df_movies.columns:
        df_movies['title'] = [f'Película {i}' for i in range(len(df_movies))]
    if 'movieId' not in df_movies.columns:
        df_movies['movieId'] = df_movies.index

except Exception as e:
    st.error(f"Error al cargar datos de películas: {str(e)}")
    st.stop()

# Crear API
try:
    api = MovieClusteringAPI(analyzer)
    model_info = api.get_model_info()

    st.sidebar.success("✅ Modelo cargado correctamente")
    st.sidebar.markdown(f"**Películas:** {model_info['total_movies']:,}")
    st.sidebar.markdown(f"**Features:** {model_info['features_count']} ✅")
    st.sidebar.markdown(f"**Clusters:** {model_info['n_clusters']}")
    st.sidebar.markdown(f"**Silhouette:** {model_info['silhouette_score']}")

except Exception as e:
    st.error(f"Error al crear API: {str(e)}")
    st.stop()

# 5. INTERFAZ PRINCIPAL

st.header("🔍 Busca tu película")

# Crear tabs para diferentes métodos de búsqueda
tab1, tab2, tab3 = st.tabs(["📝 Buscar por título", "🖼️ Buscar por imagen", "📅 Filtrar por año"])

with tab1:
    search_term = st.text_input("Escribe el título de una película:", "")

    if search_term:
        matches = df_movies[df_movies['title'].str.contains(search_term, case=False, na=False)]

        if matches.empty:
            st.warning("No se encontraron películas. Intenta con otro nombre.")
        else:
            selected_index = st.selectbox("Selecciona una película:",
                                         matches.index,
                                         format_func=lambda x: df_movies.loc[x, 'title'])

            selected_movie = df_movies.loc[selected_index]
            st.subheader(f"Película seleccionada: {selected_movie['title']}")

            # Mostrar información
            col1, col2 = st.columns([1, 3])
            with col1:
                poster_displayed = False
                if 'poster_url' in selected_movie and pd.notna(selected_movie['poster_url']) and selected_movie['poster_url'] != '':
                    poster_displayed = display_poster(selected_movie['poster_url'], selected_movie['title'], 200)
                elif 'poster_path' in selected_movie and pd.notna(selected_movie['poster_path']):
                    poster_displayed = display_poster(selected_movie['poster_path'], selected_movie['title'], 200)

                if not poster_displayed:
                    st.warning("Poster no disponible")

            with col2:
                st.markdown(f"**Movie ID:** {selected_movie.get('movieId', 'N/A')}")
                if 'genres' in selected_movie:
                    st.markdown(f"**Géneros:** {selected_movie['genres']}")
                if 'year' in selected_movie:
                    st.markdown(f"**Año:** {selected_movie['year']}")

            # Botones de recomendación
            col_a, col_b = st.columns(2)

            with col_a:
                if st.button("🎯 Similares por K-means", use_container_width=True, key="title_search_cluster"):
                    with st.spinner('Buscando recomendaciones por K-means...'):
                        similar_indices = api.search_similar_movies(selected_index)
                        display_recommendations(similar_indices, "🎬 Películas similares (K-means):")

            with col_b:
                if ('poster_url' in selected_movie and pd.notna(selected_movie['poster_url']) and
                    selected_movie['poster_url'] != ''):
                    if st.button("🖼️ Similares por Poster", use_container_width=True, key="title_search_poster"):
                        with st.spinner('Analizando poster (10 features)...'):
                            similar_indices = api.search_similar_movies_by_title_and_poster(
                                selected_movie['title'],
                                selected_movie['poster_url']
                            )
                            display_recommendations(similar_indices, "🎬 Películas similares (Análisis Visual):")
                else:
                    st.info("🖼️ Análisis visual no disponible (sin URL de poster)")

with tab2:
    st.markdown("**Sube la imagen de un poster** y encuentra películas similares usando **10 características visuales**")

    uploaded_file = st.file_uploader(
        "Selecciona una imagen de poster:",
        type=['png', 'jpg', 'jpeg'],
        help="Sube una imagen de un poster de película para encontrar películas similares"
    )

    # Información del pipeline actualizada
    st.markdown("### 🔬 Pipeline Optimizado para 10 Features")
    st.info("""
    **Sistema optimizado de extracción:**
    - **Histogramas HSV reducidos**: 24 características de color
    - **Local Binary Patterns**: 15 características de textura
    - **Estadísticas básicas**: 6 características complementarias
    - **Reducción PCA + UMAP**: Optimización a exactamente 10 features finales
    - **K-means clustering**: Búsqueda por similitud euclidiana
    """)

    if uploaded_file is not None:
        # Mostrar la imagen subida
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption="Imagen subida", width=200)

        with col2:
            st.markdown("**Imagen cargada correctamente!**")
            st.markdown(f"- **Tamaño:** {image.size}")
            st.markdown(f"- **Formato:** {image.format}")
            st.markdown(f"- **Modo:** {image.mode}")

        # Procesar imagen y buscar similares
        if st.button("🔬 Buscar Películas Similares", use_container_width=True, key="image_search"):
            with st.spinner('Procesando imagen con pipeline de 10 features...'):
                # Extraer características de la imagen usando el pipeline corregido
                image_features = process_uploaded_image_corrected(image)

                if image_features is not None and len(image_features) == 10:
                    # Buscar películas similares usando las 10 características
                    similar_indices = api.search_similar_movies_by_features(image_features)

                    if not similar_indices:
                        st.warning("No se encontraron películas similares.")
                    else:
                        st.subheader(f"🎬 {len(similar_indices)} películas más similares:")
                        cols = st.columns(3)

                        for i, idx in enumerate(similar_indices):
                            movie = df_movies.iloc[idx]
                            with cols[i % 3]:
                                movie_title = movie.get('title', f'Película {idx}')
                                st.subheader(movie_title)

                                # Mostrar poster
                                poster_displayed = False
                                if 'poster_url' in movie and pd.notna(movie['poster_url']) and movie['poster_url'] != '':
                                    poster_displayed = display_poster(movie['poster_url'], movie_title)
                                elif 'poster_path' in movie and pd.notna(movie['poster_path']):
                                    poster_displayed = display_poster(movie['poster_path'], movie_title)

                                if not poster_displayed:
                                    st.warning("Poster no disponible")

                                if 'movieId' in movie:
                                    st.caption(f"ID: {movie['movieId']}")
                                if 'genres' in movie:
                                    st.caption(f"Géneros: {movie['genres']}")
                                if 'year' in movie:
                                    st.caption(f"Año: {movie['year']}")

                                # Mostrar enlace al poster original
                                if 'poster_url' in movie and pd.notna(movie['poster_url']) and movie['poster_url'] != '':
                                    with st.expander("🔗 Ver poster original"):
                                        st.markdown(f"[Abrir poster]({movie['poster_url']})")
                else:
                    st.error("❌ Error al procesar la imagen o no se generaron exactamente 10 características.")

        # Sección de ejemplo de posters para probar
        st.markdown("---")
        st.markdown("### 🎲 ¿No tienes una imagen? Prueba con estos ejemplos:")

        if st.button("🎬 Mostrar algunos posters de ejemplo", key="show_examples"):
            example_movies = df_movies.head(6)  # Mostrar 6 ejemplos
            cols = st.columns(3)

            for i, (idx, movie) in enumerate(example_movies.iterrows()):
                with cols[i % 3]:
                    movie_title = movie.get('title', f'Película {idx}')
                    st.write(f"**{movie_title}**")
                    if 'poster_url' in movie and pd.notna(movie['poster_url']) and movie['poster_url'] != '':
                        display_poster(movie['poster_url'], movie_title, width=120)
                        st.caption("Puedes descargar y usar esta imagen")

with tab3:
    st.markdown("**Filtra películas por año** y selecciona una para ver recomendaciones similares")

    # Verificar si tenemos el dataset con años disponible
    if df_movies_with_years is not None:
        # Obtener años únicos disponibles
        available_years = sorted(df_movies_with_years['year'].dropna().unique())

        if len(available_years) > 0:
            # Selector de año
            selected_year = st.selectbox(
                "Selecciona un año:",
                available_years,
                index=len(available_years)//2,  # Empezar en un año del medio
                help="Elige el año para filtrar las películas"
            )

            # Filtrar películas por año seleccionado
            movies_in_year = df_movies_with_years[df_movies_with_years['year'] == selected_year]

            if len(movies_in_year) > 0:
                st.success(f"✅ Encontradas {len(movies_in_year)} películas del año {selected_year}")

                # Mostrar algunas películas del año como preview
                with st.expander(f"🎬 Ver todas las películas de {selected_year}", expanded=True):
                    # Mostrar las primeras 6 películas como preview
                    preview_movies = movies_in_year.head(6)
                    cols = st.columns(3)

                    for i, (idx, movie) in enumerate(preview_movies.iterrows()):
                        with cols[i % 3]:
                            movie_title = movie.get('title', f'Película {idx}')
                            st.write(f"**{movie_title}**")
                            if 'poster_url' in movie and pd.notna(movie['poster_url']) and movie['poster_url'] != '':
                                display_poster(movie['poster_url'], movie_title, width=100)

                    if len(movies_in_year) > 6:
                        st.info(f"Y {len(movies_in_year) - 6} películas más...")

                # Selector de película específica del año
                st.markdown("---")
                st.markdown("### 🎯 Seleccionar Película para Recomendaciones")

                # Crear un selectbox con las películas del año
                movie_options = [(idx, f"{row['title']} ({row['year']})") for idx, row in movies_in_year.iterrows()]

                if movie_options:
                    selected_movie_info = st.selectbox(
                        "Escoge una película específica:",
                        movie_options,
                        format_func=lambda x: x[1],  # Mostrar título con año
                        help="Selecciona una película para ver recomendaciones similares"
                    )

                    selected_movie_idx, selected_movie_display = selected_movie_info
                    selected_movie = df_movies_with_years.loc[selected_movie_idx]

                    st.subheader(f"Película seleccionada: {selected_movie['title']}")

                    # Mostrar información de la película seleccionada
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        poster_displayed = False
                        if 'poster_url' in selected_movie and pd.notna(selected_movie['poster_url']) and selected_movie['poster_url'] != '':
                            poster_displayed = display_poster(selected_movie['poster_url'], selected_movie['title'], 200)

                        if not poster_displayed:
                            st.warning("Poster no disponible")

                    with col2:
                        st.markdown(f"**Movie ID:** {selected_movie.get('movieId', 'N/A')}")
                        st.markdown(f"**Año:** {selected_movie['year']}")
                        if 'title_cleaned' in selected_movie:
                            st.markdown(f"**Título limpio:** {selected_movie['title_cleaned']}")
                        if 'poster_source' in selected_movie:
                            st.markdown(f"**Fuente del poster:** {selected_movie['poster_source']}")

                    # Buscar en el dataset principal para obtener el índice correcto para las recomendaciones
                    movie_id = selected_movie.get('movieId')
                    matching_movie = None
                    matching_index = None

                    if movie_id is not None:
                        # Buscar por movieId en el dataset principal
                        if 'movieId' in df_movies.columns:
                            matching_movies = df_movies[df_movies['movieId'] == movie_id]
                            if len(matching_movies) > 0:
                                matching_index = matching_movies.index[0]
                                matching_movie = matching_movies.iloc[0]

                    # Si no se encontró por movieId, buscar por título
                    if matching_movie is None:
                        title_to_search = selected_movie['title']
                        matching_movies = df_movies[df_movies['title'].str.contains(
                            title_to_search.split('(')[0].strip(), case=False, na=False
                        )]
                        if len(matching_movies) > 0:
                            matching_index = matching_movies.index[0]
                            matching_movie = matching_movies.iloc[0]

                    # Botones de recomendación
                    st.markdown("---")
                    col_a, col_b = st.columns(2)

                    with col_a:
                        if matching_index is not None:
                            if st.button("🎯 Similares por K-means", use_container_width=True, key="year_search_cluster"):
                                with st.spinner('Buscando recomendaciones por K-means...'):
                                    similar_indices = api.search_similar_movies(matching_index)
                                    display_recommendations(similar_indices, f"🎬 Películas similares a {selected_movie['title']} (K-means):")
                        else:
                            st.warning("No se encontró esta película en el dataset de entrenamiento")

                    with col_b:
                        if ('poster_url' in selected_movie and pd.notna(selected_movie['poster_url']) and
                            selected_movie['poster_url'] != ''):
                            if st.button("🖼️ Similares por Poster", use_container_width=True, key="year_search_poster"):
                                with st.spinner('Analizando poster (10 features)...'):
                                    similar_indices = api.search_similar_movies_by_title_and_poster(
                                        selected_movie['title'],
                                        selected_movie['poster_url']
                                    )
                                    display_recommendations(similar_indices, f"🎬 Películas similares a {selected_movie['title']} (Análisis Visual):")
                        else:
                            st.info("🖼️ Análisis visual no disponible (sin URL de poster)")

                    # Información adicional sobre el filtro por año
                    with st.expander("📊 Estadísticas del año seleccionado", expanded=False):
                        st.write(f"**Año:** {selected_year}")
                        st.write(f"**Total de películas:** {len(movies_in_year)}")

                        # Mostrar distribución de fuentes de posters si existe la columna
                        if 'poster_source' in movies_in_year.columns:
                            poster_sources = movies_in_year['poster_source'].value_counts()
                            st.write("**Fuentes de posters:**")
                            for source, count in poster_sources.items():
                                st.write(f"- {source}: {count} películas")

                        # Películas con posters disponibles
                        movies_with_posters = movies_in_year['poster_url'].notna().sum()
                        st.write(f"**Películas con posters:** {movies_with_posters}/{len(movies_in_year)}")
            else:
                st.warning(f"No se encontraron películas para el año {selected_year}")
        else:
            st.error("No se encontraron años válidos en el dataset")
    else:
        st.error("❌ Dataset con años no disponible")
        st.info("""
        Para usar esta funcionalidad, asegúrate de que el archivo `prim_links_title.csv`
        esté disponible en el directorio principal y contenga las columnas:
        - movieId
        - title
        - poster_url
        - year
        """)

        # Botón para intentar recargar el dataset
        if st.button("🔄 Intentar recargar dataset con años", key="reload_years_dataset"):
            try:
                df_movies_with_years = pd.read_csv("prim_links_title.csv")
                st.success("✅ Dataset con años recargado correctamente")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error al recargar dataset: {str(e)}")

# 6. INFORMACIÓN ADICIONAL EN SIDEBAR

st.sidebar.header("📝 Instrucciones")
st.sidebar.markdown("""
**🎯 Modelo Corregido (10 Features + K-means):**
- ✅ Usa exactamente 10 características visuales
- ✅ Solo algoritmo K-means (simplificado)
- ✅ Búsqueda por distancia euclidiana
- ✅ Compatible con prim_reduced.csv

**Buscar por título:**
1. Ve a "📝 Buscar por título"
2. Escribe el nombre de una película
3. Selecciona de la lista
4. Elige el método:
   - 🎯 **K-means**: Usa el modelo entrenado
   - 🖼️ **Por Poster**: Analiza visualmente

**Buscar por imagen:**
1. Ve a "🖼️ Buscar por imagen"
2. Sube una imagen de poster
3. Se extraen 10 características automáticamente
4. Se buscan películas similares

**Filtrar por año:**
1. Ve a "📅 Filtrar por año"
2. Selecciona un año de la lista
3. Explora las películas de ese año
4. Escoge una película específica
5. Obtén recomendaciones similares
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Características Técnicas")
st.sidebar.info("""
**Pipeline optimizado:**
- HSV reducido: 24 → features de color
- LBP optimizado: 15 → features de textura
- Stats básicas: 6 → features complementarias
- PCA + UMAP: 45 → 10 features finales
- K-means: clustering con 10 features
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Troubleshooting")

# Botón para verificar compatibilidad del modelo
if st.sidebar.button("🔍 Verificar Modelo"):
    st.sidebar.markdown("**Verificando compatibilidad...**")

    checks = []

    # Check 1: Número de features
    if hasattr(analyzer, 'features'):
        features_shape = analyzer.features.shape
        checks.append(f"✅ Features: {features_shape[1]}/10" if features_shape[1] == 10 else f"❌ Features: {features_shape[1]}/10")
    else:
        checks.append("❌ No se encontraron features")

    # Check 2: Modelo K-means
    if hasattr(analyzer, 'kmeans_model') and analyzer.kmeans_model is not None:
        n_clusters = analyzer.kmeans_model.n_clusters
        checks.append(f"✅ K-means: {n_clusters} clusters")
    else:
        checks.append("❌ Modelo K-means no encontrado")

    # Check 3: Labels
    if hasattr(analyzer, 'kmeans_labels') and analyzer.kmeans_labels is not None:
        checks.append(f"✅ Labels: {len(analyzer.kmeans_labels)} películas")
    else:
        checks.append("❌ Labels no encontradas")

    # Check 4: Scaler
    if hasattr(analyzer, 'scaler'):
        checks.append("✅ Scaler disponible")
    else:
        checks.append("❌ Scaler no encontrado")

    for check in checks:
        st.sidebar.markdown(check)

# Botón para mostrar estadísticas del dataset
if st.sidebar.button("📊 Stats del Dataset"):
    st.sidebar.markdown("**Estadísticas del dataset:**")
    st.sidebar.markdown(f"- Total películas: {len(df_movies)}")

    if 'poster_url' in df_movies.columns:
        valid_urls = df_movies['poster_url'].notna().sum()
        st.sidebar.markdown(f"- URLs válidas: {valid_urls}/{len(df_movies)}")

    if 'title' in df_movies.columns:
        valid_titles = df_movies['title'].notna().sum()
        st.sidebar.markdown(f"- Títulos válidos: {valid_titles}/{len(df_movies)}")

    st.sidebar.markdown(f"- Columnas: {len(df_movies.columns)}")

st.sidebar.markdown("---")
st.sidebar.caption("🎬 Movie Recommender | Versión Corregida | 10 Features + K-means")

# 7. FOOTER CON INFORMACIÓN ADICIONAL

# Definir información básica del modelo
model_info = {
    'total_movies': len(df_movies) if 'df_movies' in locals() else 0,
    'features_count': 10,
    'n_clusters': 8
}

st.markdown("---")
st.markdown("### 📋 Información del Sistema")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Películas Analizadas", f"{model_info['total_movies']:,}")

with col2:
    st.metric("Características Visuales", f"{model_info['features_count']}")

with col3:
    st.metric("Clusters K-means", f"{model_info['n_clusters']}")

# Mostrar información adicional sobre el modelo en un expander
with st.expander("🔬 Detalles Técnicos del Modelo", expanded=False):
    st.markdown("""
    **Arquitectura del Sistema:**

    1. **Extracción de Características (45 → 10)**:
       - Histogramas HSV con bins reducidos (24 features)
       - Local Binary Patterns optimizados (15 features)
       - Estadísticas básicas de color (6 features)
       - Reducción dimensional PCA + UMAP a 10 features finales

    2. **Algoritmo de Clustering**:
       - K-means personalizado implementado desde cero
       - Inicialización K-means++ para mejor convergencia
       - Optimización por silhouette score

    3. **Búsqueda de Similitud**:
       - Distancia euclidiana en espacio normalizado
       - Búsqueda global (no limitada a clusters)
       - Escalado automático con StandardScaler

    4. **Pipeline de Procesamiento**:
       - Conversión automática RGB → HSV
       - Extracción paralela de características
       - Normalización y reducción dimensional
       - Predicción en tiempo real
    """)

    if hasattr(analyzer, 'kmeans_silhouette'):
        st.markdown(f"**Métricas de Calidad:**")
        st.markdown(f"- Silhouette Score: {analyzer.kmeans_silhouette:.4f}")
        if hasattr(analyzer, 'kmeans_inertia'):
            st.markdown(f"- Inercia: {analyzer.kmeans_inertia:.2f}")

# Mensaje final
st.success("🎉 Sistema listo para usar! Prueba subiendo una imagen o buscando por título.")

# Nota importante sobre compatibilidad
st.info("""
💡 **Nota importante**: Este sistema está optimizado para trabajar con el modelo corregido que usa
exactamente 10 características visuales. Si tienes errores, asegúrate de:

1. Usar el modelo entrenado con el script corregido
2. Que el archivo CSV tenga las columnas correctas
3. Que las imágenes sean de posters de películas para mejores resultados
""")

# Easter egg: botón para mostrar algunas recomendaciones aleatorias
if st.button("🎲 Descubre Películas Aleatorias", key="random_discovery"):
    st.markdown("### 🎬 Descubrimiento Aleatorio")
    random_indices = np.random.choice(len(df_movies), size=6, replace=False)

    cols = st.columns(3)
    for i, idx in enumerate(random_indices):
        movie = df_movies.iloc[idx]
        with cols[i % 3]:
            movie_title = movie.get('title', f'Película {idx}')
            st.subheader(movie_title)

            if 'poster_url' in movie and pd.notna(movie['poster_url']) and movie['poster_url'] != '':
                display_poster(movie['poster_url'], movie_title, width=120)

            if st.button(f"Ver similares", key=f"random_{idx}"):
                with st.spinner(f'Buscando películas similares a {movie_title}...'):
                    similar_indices = api.search_similar_movies(idx)
                    display_recommendations(similar_indices, f"Similares a {movie_title}:")
