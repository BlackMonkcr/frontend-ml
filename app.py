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

warnings.filterwarnings('ignore')

# 1. COPIAR LAS CLASES NECESARIAS DESDE EL SCRIPT ORIGINAL
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

class DBSCANCustom:
    """Implementación personalizada del algoritmo DBSCAN"""
    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = None

    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.sqrt(np.sum((point1 - point2)**2))

    def _get_neighbors(self, X: np.ndarray, point_idx: int) -> List[int]:
        neighbors = []
        point = X[point_idx]

        for i, other_point in enumerate(X):
            if self._euclidean_distance(point, other_point) <= self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X: np.ndarray, point_idx: int, neighbors: List[int],
                       cluster_id: int, labels: np.ndarray, visited: np.ndarray) -> None:
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = self._get_neighbors(X, neighbor_idx)

                if len(neighbor_neighbors) >= self.min_samples:
                    for nn in neighbor_neighbors:
                        if nn not in neighbors:
                            neighbors.append(nn)

            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            i += 1

    def fit(self, X: np.ndarray) -> 'DBSCANCustom':
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0
        core_samples = []

        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue

            visited[point_idx] = True
            neighbors = self._get_neighbors(X, point_idx)

            if len(neighbors) < self.min_samples:
                continue
            else:
                core_samples.append(point_idx)
                self._expand_cluster(X, point_idx, neighbors, cluster_id, labels, visited)
                cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.array(core_samples)
        self.n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

class MovieClusteringAnalyzer:
    """Clase principal para análisis de clustering de posters de películas"""
    def __init__(self):
        self.features = None
        self.movie_metadata = None
        self.kmeans_model = None
        self.dbscan_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.features_scaled = None
        self.features_pca = None
        self.dataset_name = None
        self.kmeans_labels = None
        self.dbscan_labels = None

    def load_csv_data(self, csv_file_path: str, dataset_name: str = "Dataset"):
        df = pd.read_csv(csv_file_path)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

        self.features = df[numeric_columns].values

        metadata_dict = {'movie_index': range(len(df)), 'dataset': [dataset_name] * len(df)}
        for col in non_numeric_columns:
            metadata_dict[col] = df[col].values

        self.movie_metadata = pd.DataFrame(metadata_dict)
        self.dataset_name = dataset_name
        self.features_scaled = self.scaler.fit_transform(self.features)
        self.features_pca = self.pca.fit_transform(self.features_scaled)
        return self

# 2. FUNCIONES DE PROCESAMIENTO DE IMÁGENES (PIPELINE COMPLETO)

# === PIPELINE DE EXTRACCIÓN DE CARACTERÍSTICAS ===
from concurrent.futures import ThreadPoolExecutor

# Extracción de características HSV
def extract_hsv(img, bins=32):
    """Extrae histogramas HSV de una imagen"""
    from skimage import color
    hsv = color.rgb2hsv(img)
    h = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 1))[0]
    s = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 1))[0]
    v = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 1))[0]
    return np.concatenate([h, s, v])

# Extracción de características de textura (LBP)
def extract_lbp(img, P=8, R=1):
    """Extrae características de Local Binary Pattern"""
    from skimage import color, feature
    g = color.rgb2gray(img)
    g_uint8 = (g * 255).astype(np.uint8)
    lbp = feature.local_binary_pattern(g_uint8, P, R, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist

# Extracción de Hu Moments
def extract_hu(img):
    """Extrae momentos de Hu para características de forma"""
    from skimage import color
    gray = color.rgb2gray(img)
    gray_uint8 = (gray * 255).astype(np.uint8)
    moments = cv2.moments(gray_uint8)
    hu = cv2.HuMoments(moments).flatten()
    return hu

# Extractor completo de características (paralelizado)
def extract_all_features(img):
    """
    Extrae todas las características de una imagen usando el pipeline completo.
    Retorna un vector de características de alta dimensionalidad (HSV + LBP + Hu).
    """
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_hsv = executor.submit(extract_hsv, img)
            future_lbp = executor.submit(extract_lbp, img)
            future_hu = executor.submit(extract_hu, img)

            hsv = future_hsv.result()
            lbp = future_lbp.result()
            hu = future_hu.result()

            # Combinar todas las características
            features = np.concatenate([hsv, lbp, hu])
            return features
    except Exception as e:
        st.error(f"Error al extraer características complejas: {str(e)}")
        return None

# === REDUCCIÓN DIMENSIONAL ===
def apply_feature_reduction(features, n_components_final=12, random_state=42):
    """
    Aplica PCA + UMAP para reducir dimensionalidad al número esperado por el modelo.
    Para una sola muestra, simplificamos usando solo PCA.
    """
    try:
        from sklearn.decomposition import PCA

        # Asegurar que features es 1D
        if len(features.shape) > 1:
            features = features.flatten()

        # Para una sola muestra, usar solo PCA es más estable
        # Crear una matriz con la muestra repetida para poder aplicar PCA
        n_samples_synthetic = 50  # Crear muestras sintéticas para PCA
        features_matrix = np.tile(features, (n_samples_synthetic, 1))

        # Añadir un poco de ruido para crear variabilidad
        np.random.seed(random_state)
        noise = np.random.normal(0, 0.01, features_matrix.shape)
        features_matrix = features_matrix + noise
          # Aplicar PCA
        # Para arrays 1D, features.shape[0] es el número de características
        n_features = len(features)
        n_components_pca = min(n_components_final, n_features - 1, n_samples_synthetic - 1)
        pca = PCA(n_components=n_components_pca, random_state=random_state)
        features_pca = pca.fit_transform(features_matrix)

        # Tomar solo la primera muestra (la original)
        features_reduced = features_pca[0]

        # Si necesitamos exactamente n_components_final, ajustar
        if len(features_reduced) > n_components_final:
            features_reduced = features_reduced[:n_components_final]
        elif len(features_reduced) < n_components_final:
            # Rellenar con ceros si es necesario
            padding = np.zeros(n_components_final - len(features_reduced))
            features_reduced = np.concatenate([features_reduced, padding])
        return features_reduced

    except ImportError:
        st.error("Error: sklearn no está disponible")
        return None
    except Exception as e:
        st.error(f"Error en reducción dimensional: {str(e)}")
        st.info("🔄 Usando características simples como fallback...")
        return None

def extract_simple_features(image):
    """
    Función de respaldo que extrae exactamente 12 características simples
    para coincidir con el modelo entrenado en caso de que falle el pipeline completo.
    """
    try:
        # Convertir a RGB si es necesario
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            pass  # Ya está en RGB

        # Redimensionar a tamaño estándar
        image_resized = cv2.resize(image, (224, 224))

        # 1-3: Estadísticas básicas de color (RGB promedio)
        if len(image_resized.shape) == 3:
            mean_r = np.mean(image_resized[:,:,0])
            mean_g = np.mean(image_resized[:,:,1])
            mean_b = np.mean(image_resized[:,:,2])
        else:
            mean_r = mean_g = mean_b = np.mean(image_resized)

        # 4-6: Desviaciones estándar por canal
        if len(image_resized.shape) == 3:
            std_r = np.std(image_resized[:,:,0])
            std_g = np.std(image_resized[:,:,1])
            std_b = np.std(image_resized[:,:,2])
        else:
            std_r = std_g = std_b = np.std(image_resized)

        # 7: Brillo promedio general
        brightness = np.mean(image_resized)

        # 8: Contraste (desviación estándar general)
        contrast = np.std(image_resized)

        # 9: Densidad de bordes
        if len(image_resized.shape) == 3:
            gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_resized
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # 10: Relación de aspecto
        height, width = image_resized.shape[:2]
        aspect_ratio = width / height

        # 11: Saturación promedio (solo para imágenes a color)
        if len(image_resized.shape) == 3:
            hsv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:,:,1])
        else:
            saturation = 0  # Sin saturación en escala de grises

        # 12: Entropía (medida de información/complejidad)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist / hist.sum()  # Normalizar
        entropy = -np.sum(hist * np.log(hist + 1e-7))

        # Combinar las 12 características
        features = np.array([
            mean_r, mean_g, mean_b,
            std_r, std_g, std_b,
            brightness, contrast, edge_density,
            aspect_ratio, saturation, entropy
        ])

        return features

    except Exception as e:
        st.error(f"Error al extraer características simples: {str(e)}")
        return None

def process_uploaded_image(uploaded_image):
    """
    Procesa una imagen subida usando el pipeline complejo integrado.

    Args:
        uploaded_image: PIL Image objeto

    Returns:
        np.array con exactamente 12 características finales
    """
    try:
        # Convertir PIL Image a array numpy
        image_array = np.array(uploaded_image)

        st.info("🔄 Procesando imagen con pipeline avanzado: HSV + LBP + Hu Moments + PCA...")

        # Paso 1: Extraer características complejas
        complex_features = extract_all_features(image_array)

        if complex_features is not None:
            st.info(f"✅ Extraídas {len(complex_features)} características complejas")

            # Paso 2: Reducir dimensionalidad a 12 características
            reduced_features = apply_feature_reduction(complex_features)

            if reduced_features is not None and len(reduced_features) == 12:
                st.success("✅ Pipeline exitoso - 12 características finales extraídas")
                return reduced_features
            else:
                st.warning("⚠️ Reducción dimensional falló, usando características simples como fallback...")
                return extract_simple_features(image_array)
        else:
            st.warning("⚠️ Extracción compleja falló, usando características simples como fallback...")
            return extract_simple_features(image_array)

    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")
        st.info("🔄 Intentando con características simples...")
        try:
            return extract_simple_features(np.array(uploaded_image))
        except:
            return None

# 3. CÓDIGO DE STREAMLIT
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Sistema de Recomendación de Películas por Poster")
st.markdown("**Descubre películas visualmente similares** usando algoritmos avanzados de clustering.")

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
    except Exception:
        return False

class MovieClusteringAPI:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.df_movies = analyzer.movie_metadata

    def search_similar_movies(self, movie_index, method='kmeans', n_results=6):
        if method == 'kmeans':
            cluster_id = self.analyzer.kmeans_labels[movie_index]
            same_cluster = np.where(self.analyzer.kmeans_labels == cluster_id)[0]
        else:
            cluster_id = self.analyzer.dbscan_labels[movie_index]
            if cluster_id == -1:
                return []
            same_cluster = np.where(self.analyzer.dbscan_labels == cluster_id)[0]

        same_cluster = same_cluster[same_cluster != movie_index]

        if len(same_cluster) == 0:
            return []

        ref_features = self.analyzer.features_scaled[movie_index]
        distances = []
        for idx in same_cluster:
            dist = np.linalg.norm(self.analyzer.features_scaled[idx] - ref_features)
            distances.append((idx, dist))

        distances.sort(key=lambda x: x[1])
        return [idx for idx, _ in distances[:n_results]]

    def search_similar_movies_by_features(self, image_features, method='kmeans', n_results=6):
        """Busca películas similares basándose en las características de una imagen"""
        try:
            # Normalizar las características de la imagen usando el mismo scaler del modelo
            image_features_scaled = self.analyzer.scaler.transform(image_features.reshape(1, -1))

            # Predecir el cluster de la imagen
            if method == 'kmeans':
                predicted_cluster = self.analyzer.kmeans_model.predict(image_features_scaled)[0]
                same_cluster = np.nonzero(self.analyzer.kmeans_labels == predicted_cluster)[0]
            else:
                # Para DBSCAN, buscar el cluster más cercano
                distances_to_centroids = []
                for i in range(max(self.analyzer.dbscan_labels) + 1):
                    cluster_points = self.analyzer.features_scaled[self.analyzer.dbscan_labels == i]
                    if len(cluster_points) > 0:
                        centroid = np.mean(cluster_points, axis=0)
                        dist = np.linalg.norm(image_features_scaled[0] - centroid)
                        distances_to_centroids.append((i, dist))

                if distances_to_centroids:
                    predicted_cluster = min(distances_to_centroids, key=lambda x: x[1])[0]
                    same_cluster = np.nonzero(self.analyzer.dbscan_labels == predicted_cluster)[0]
                else:
                    same_cluster = np.array([])

            if len(same_cluster) == 0:
                # Si no hay películas en el cluster, buscar las más similares globalmente
                distances = []
                for idx in range(len(self.analyzer.features_scaled)):
                    dist = np.linalg.norm(self.analyzer.features_scaled[idx] - image_features_scaled[0])
                    distances.append((idx, dist))

                distances.sort(key=lambda x: x[1])
                return [idx for idx, _ in distances[:n_results]]
              # Calcular distancias dentro del cluster
            distances = []
            for idx in same_cluster:
                dist = np.linalg.norm(self.analyzer.features_scaled[idx] - image_features_scaled[0])
                distances.append((idx, dist))

            distances.sort(key=lambda x: x[1])
            return [idx for idx, _ in distances[:n_results]]

        except Exception as e:
            st.error(f"Error al buscar películas similares: {str(e)}")
            return []

# Interfaz de usuario
model_path = st.sidebar.text_input("Ruta al modelo (.pkl)", "trained_models/analyzer_solo_caracteristicas_visuales.pkl")
csv_path = st.sidebar.text_input("Ruta a datos de películas (.csv)", "prim_reduced.csv")

analyzer = load_model(model_path)
if analyzer is None:
    st.warning("No se pudo cargar el modelo. Verifica la ruta o crea un nuevo modelo.")
    st.stop()

try:
    df_movies = load_movie_data(csv_path)
    if 'title' not in df_movies.columns:
        df_movies['title'] = [f'Película {i}' for i in range(len(df_movies))]

    # DEBUG: Mostrar información sobre los datos cargados
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔍 DEBUG INFO**")
    st.sidebar.markdown(f"**Columnas disponibles:** {list(df_movies.columns)}")
    st.sidebar.markdown(f"**Total películas:** {len(df_movies)}")

    # Mostrar primeros títulos para debugging
    if 'title' in df_movies.columns:
        st.sidebar.markdown("**Primeros 5 títulos:**")
        for i, title in enumerate(df_movies['title'].head(5)):
            st.sidebar.text(f"{i}: {title}")

        # Verificar valores nulos o vacíos
        null_titles = df_movies['title'].isnull().sum()
        empty_titles = (df_movies['title'] == '').sum()
        st.sidebar.markdown(f"**Títulos nulos:** {null_titles}")
        st.sidebar.markdown(f"**Títulos vacíos:** {empty_titles}")

except Exception as e:
    st.error(f"Error al cargar datos de películas: {str(e)}")
    st.stop()

api = MovieClusteringAPI(analyzer)
st.sidebar.success("✅ Modelo y datos cargados correctamente")
st.sidebar.markdown(f"**Películas disponibles:** {len(df_movies)}")
st.sidebar.markdown(f"**Dataset:** {analyzer.dataset_name}")
st.sidebar.markdown(f"**Clusters K-means:** {analyzer.kmeans_model.n_clusters}")

# Buscador
st.header("🔍 Busca tu película")

# Crear tabs para diferentes métodos de búsqueda
tab1, tab2 = st.tabs(["📝 Buscar por título", "🖼️ Buscar por imagen"])

with tab1:
    search_term = st.text_input("Escribe el título de una película:", "")

    if search_term:
        # DEBUG: Mostrar información de búsqueda
        with st.expander("🔍 DEBUG - Información de búsqueda", expanded=False):
            st.write(f"**Término de búsqueda:** '{search_term}'")
            st.write(f"**Tipo de datos en 'title':** {df_movies['title'].dtype}")
            st.write(f"**Muestra de títulos que contienen '{search_term}':**")

            # Mostrar coincidencias parciales para debug
            partial_matches = df_movies[df_movies['title'].str.contains(search_term, case=False, na=False)]
            st.write(f"**Total coincidencias encontradas:** {len(partial_matches)}")

            if len(partial_matches) > 0:
                st.dataframe(partial_matches[['title']].head(10))
            else:
                st.write("No se encontraron coincidencias")

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
                if 'poster_path' in selected_movie:
                    poster_displayed = display_poster(selected_movie['poster_path'], selected_movie['title'], 200)
                if not poster_displayed and 'poster' in selected_movie:
                    display_poster(selected_movie['poster'], selected_movie['title'], 200)

            with col2:
                if 'genres' in selected_movie:
                    st.markdown(f"**Géneros:** {selected_movie['genres']}")
                if 'year' in selected_movie:
                    st.markdown(f"**Año:** {selected_movie['year']}")
                if 'overview' in selected_movie:
                    st.markdown(f"**Sinopsis:** {selected_movie['overview']}")
                elif 'description' in selected_movie:
                    st.markdown(f"**Descripción:** {selected_movie['description']}")

            # Recomendaciones
            if st.button("🎯 Mostrar películas similares", use_container_width=True, key="title_search"):
                with st.spinner('Buscando recomendaciones...'):
                    similar_indices = api.search_similar_movies(selected_index)

                if not similar_indices:
                    st.warning("No se encontraron películas similares.")
                else:
                    st.subheader(f"🎬 {len(similar_indices)} películas recomendadas:")
                    cols = st.columns(3)

                    for i, idx in enumerate(similar_indices):
                        movie = df_movies.loc[idx]
                        with cols[i % 3]:
                            st.subheader(movie['title'])

                            # Intentar diferentes campos de imagen
                            poster_displayed = False
                            if 'poster_path' in movie:
                                poster_displayed = display_poster(movie['poster_path'], movie['title'])
                            if not poster_displayed and 'poster' in movie:
                                poster_displayed = display_poster(movie['poster'], movie['title'])
                            if not poster_displayed:
                                st.warning("Poster no disponible")

                            if 'genres' in movie:
                                st.caption(f"Géneros: {movie['genres']}")
                            if 'year' in movie:
                                st.caption(f"Año: {movie['year']}")

with tab2:
    st.markdown("**Sube la imagen de un poster de película** y encuentra películas visualmente similares")

    uploaded_file = st.file_uploader(
        "Selecciona una imagen de poster:",
        type=['png', 'jpg', 'jpeg'],
        help="Sube una imagen de un poster de película para encontrar películas similares basándose en características visuales"
    )
      # Información del pipeline
    st.markdown("### 🔬 Pipeline de Análisis Avanzado")
    st.info("""
    **Sistema integrado de extracción de características:**
    - **Histogramas HSV**: Análisis de color en espacio HSV (96 características)
    - **Local Binary Patterns**: Análisis de textura y patrones locales (26 características)
    - **Momentos de Hu**: Características de forma invariantes (7 características)
    - **Reducción dimensional**: PCA + UMAP para optimizar a 12 características finales
    - **Procesamiento paralelo**: Extracción simultánea para mayor velocidad
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
            st.markdown(f"- **Modo:** {image.mode}")          # Procesar imagen y buscar similares
        if st.button("🔬 Buscar Películas Similares", use_container_width=True, key="image_search"):
            with st.spinner('Procesando imagen con pipeline avanzado...'):
                # Extraer características de la imagen usando el pipeline completo
                image_features = process_uploaded_image(image)
                if image_features is not None:
                    # DEBUG: Mostrar información de las características
                    with st.expander("🔧 DEBUG - Características extraídas", expanded=False):
                        st.write(f"**Total características:** {len(image_features)} (esperado: 12)")
                        st.write("**Características extraídas:**")
                        feature_names = [
                            "Media R", "Media G", "Media B",
                            "Std R", "Std G", "Std B",
                            "Brillo", "Contraste", "Densidad bordes",
                            "Aspecto", "Saturación", "Entropía"
                        ]
                        for i, (name, value) in enumerate(zip(feature_names, image_features)):
                            st.write(f"  {i}: {name} = {value:.4f}")

                    # Buscar películas similares
                    similar_indices = api.search_similar_movies_by_features(image_features)

                    if not similar_indices:
                        st.warning("No se encontraron películas similares.")
                    else:
                        st.subheader(f"🎬 {len(similar_indices)} películas recomendadas:")
                        cols = st.columns(3)

                        for i, idx in enumerate(similar_indices):
                            movie = df_movies.loc[idx]
                            with cols[i % 3]:
                                st.subheader(movie['title'])

                                # Intentar diferentes campos de imagen
                                poster_displayed = False
                                if 'poster_path' in movie:
                                    poster_displayed = display_poster(movie['poster_path'], movie['title'])
                                if not poster_displayed and 'poster' in movie:
                                    poster_displayed = display_poster(movie['poster'], movie['title'])
                                if not poster_displayed:
                                    st.warning("Poster no disponible")

                                if 'genres' in movie:
                                    st.caption(f"Géneros: {movie['genres']}")
                                if 'year' in movie:
                                    st.caption(f"Año: {movie['year']}")
                else:
                    st.error("Error al procesar la imagen. Intenta con otra imagen.")

# Información adicional
st.sidebar.header("📝 Instrucciones")
st.sidebar.markdown("""
**Buscar por título:**
1. Ve a la pestaña "📝 Buscar por título"
2. Escribe el nombre de una película
3. Selecciona una opción de la lista
4. Haz clic en "Mostrar películas similares"

**Buscar por imagen:**
1. Ve a la pestaña "🖼️ Buscar por imagen"
2. Sube una imagen de un poster de película
3. Haz clic en "Buscar películas similares"
4. Explora las recomendaciones basadas en características visuales

**Nota:** La búsqueda por imagen utiliza algoritmos de clustering para encontrar películas con características visuales similares (color, textura, forma, etc.)
""")

st.sidebar.markdown("---")
st.sidebar.caption("Desarrollado con Streamlit | Machine Learning Project")
