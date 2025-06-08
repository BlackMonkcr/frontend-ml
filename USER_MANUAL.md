# Manual de Usuario - Sistema de Recomendación de Películas v2.0

## 🎬 Descripción General

El Sistema de Recomendación de Películas es una aplicación web avanzada que utiliza técnicas de Machine Learning y Computer Vision para encontrar películas visualmente similares basándose en sus posters.

### ✨ Características Principales

- **Análisis Visual Avanzado**: Extracción de características de color, textura y forma
- **Búsqueda Dual**: Por título de película o imagen de poster
- **Pipeline Completo**: HSV + LBP + Hu Moments + PCA + UMAP
- **Recomendaciones Inteligentes**: Distancia euclidiana para mayor precisión
- **Interfaz Intuitiva**: Diseño moderno con Streamlit

## 🚀 Instalación y Configuración

### Requisitos del Sistema
- Python 3.8+
- 4GB RAM mínimo
- Conexión a internet (para cargar posters)

### Dependencias
```bash
pip install streamlit pandas numpy scikit-learn opencv-python scikit-image umap-learn pillow requests
```

### Archivos Necesarios
```
frontend/
├── app.py                    # Aplicación principal
├── prim_reduced.csv         # Dataset de películas
├── trained_models/          # Modelos entrenados
│   └── analyzer_*.pkl
└── requirements.txt
```

### Formato del Dataset (prim_reduced.csv)
```csv
movieId,title,poster_url,poster_source
171751,Munna bhai M.B.B.S. (2003),https://image.tmdb.org/t/p/w185/nZNUTxGsSB4nLEC9Bpa2xfu81qV.jpg,tmdb_by_id
```

## 🎯 Guía de Uso

### Iniciar la Aplicación
```bash
cd frontend
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`

### Configuración Inicial

#### Panel Lateral (Sidebar)
1. **Ruta al modelo (.pkl)**: Ubicación del modelo entrenado
   - Por defecto: `trained_models/analyzer_solo_caracteristicas_visuales.pkl`

2. **Ruta a datos de películas (.csv)**: Dataset de películas
   - Por defecto: `prim_reduced.csv`

3. **Información Debug**: Verificación de datos cargados
   - Columnas disponibles
   - Total de películas
   - Verificación de URLs

## 📝 Métodos de Búsqueda

### 1. Búsqueda por Título

#### Paso a Paso:
1. **Navegar a la pestaña** "📝 Buscar por título"
2. **Escribir** el nombre de una película en el campo de texto
3. **Seleccionar** una película de la lista de coincidencias
4. **Elegir el método** de análisis:
   - 🎯 **Similares por Clustering**: Usa el modelo entrenado
   - 🖼️ **Similares por Poster**: Analiza visualmente el poster

#### Información Mostrada:
- **Poster**: Imagen del poster (si está disponible)
- **Movie ID**: Identificador único
- **Géneros**: Categorías de la película (si disponible)
- **Año**: Año de lanzamiento (si disponible)
- **Sinopsis**: Descripción de la película (si disponible)
- **Fuente del Poster**: Origen de la imagen

#### Resultados:
- **6 películas similares** organizadas en 3 columnas
- **Posters clickeables** con enlaces a imágenes originales
- **Información adicional** por cada recomendación

### 2. Búsqueda por Imagen

#### Paso a Paso:
1. **Navegar a la pestaña** "🖼️ Buscar por imagen"
2. **Subir una imagen** de poster (PNG, JPG, JPEG)
3. **Hacer clic** en "🔬 Buscar Películas Similares"
4. **Esperar** el procesamiento (análisis automático)
5. **Explorar** las recomendaciones

#### Proceso de Análisis:
1. **Extracción HSV**: 96 características de color
2. **Local Binary Patterns**: ~26 características de textura
3. **Hu Moments**: 7 características de forma
4. **Reducción PCA**: Primera optimización dimensional
5. **Reducción UMAP**: Optimización final
6. **Búsqueda**: Distancia euclidiana en espacio reducido

#### Información Debug:
- **Total características**: Número extraído
- **Primeras 10 características**: Valores específicos
- **Tiempo de procesamiento**: Indicadores de progreso

## 🔍 Interpretación de Resultados

### Calidad de Recomendaciones

#### Excelente Similitud (Distancia < 0.1)
- Posters muy similares visualmente
- Colores y composición parecidos
- Géneros probablemente relacionados

#### Buena Similitud (Distancia 0.1-0.3)
- Similitudes evidentes en color o composición
- Posible relación temática
- Estilo visual compatible

#### Similitud Moderada (Distancia 0.3-0.5)
- Algunas características compartidas
- Útil para exploración
- Puede revelar patrones interesantes

#### Similitud Baja (Distancia > 0.5)
- Pocas características compartidas
- Recomendaciones más exploratorias
- Útil para descubrir géneros nuevos

### Factores que Influyen en las Recomendaciones

#### Color (HSV):
- **Tono**: Paleta de colores dominante
- **Saturación**: Intensidad de los colores
- **Valor**: Brillo general de la imagen

#### Textura (LBP):
- **Patrones locales**: Distribución de píxeles
- **Rugosidad**: Complejidad visual
- **Repeticiones**: Elementos texturales

#### Forma (Hu Moments):
- **Contornos**: Formas principales
- **Distribución**: Organización espacial
- **Simetría**: Balance visual

## 🛠️ Funciones Avanzadas

### Debug y Diagnóstico

#### Información de Debug (Expandible):
- **Características extraídas**: Valores numéricos específicos
- **Tiempo de procesamiento**: Duración de cada etapa
- **Errores**: Mensajes de diagnóstico
- **Coincidencias de búsqueda**: Resultados de filtrado

#### Verificación de Datos:
- **Columnas del CSV**: Estructura del dataset
- **URLs válidas**: Disponibilidad de posters
- **Valores nulos**: Completitud de datos

### Ejemplos y Pruebas

#### Posters de Ejemplo:
- **Botón "Mostrar ejemplos"**: 6 posters del dataset
- **Imágenes de prueba**: Para validar funcionamiento
- **Enlaces directos**: A posters originales

#### Casos de Uso Típicos:
- **Descubrir películas similares**: A favoritas conocidas
- **Explorar géneros**: Mediante similitud visual
- **Validar clasificaciones**: Comparar con expectativas
- **Análisis de marketing**: Estudiar estrategias visuales

## ⚠️ Solución de Problemas

### Errores Comunes

#### "No se pudo cargar el modelo"
- **Verificar ruta**: Confirmar ubicación del archivo .pkl
- **Revisar permisos**: Acceso de lectura al archivo
- **Comprobar formato**: Modelo compatible con scikit-learn

#### "Error al cargar datos de películas"
- **Verificar CSV**: Formato y separadores correctos
- **Columnas requeridas**: movieId, title, poster_url
- **Codificación**: UTF-8 recomendado

#### "Poster no disponible"
- **URL inválida**: Enlace roto o inaccesible
- **Conexión a internet**: Verificar conectividad
- **Timeout**: Servidor de imágenes lento

#### "Error al procesar imagen"
- **Formato no soportado**: Solo PNG, JPG, JPEG
- **Imagen corrupta**: Archivo dañado
- **Tamaño excesivo**: Reducir resolución

### Optimización de Rendimiento

#### Para Datasets Grandes:
- **Limitar resultados**: Reducir número de recomendaciones
- **Cache**: Reutilizar características extraídas
- **Muestreo**: Usar subconjunto para pruebas

#### Para Imágenes Grandes:
- **Redimensionar**: Máximo 1024x1024 píxeles
- **Comprimir**: Reducir calidad JPEG
- **Formato**: Preferir PNG para logos simples

## 📊 Especificaciones Técnicas

### Pipeline de Características
```
Imagen → HSV (96) + LBP (~26) + Hu (7) → PCA (50) → UMAP (10) → Búsqueda
```

### Algoritmos Utilizados
- **HSV**: Histogramas en espacio de color HSV
- **LBP**: Local Binary Patterns (uniform)
- **Hu**: Momentos de Hu invariantes
- **PCA**: Análisis de Componentes Principales
- **UMAP**: Uniform Manifold Approximation

### Métricas de Similitud
- **Distancia Euclidiana**: En espacio reducido
- **Normalización**: StandardScaler
- **Ordenamiento**: Por distancia ascendente

## 🔄 Actualizaciones y Mantenimiento

### Actualizar Dataset:
1. **Reemplazar CSV**: Con nuevo archivo
2. **Verificar formato**: Columnas requeridas
3. **Reiniciar aplicación**: Para recargar datos

### Actualizar Modelo:
1. **Generar nuevo modelo**: Con pipeline actualizado
2. **Guardar como .pkl**: En carpeta trained_models/
3. **Actualizar ruta**: En configuración lateral

### Monitoreo:
- **Log de errores**: Streamlit console
- **Performance**: Tiempo de respuesta
- **Precisión**: Calidad de recomendaciones

---

## 🆘 Soporte

### Contacto:
- **Issues**: Reportar en repositorio
- **Documentación**: Consultar archivos README
- **Logs**: Revisar console de Streamlit

### Recursos Adicionales:
- **Pipeline.py**: Código de extracción
- **Changelog**: Historial de cambios
- **Examples**: Casos de uso documentados

**¡Disfruta explorando recomendaciones de películas visualmente similares!** 🎬🔍
