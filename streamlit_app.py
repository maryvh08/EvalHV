import fitz  # PyMuPDF para trabajar con PDFs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from fpdf import FPDF
from collections import Counter
import pytesseract
from PIL import Image
import io
import re
import json

# Cargar las palabras clave desde el archivo JSON
def load_indicators(filepath="indicators.json"):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)

# Cargar indicadores al inicio del script
indicators = load_indicators()

advice = {
    "DCA": {
        "Diseño académico": [
            "Desarrolla capacidades pedagógicas para diseñar programas de formación académica.",
            "Aprende a coordinar eventos académicos de alto impacto como talleres y olimpiadas.",
            "Promueve la interdisciplinaridad en las actividades académicas."
        ],
        "Innovación e investigación": [
            "Domina herramientas de investigación para generar contenido relevante y actualizado.",
            "Fortalece tus conocimientos en innovación educativa y herramientas tecnológicas.",
            "Domina la gestión de talento para identificar y apoyar a asociados destacados.",
            "Fomenta la colaboración con instituciones académicas externas."
        ],
        "Formación y capacitación": [
            "Fomenta la integración del entorno académico con los objetivos de ANEIAP.",
            "Aprende a diseñar sistemas de evaluación de impacto académico.",
            "Asegura que las actividades estén alineadas con los valores y objetivos del capítulo."
        ]
    },
    "DCC": {
        "Estrategia de comunicación": [
            "Desarrolla habilidades de redacción y storytelling para fortalecer la marca ANEIAP.",
            "Gestiona relaciones públicas para ampliar la visibilidad de los proyectos capitulares.",
            "Aprende a monitorear y analizar métricas de comunicación para evaluar el impacto.",
            "Trabaja en estrategias de fidelización de asociados mediante campañas comunicativas."
        ],
        "Producción audiovisual": [
            "Aprende a diseñar estrategias de comunicación eficaces para captar la atención de asociados y externos.",
            "Domina herramientas de diseño gráfico y edición audiovisual para generar contenido atractivo.",
            "Fomenta la interacción digital mediante plataformas sociales y blogs.",
            "Crea planes de contenido que estén alineados con los objetivos del capítulo."
        ],
        "Gestión de documental": [
            "Participa en iniciativas capitulares relacionadas a la gestión documental.",
            "Infórmate acerca de la gestión documental y el manejo de los datos a nivel nacional. "
        ]
    },
    "DCD": {
        "Gestión de asociados": [
            "Fomenta la integración y permanencia de los asociados mediante eventos y actividades.",
            "Aprende a diseñar planes de reclutamiento y retención efectivos.",
            "Coordina procesos de incorporación de nuevos asociados mediante estrategias claras."
        ],
        "Integración y bienestar": [
            "Implementa sistemas de reconocimiento como incentivos y galas de premios.",
            "Fortalece tus habilidades en la organización de eventos de integración.",
            "Trabaja en estrategias para medir la satisfacción de los asociados."
        ],
        "Sostenimiento y sociedad": [
            "Domina las herramientas de gestión de clima organizacional.",
            "Desarrolla programas que fomenten la responsabilidad social y ambiental.",
            "Aprende a gestionar la comunicación con asociados para mantenerlos comprometidos.",
            "Asegúrate de mantener un enfoque centrado en las personas y sus necesidades."
        ]
    },
    "DCF": {
        "Gestión financiera": [
            "Aprende a diseñar presupuestos y controlar el flujo de caja del capítulo.",
            "Domina la elaboración de informes financieros claros y precisos.",
            "Implementa sistemas para la gestión de cuentas por pagar y cobrar."
        ],
        "Sostenibilidad económica": [
            "Gestiona actividades de obtención de recursos de manera eficiente.",
            "Fortalece tus conocimientos en análisis financiero y proyecciones económicas.",
            "Fortalece tus habilidades en la planeación de sostenibilidad financiera.",
            "Desarrolla indicadores de gestión financiera para evaluar el desempeño del capítulo."
        ],
        "Análisis  y transparencia": [
            "Aprende a negociar con proveedores y patrocinadores.",
            "Asegúrate de documentar todas las operaciones financieras con transparencia.",
            "Trabaja en estrategias para diversificar las fuentes de ingresos."
        ]
    },
    "DCM": {
        "Estrategias de branding": [
            "Aprende a diseñar estrategias de branding para posicionar la marca ANEIAP.",
            "Crea planes de fidelización para mantener asociados comprometidos.",
            "Fomenta la innovación en productos y servicios ofrecidos por el capítulo."
        ],
        "Promoción y visibilidad": [
            "Domina herramientas de análisis de mercado para identificar necesidades de los asociados.",
            "Implementa sistemas de CRM para gestionar relaciones con asociados y aliados.",
            "Coordina la organización de eventos para promover la participación de asociados.",
            "Mide el impacto de las campañas y ajusta estrategias según los resultados."
        ],
        "Gestión comercial": [
            "Aprende a manejar campañas publicitarias en redes sociales y otros medios.",
            "Trabaja en estrategias de gestión de alianzas estratégicas con empresas.",
            "Mantén un enfoque en la sostenibilidad y responsabilidad social en todas las iniciativas."
        ]
    },
      "PC": {
        "Liderazgo y estrategia": [
            "Desarrolla habilidades intrapersonales como autogestión emocional y disciplina para liderar de forma efectiva.",
            "Fomenta el liderazgo y la comunicación asertiva para relacionarte eficientemente con las instancias internas y externas.",
            "Fortalece tus habilidades en planeación estratégica para dirigir proyectos y alcanzar objetivos.",
            "Promueve la cohesión del equipo a través de motivación y delegación responsable."
        ],
        "Gestión organizacional": [
            "Conoce y comprende a fondo los procesos operativos y normativos de ANEIAP.",
            "Aprende a gestionar recursos de manera eficiente para asegurar la sostenibilidad del capítulo.",
            "Mejora tu capacidad para resolver conflictos y adaptarte a los cambios en el entorno organizacional.",
            "Sé un ejemplo de integridad y transparencia en todas las gestiones realizadas."
        ],
        "Relaciones y representación": [
            "Fortalece la representación de la asociación en entes externos mediante estrategias de visibilidad.",
            "Practica la moderación de asambleas para guiar decisiones clave del capítulo."
        ]
    },
    "CCP": {
        "Gestión de proyectos": [
            "Domina las metodologías de gestión de proyectos para asegurar la correcta ejecución de los mismos.",
            "Aprende a identificar y mitigar riesgos asociados a los proyectos.",
            "Promueve el desarrollo de indicadores de éxito para evaluar los proyectos implementados.",
            "Define sistemas de seguimiento continuo para asegurar la alineación con los objetivos del capítulo."
        ],
        "Innovación y creatividad": [
            "Identifica y prioriza objetivos de innovación en los proyectos para aumentar su impacto.",
            "Fomenta la creatividad en los equipos para diseñar proyectos innovadores.",
            "Comprende la importancia de la sostenibilidad en cada proyecto que líderes."
        ],
        "Colaboración estratégica": [
            "Desarrolla habilidades en la formación de equipos multidisciplinarios para proyectos complejos.",
            "Fortalece la creación de alianzas estratégicas con patrocinadores y socios.",
            "Trabaja en habilidades de negociación para garantizar recursos y apoyo."
        ]
    },
    "IC": {
        "Auditoría y control": [
            "Domina las funciones de veeduría, asesoría y control interno en los proyectos del capítulo.",
            "Aprende a utilizar herramientas de auditoría y evaluación de proyectos.",
            "Participa activamente en el seguimiento del cumplimiento de los objetivos del capítulo.",
            "Asegura que toda documentación esté alineada con los estándares de ANEIAP."
        ],
        "Normativa y transparencia": [
            "Fortalece tu capacidad para analizar e interpretar normativas asociativas.",
            "Fomenta la transparencia en todos los procesos de la asociación.",
            "Adquiere experiencia en la moderación de conflictos internos."
        ],
        "Seguimiento y evaluación": [
            "Desarrolla habilidades para emitir conceptos imparciales y objetivos sobre situaciones complejas.",
            "Aprende a coordinar reuniones estratégicas para abordar desviaciones en los proyectos.",
            "Fortalece la representación de intereses de asociados ante la Junta Directiva."
        ]
    }
}

def preprocess_image(img):
    """
    Aplica preprocesamiento a una imagen para mejorar la precisión de OCR.
    """
    img = img.convert("L")  # Convertir a escala de grises
    img = img.filter(ImageFilter.MedianFilter())  # Reducir ruido
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)  # Mejorar contraste
    return img

def extract_text_with_ocr(pdf_path):
    """
    Extrae texto de un PDF utilizando OCR con preprocesamiento.
    :param pdf_path: Ruta del archivo PDF.
    :return: Texto extraído del PDF.
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            # Intentar extraer texto con PyMuPDF
            page_text = page.get_text()
            if not page_text.strip():  # Si no hay texto, usar OCR
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes(output="png")))
                img = preprocess_image(img)  # Preprocesar imagen
                page_text = pytesseract.image_to_string(img, config="--psm 6")  # Configuración personalizada
            text += page_text
    return text

def extract_experience_section_with_ocr(pdf_path):
    """
    Extrae la sección 'EXPERIENCIA EN ANEIAP' de un archivo PDF con soporte de OCR.
    :param pdf_path: Ruta del archivo PDF.
    :return: Texto de la sección 'EXPERIENCIA EN ANEIAP'.
    """
    text = extract_text_with_ocr(pdf_path)
    
    # Palabras clave para identificar el inicio y final de la sección
    start_keyword = "EXPERIENCIA EN ANEIAP"
    end_keywords = [
        "EVENTOS ORGANIZADOS", 
        "Reconocimientos individuales", 
        "Reconocimientos", 
        "Reconocimientos grupales"
        "Reconocimientos"
    ]
    
    # Encontrar índice de inicio
    start_idx = text.lower().find(start_keyword.lower())
    if start_idx == -1:
        return None  # No se encontró la sección

    # Encontrar índice más cercano de fin basado en palabras clave
    end_idx = len(text)  # Por defecto, tomar hasta el final
    for keyword in end_keywords:
        idx = text.lower().find(keyword.lower(), start_idx)
        if idx != -1:
            end_idx = min(end_idx, idx)

    # Extraer la sección entre inicio y fin
    experience_text = text[start_idx:end_idx].strip()

    # Lista de renglones a excluir 
    exclude_lines = [
        "a nivel capitular",
        "a nivel nacional",
        "a nivel seccional",
        "reconocimientos individuales",
        "reconocimientos grupales",
        "nacional 2024",
        "cargos",
        "trabajo capitular",
        "trabajo nacional",
        "actualización profesional",
        "nacional 2021-2023",
        "nacional 20212023"
    ]
    
    experience_lines = experience_text.split("\n")
    cleaned_lines = []
    for line in experience_lines:
        line = line.strip()
        line = re.sub(r"[^\w\s]", "", line)  # Eliminar caracteres no alfanuméricos excepto espacios
        normalized_line = re.sub(r"\s+", " ", line).lower()  # Normalizar espacios y convertir a minúsculas
        if (
            normalized_line
            and normalized_line not in exclude_lines
            and normalized_line != start_keyword.lower()
            and normalized_line not in [kw.lower() for kw in end_keywords]
        ):
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
    
    # Debugging: Imprime líneas procesadas
    print("Líneas procesadas:")
    for line in cleaned_lines:
        print(f"- {line}")
    
    return "\n".join(cleaned_lines)

def extract_cleaned_lines(text):
    """
    Limpia y filtra las líneas de un texto.
    :param text: Texto a procesar.
    :return: Lista de líneas limpias.
    """
    lines = text.split("\n")
    return [line.strip() for line in lines if line.strip()]  # Eliminar líneas vacías y espacios.

def calculate_all_indicators(lines, position_indicators):
    """
    Calcula los porcentajes de todos los indicadores para un cargo.
    :param lines: Lista de líneas de la sección "EXPERIENCIA EN ANEIAP".
    :param position_indicators: Diccionario de indicadores y palabras clave del cargo.
    :return: Diccionario con los porcentajes por indicador.
    """
    total_lines = len(lines)
    if total_lines == 0:
        return {indicator: 0 for indicator in position_indicators}  # Evitar división por cero

    indicator_results = {}
    for indicator, keywords in position_indicators.items():
        relevant_lines = sum(
            any(keyword.lower() in line.lower() for keyword in keywords) for line in lines
        )
        indicator_results[indicator] = (relevant_lines / total_lines) * 100  # Cálculo del porcentaje
    return indicator_results

def calculate_indicators_for_report(lines, position_indicators):
    """
    Calcula los porcentajes de relevancia de indicadores para el reporte.
    :param lines: Lista de líneas de la sección "EXPERIENCIA EN ANEIAP".
    :param position_indicators: Diccionario de indicadores y palabras clave del cargo.
    :return: Diccionario con los porcentajes por indicador y detalles de líneas relevantes.
    """
    total_lines = len(lines)
    if total_lines == 0:
        return {indicator: {"percentage": 0, "relevant_lines": 0} for indicator in position_indicators}

    indicator_results = {}
    for indicator, keywords in position_indicators.items():
        relevant_lines = sum(
            any(keyword.lower() in line.lower() for keyword in keywords) for line in lines
        )
        percentage = (relevant_lines / total_lines) * 100
        indicator_results[indicator] = {"percentage": percentage, "relevant_lines": relevant_lines}

    return indicator_results

# Función para calcular la similitud usando TF-IDF y similitud de coseno
def calculate_similarity(text1, text2):
    """Calcula la similitud entre dos textos usando TF-IDF y similitud de coseno."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity * 100

def calculate_presence(lines, keywords):
    """
    Calcula el porcentaje de renglones que contienen palabras clave específicas.
    :param lines: Lista de renglones a evaluar.
    :param keywords: Lista de palabras clave a buscar.
    :return: Porcentaje de renglones que contienen al menos una palabra clave.
    """
    matched_lines = sum(
        any(keyword.lower() in line.lower() for keyword in keywords) for line in lines
    )
    return (matched_lines / len(lines)) * 100 if lines else 0

def generate_report(pdf_path, position, candidate_name):
    """Genera un reporte en PDF basado en la comparación de la hoja de vida con funciones, perfil e indicadores."""
    experience_text = extract_experience_section_with_ocr(pdf_path)
    if not experience_text:
        st.error("No se encontró la sección 'EXPERIENCIA EN ANEIAP' en el PDF.")
        return

    # Dividir la experiencia en líneas
    lines = extract_cleaned_lines(experience_text)
    lines= experience_text.split("\n")
    lines = [line.strip() for line in lines if line.strip()]  # Eliminar líneas vacías

    # Obtener los indicadores y palabras clave para el cargo seleccionado
    position_indicators = indicators.get(position, {})

    indicator_results = calculate_all_indicators(lines, position_indicators)

    # Cargar funciones y perfil
    try:
        with fitz.open(f"Funciones//F{position}.pdf") as func_doc:
            functions_text = func_doc[0].get_text()
        with fitz.open(f"Perfiles/P{position}.pdf") as profile_doc:
            profile_text = profile_doc[0].get_text()
    except Exception as e:
        st.error(f"Error al cargar funciones o perfil: {e}")
        return

    line_results = []

    # Evaluación de renglones
    for line in lines:
        line = line.strip()
        if not line:  # Ignorar líneas vacías
            continue

        # Dividir la experiencia en líneas
        lines = extract_cleaned_lines(experience_text)
        lines = experience_text.split("\n")
        lines = [line.strip() for line in lines if line.strip()]  # Eliminar líneas vacías
    
        # Obtener los indicadores y palabras clave para el cargo seleccionado
        position_indicators = indicators.get(position, {})
        indicator_results = {}

        # Calcular el porcentaje por cada indicador
        indicator_results = calculate_indicators_for_report(lines, position_indicators)
        for indicator, keywords in position_indicators.items():
            indicator_results = calculate_indicators_for_report(lines, position_indicators)

        # Calcular la presencia total (si es necesario)
        total_presence = sum(result["percentage"] for result in indicator_results.values())

        # Normalizar los porcentajes si es necesario
        if total_presence > 0:
            for indicator in indicator_results:
                indicator_results[indicator]["percentage"] = (indicator_results[indicator]["percentage"] / total_presence) * 100

        # Evaluación general de concordancia
        if any(keyword.lower() in line.lower() for kw_set in position_indicators.values() for keyword in kw_set):
            func_match = 100.0
            profile_match = 100.0
        else:
            # Calcular similitud 
            func_match = calculate_similarity(line, functions_text)
            profile_match = calculate_similarity(line, profile_text)
        
        # Solo agregar al reporte si no tiene 0% en ambas métricas
        if func_match > 0 or profile_match > 0:
            line_results.append((line, func_match, profile_match))

    # Normalización de los resultados de indicadores
    total_presence = sum(indicator["percentage"] for indicator in indicator_results.values())
    if total_presence > 0:
        for indicator in indicator_results:
            indicator_results[indicator]["percentage"] = (indicator_results[indicator]["percentage"] / total_presence) * 100
            
    # Cálculo de concordancia global
    if line_results:  # Evitar división por cero si no hay ítems válidos
        global_func_match = sum([res[1] for res in line_results]) / len(line_results)
        global_profile_match = sum([res[2] for res in line_results]) / len(line_results)
    else:
        global_func_match = 0
        global_profile_match = 0

    #Calculo puntajes
    func_score = round((global_func_match * 5) / 100, 2)
    profile_score = round((global_profile_match * 5) / 100, 2)
    
    # Crear reporte en PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Times", size=12)

    def clean_text(text):
        """Reemplaza caracteres no compatibles con latin-1."""
        return text.encode('latin-1', 'replace').decode('latin-1')
    
    # Título del reporte
    pdf.set_font("Helvetica", style="B", size=14)  
    pdf.cell(200, 10, txt=f"Reporte de Concordancia de {candidate_name} para el cargo de {position}", ln=True, align='C')
    
    pdf.ln(3)
    
    #Concordancia de items
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Análisis de items:", ln=True)
    pdf.set_font("Arial", style="", size=12)
    for line, func_match, profile_match in line_results:
        pdf.multi_cell(0, 10, clean_text(f"Item: {line}"))
        pdf.multi_cell(0, 10, clean_text(f"- Concordancia con funciones: {func_match:.2f}%"))
        pdf.multi_cell(0, 10, clean_text( f"- Concordancia con perfil: {profile_match:.2f}%"))

     # Total de líneas analizadas
    pdf.set_font("Arial", style="B", size=12)
    total_lines = len(lines)
    pdf.cell(0, 10, f"Total de líneas analizadas: {total_lines}", ln=True)
    pdf.ln(5)

    # Resultados de indicadores
    pdf.cell(0, 10, "Resultados por Indicadores:", ln=True)
    pdf.set_font("Arial", size=12)
    for indicator, result in indicator_results.items():
        relevant_lines = result["relevant_lines"]
        percentage = (relevant_lines / total_lines) * 100 if total_lines > 0 else 0
        pdf.cell(0, 10, f"- {indicator}: {percentage:.2f}% ({relevant_lines} items relacionados)", ln=True)

    # Indicador con menor presencia
    lowest_indicator = min(indicator_results, key=lambda k: indicator_results[k]["relevant_lines"])
    pdf.ln(5)
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Indicador con Menor Presencia:", ln=True)
    pdf.set_font("Arial", size=12)
    lowest_relevant_lines = indicator_results[lowest_indicator]["relevant_lines"]
    lowest_percentage = (lowest_relevant_lines / total_lines) * 100 if total_lines > 0 else 0
    pdf.cell(0, 10, f"{lowest_indicator} ({lowest_percentage:.2f}%)", ln=True)

    # Consejos para mejorar indicadores con baja presencia
    low_performance_indicators = {k: v for k, v in indicator_results.items() if (v["relevant_lines"] / total_lines) * 100 < 50.0}
    if low_performance_indicators:
        pdf.ln(5)
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, "Consejos para Mejorar:", ln=True)
        pdf.set_font("Arial", size=12)
        for indicator, result in low_performance_indicators.items():
            percentage = (result["relevant_lines"] / total_lines) * 100 if total_lines > 0 else 0
            pdf.cell(0, 10, f"- {indicator}: ({percentage:.2f}%)", ln=True)
            for tip in advice[position].get(indicator, []):
                pdf.multi_cell(0, 10, f"  * {tip}")
    
    #Concordancia global
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Concordancia Global:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"La concordancia Global de Funciones es: {global_func_match:.2f}%", ln=True)
    pdf.cell(0, 10, f"La concordancia Global de Perfil es: {global_profile_match:.2f}%", ln=True)

    #Puntaje global
    pdf.set_font("Arial", style="B", size=12)
    pdf.multi_cell(0, 10, "\nPuntaje Global:")
    pdf.set_font("Arial", style="", size=12)
    pdf.multi_cell(0,10, f"- El puntaje respecto a las funciones de cargo es: {func_score}")
    pdf.multi_cell(0,10, f"- El puntaje respecto al perfil de cargo es: {profile_score}")

    # Interpretación de resultados
    pdf.set_font("Arial", style="B", size=12)
    pdf.multi_cell(0, 10, "\nInterpretación de resultados:")
    pdf.set_font("Arial", style="", size=12)
    if global_profile_match >75 and global_func_match > 75:
        pdf.multi_cell(0, 10, f"- Alta Concordancia (> 0.75): El análisis revela que {candidate_name} tiene una excelente adecuación con las funciones del cargo de {position} y el perfil buscado. La experiencia detallada en su hoja de vida está estrechamente alineada con las responsabilidades y competencias requeridas para este rol crucial en la prevalencia del Capítulo. La alta concordancia indica que {candidate_name} está bien preparado para asumir este cargo y contribuir significativamente al éxito y la misión del Capítulo. Se recomienda proceder con el proceso de selección y considerar a {candidate_name} como una opción sólida para el cargo.")
    
    elif 50 < global_profile_match < 75 and 50 < global_func_match < 75:
        pdf.multi_cell(0, 10, f"- Buena Concordancia (> 0.50): El análisis muestra que {candidate_name} tiene una buena correspondencia con las funciones del cargo de {position} y el perfil deseado. Aunque su experiencia en la asociación es relevante, existe margen para mejorar. {candidate_name} muestra potencial para cumplir con el rol crucial en la prevalencia del Capítulo, pero se recomienda que continúe desarrollando sus habilidades y acumulando más experiencia relacionada con el cargo objetivo. Su candidatura debe ser considerada con la recomendación de enriquecimiento adicional.")
        
    else:
        pdf.multi_cell(0, 10, f"- Baja Concordancia (< 0.50): El análisis indica que {candidate_name} tiene una baja concordancia con los requisitos del cargo de {position} y el perfil buscado. Esto sugiere que aunque el aspirante posee algunas experiencias relevantes, su historial actual no cubre adecuadamente las competencias y responsabilidades necesarias para este rol crucial en la prevalencia del Capítulo. Se aconseja a {candidate_name} enfocarse en mejorar su perfil profesional y desarrollar las habilidades necesarias para el cargo. Este enfoque permitirá a {candidate_name} alinear mejor su perfil con los requisitos del puesto en futuras oportunidades.")

    # Conclusión
    pdf.multi_cell(0, 10, f"Este análisis es generado debido a que es crucial tomar medidas estratégicas para garantizar que  los candidatos estén bien preparados para el rol de {position}. Los aspirantes con alta concordancia deben ser considerados seriamente para el cargo, ya que están en una posición favorable para asumir responsabilidades significativas y contribuir al éxito del Capítulo. Aquellos con buena concordancia deberían continuar desarrollando su experiencia, mientras que los aspirantes con  baja concordancia deberían recibir orientación para mejorar su perfil profesional y acumular más  experiencia relevante. Estas acciones asegurarán que el proceso de selección se base en una evaluación completa y precisa de las capacidades de cada candidato, fortaleciendo la gestión y el  impacto del Capítulo.")
    
    # Mensaje de agradecimiento
    pdf.cell(0, 10, f"Muchas gracias {candidate_name} por tu interés en convertirte en {position}. ¡Éxitos en tu proceso!")

    # Guardar PDF
    report_path = f"Reporte_analisis_cargo_{position}_{candidate_name}.pdf"
    pdf.output(report_path, 'F')

    st.success("Reporte generado exitosamente.")
    st.download_button(
        label="Descargar Reporte",
        data=open(report_path, "rb"),
        file_name=report_path,
        mime="application/pdf"
    )

# Interfaz en Streamlit
imagen_aneiap = 'Evaluador Hoja de Vida ANEIAP UNINORTE.jpg'
st.title("Evaluador de Hoja de Vida ANEIAP")
st.image(imagen_aneiap, use_container_width=True)
st.subheader("¿Qué tan listo estás para asumir un cargo de junta directiva Capitular? Descúbrelo aquí 🦁")
st.write("Sube tu hoja de vida ANEIAP (en formato PDF) para evaluar tu perfil.")

# Entrada de datos del usuario
candidate_name = st.text_input("Nombre del candidato:")
uploaded_file = st.file_uploader("Sube tu hoja de vida ANEIAP en formato PDF", type="pdf")
position = st.selectbox("Selecciona el cargo al que aspiras:", [
    "DCA", "DCC", "DCD", "DCF", "DCM", "CCP", "IC", "PC"
])

# Configuración BOTÓN GENERAR REPORTE
if st.button("Generar Reporte"):
    if uploaded_file is not None:
        with open("uploaded_cv.pdf", "wb") as f:
            f.write(uploaded_file.read())
        generate_report("uploaded_cv.pdf", position, candidate_name)
    else:
        st.error("Por favor, sube un archivo PDF para continuar.")

st.write(f"---")

st.subheader("Recomendaciones a tener en cuenta ✅")
st.markdown("""
- Asegurate de que tu HV no haya sido cambiada de formato varias veces, esto puede complicar la lectura y extracción del texto.
- Describir tu EXPERIENCIA EN ANEIAP en forma de viñetas facilitará el análisis de la misma.
- Evita que en tu HV no tenga separado el subtítulo "EXPERIENCIA EN ANEIAP" del contenido de esta sección para evitar inconsistencias en el análisis.
- Evita utilizar tablas en tu sección de EXPERIENCIA EN ANEIAP para un mejor análisis de la información.
""")

st.write("") 

st.write("ℹ️ Aquí puedes encontrar información si quieres saber un poco más") 

st.write("") 

# Configuración del enlace MANUALES
link_url_Manuales = "https://drive.google.com/drive/folders/18OIh99ZxE1LThqzy1A406f1kbot6b4bf"
link_label_Manuales = "Manuales de cargo"

# Configuración del enlace INDICADORES
link_url_indicadores = "https://docs.google.com/document/d/1BM07wuVaXEWcdurTRr8xBzjsB1fiWt6wGqOzLiyQBs8/edit?usp=drive_link"
link_label_indicadores = "Info indicadores"

# Contenedor para centrar los botones
st.markdown(f"""
    <div style="display: flex; justify-content: center; gap: 20px;">
        <a href="{link_url_Manuales}" target="_blank" style="text-decoration:none;">
            <button style="
                background-color: #F1501B;
                border: none;
                color: white;
                padding: 12px 24px;
                text-align: center;
                font-size: 16px;
                cursor: pointer;
                border-radius: 4px;
            ">
                {link_label_Manuales}
            </button>
        </a>
        <a href="{link_url_indicadores}" target="_blank" style="text-decoration:none;">
            <button style="
                background-color: #F1501B;
                border: none;
                color: white;
                padding: 12px 24px;
                text-align: center;
                font-size: 16px;
                cursor: pointer;
                border-radius: 4px;
            ">
                {link_label_indicadores}
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

st.markdown(
"""
<div style="text-align: center; font-weight: bold; font-size: 20px;">
DISCLAIMER: LA INFORMACIÓN PROPORCIONADA POR ESTA HERRAMIENTA NO REPRESENTA NINGÚN TIPO DE DECISIÓN, SU FIN ES MERAMENTE ILUSTRATIVO
</div>
""",
unsafe_allow_html=True
)
