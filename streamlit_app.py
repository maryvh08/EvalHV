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

# Cargar las palabras clave y consejos desde los archivos JSON
def load_indicators(filepath="indicators.json"):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)
def load_advice(filepath="advice.json"):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)

# Cargar indicadores y consejos al inicio del script
indicators = load_indicators()
advice = load_advice()

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
    """
    Calcula la similitud entre dos textos usando TF-IDF y similitud de coseno.
    :param text1: Primer texto.
    :param text2: Segundo texto.
    :return: Porcentaje de similitud.
    """
    if not text1.strip() or not text2.strip():  # Evitar problemas con entradas vacías
        return 0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity * 100

def calculate_presence(texts, keywords):
    """
    Calcula el porcentaje de palabras clave presentes en los textos.
    :param texts: Lista de textos (e.g., detalles).
    :param keywords: Lista de palabras clave a buscar.
    :return: Porcentaje de coincidencia.
    """
    total_keywords = len(keywords)
    if total_keywords == 0:  # Evitar división por cero
        return 0

    matches = sum(1 for text in texts for keyword in keywords if keyword.lower() in text.lower())
    return (matches / total_keywords) * 100

# FUNCIONES PARA PRIMARY
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
        "nacional 20212023",
        "experiencia aneiap"
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


# FUNCIONES PARA SECUNDARY
def clean_text_for_pdf(text):
    """
    Limpia caracteres no soportados por FPDF reemplazándolos por equivalentes.
    :param text: Texto a limpiar.
    :return: Texto limpio.
    """
    replacements = {
        "\u2013": "-",  # Guion largo
        "\u2014": "-",  # Guion em dash
        "\u201c": "\"",  # Comillas dobles de apertura
        "\u201d": "\"",  # Comillas dobles de cierre
        "\u2018": "'",  # Comillas simples de apertura
        "\u2019": "'",  # Comillas simples de cierre
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return text

def extract_experience_items_with_details(pdf_path):
    """
    Extrae encabezados (en negrita) y sus detalles de la sección 'EXPERIENCIA EN ANEIAP'.
    :param pdf_path: Ruta del PDF.
    :return: Diccionario donde las claves son los encabezados y los valores son listas de detalles.
    """
    items = {}
    current_item = None
    in_experience_section = False  # Bandera para identificar si estamos dentro de la sección

    with fitz.open(pdf_path) as doc:
        for page in doc:
            blocks = page.get_text("dict")["blocks"]  # Extraer bloques de texto con formato

            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()

                        # Detectar el inicio y el fin de la sección
                        if text.lower().startswith("experiencia en aneiap"):
                            in_experience_section = True
                            continue
                        elif text.lower().startswith("reconocimientos") or text.lower().startswith("eventos organizados"):
                            in_experience_section = False
                            break

                        if not in_experience_section:
                            continue

                        # Detectar encabezados (negrita)
                        if "bold" in span["font"].lower():
                            current_item = text  # Nuevo encabezado detectado
                            items[current_item] = []  # Crear una lista vacía para los detalles
                        elif current_item:
                            # Agregar texto subsiguiente como detalle
                            items[current_item].append(text)

    return items

def analyze_items_and_details(items, position_indicators, functions_text, profile_text):
    """
    Analiza encabezados y viñetas según indicadores, funciones y perfil del cargo.
    :param items: Diccionario con encabezados y viñetas.
    :param position_indicators: Indicadores del cargo seleccionado.
    :param functions_text: Texto de las funciones del cargo.
    :param profile_text: Texto del perfil del cargo.
    :return: Diccionario con resultados del análisis.
    """
    results = {}
    for header, details in items.items():
        # Evaluar encabezado
        header_match = calculate_all_indicators([header], position_indicators)
        header_func_match = calculate_similarity(header, functions_text)
        header_profile_match = calculate_similarity(header, profile_text)

        # Evaluar viñetas
        detail_match = calculate_all_indicators(details, position_indicators)
        detail_func_match = sum(calculate_similarity(d, functions_text) for d in details) / max(len(details), 1)
        detail_profile_match = sum(calculate_similarity(d, profile_text) for d in details) / max(len(details), 1)

        # Consolidar resultados
        results[header] = {
            "header_match": header_match,
            "header_func_match": header_func_match,
            "header_profile_match": header_profile_match,
            "detail_match": detail_match,
            "detail_func_match": detail_func_match,
            "detail_profile_match": detail_profile_match,
        }

    return results

def get_critical_advice(critical_indicators, position):
    """
    Genera una lista de consejos basados en indicadores críticos.
    :param critical_indicators: Diccionario con los indicadores críticos y sus porcentajes.
    :param position: Cargo al que aspira el candidato.
    :return: Diccionario con los indicadores críticos y sus respectivos consejos.
    """
    critical_advice = {}

    for indicator in critical_indicators:
        # Obtener los consejos para el indicador crítico
        if position in advice and indicator in advice[position]:
            critical_advice[indicator] = advice[position][indicator]
        else:
            critical_advice[indicator] = ["No hay consejos disponibles para este indicador."]

    return critical_advice

def analyze_and_generate_descriptive_report(pdf_path, position, candidate_name, advice, indicators):
    """
    Analiza un CV descriptivo y genera un reporte PDF.
    :param pdf_path: Ruta del PDF.
    :param position: Cargo al que aspira.
    :param candidate_name: Nombre del candidato.
    :param advice: Diccionario con consejos.
    :param indicators: Diccionario con indicadores y palabras clave.
    """
    def clean_text_for_pdf(text):
        """
        Limpia caracteres no soportados por FPDF reemplazándolos por equivalentes.
        :param text: Texto a limpiar.
        :return: Texto limpio.
        """
        replacements = {
            "\u2013": "-",  # Guion largo
            "\u2014": "-",  # Guion em dash
            "\u201c": "\"",  # Comillas dobles de apertura
            "\u201d": "\"",  # Comillas dobles de cierre
            "\u2018": "'",  # Comillas simples de apertura
            "\u2019": "'",  # Comillas simples de cierre
        }
        for original, replacement in replacements.items():
            text = text.replace(original, replacement)
        return text

    # Extraer texto de la sección EXPERIENCIA EN ANEIAP
    items = extract_experience_items_with_details(pdf_path)
    if not items:
        st.error("No se encontraron encabezados y detalles para analizar.")
        return

    # Cargar funciones y perfil del cargo
    try:
        with fitz.open(f"Funciones//F{position}.pdf") as func_doc:
            functions_text = func_doc[0].get_text()
        with fitz.open(f"Perfiles//P{position}.pdf") as profile_doc:
            profile_text = profile_doc[0].get_text()
    except Exception as e:
        st.error(f"Error al cargar funciones o perfil: {e}")
        return

    # Filtrar indicadores correspondientes al cargo seleccionado
    position_indicators = indicators.get(position, {})
    if not position_indicators:
        st.error("No se encontraron indicadores para el cargo seleccionado.")
        return

    # Analizar encabezados y detalles
    item_results = {}
    related_items_count = {indicator: 0 for indicator in position_indicators}

    for header, details in items.items():
    header_and_details = f"{header} {' '.join(details)}"  # Combinar encabezado y detalles

    # Calcular concordancia con funciones y perfil utilizando palabras clave específicas
    func_match = (
        100
        if any(keyword.lower() in header_and_details.lower() for keyword in functions_text.split() if keyword in position_indicators)
        else calculate_similarity(header_and_details, functions_text)
    )
    profile_match = (
        100
        if any(keyword.lower() in header_and_details.lower() for keyword in profile_text.split() if keyword in position_indicators)
        else calculate_similarity(header_and_details, profile_text)
    )

    # Ignorar ítems con 0% en funciones y perfil
    if func_match == 0 and profile_match == 0:
        continue

    # Evaluar indicadores únicamente para el cargo seleccionado
    for indicator, keywords in position_indicators.items():
        if any(keyword.lower() in header_and_details.lower() for keyword in keywords):
            related_items_count[indicator] += 1

    item_results[header] = {
        "Funciones del Cargo": func_match,
        "Perfil del Cargo": profile_match,
    }

    # Calcular porcentajes de indicadores
    total_items = len(items)
    indicator_percentages = {
        indicator: (count / total_items) * 100 if total_items > 0 else 0 for indicator, count in related_items_count.items()
    }

    # Consejos para indicadores críticos (<50% de concordancia)
    critical_advice = {
        indicator: advice.get(position, {}).get(indicator, ["No hay consejos disponibles para este indicador."])
        for indicator, percentage in indicator_percentages.items() if percentage < 50
    }

    # Calcular concordancia global para funciones y perfil
    if item_results:
        global_func_match = sum(res["Funciones del Cargo"] for res in item_results.values()) / len(item_results)
        global_profile_match = sum(res["Perfil del Cargo"] for res in item_results.values()) / len(item_results)
    else:
        global_func_match = 0
        global_profile_match = 0

    # Calcular puntaje global
    func_score = round((global_func_match * 5) / 100, 2)
    profile_score = round((global_profile_match * 5) / 100, 2)

    # Crear el reporte PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Título del reporte
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(200, 10, txt=f"Reporte de Análisis Descriptivo - {candidate_name}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Cargo: {position}", ln=True, align='C')
    pdf.ln(10)

    # Resultados por ítem
    for header, result in item_results.items():
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, clean_text_for_pdf(f"Ítem: {header}"), ln=True)
        pdf.set_font("Arial", size=12)
        for key, value in result.items():
            pdf.cell(0, 10, f"- {key}: {value:.2f}%", ln=True)
        pdf.ln(5)

    # Total de líneas analizadas
    pdf.set_font("Arial", style="B", size=12)
    total_items = len(item_results)
    pdf.cell(0, 10, clean_text_for_pdf(f"Total de líneas analizadas: {total_items}"), ln=True)
    pdf.ln(5)

    # Resultados por indicadores
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, clean_text_for_pdf("Resultados por Indicadores:"), ln=True)
    pdf.set_font("Arial", size=12)
    for indicator, percentage in indicator_percentages.items():
        related_items = related_items_count[indicator]
        pdf.cell(0, 10, clean_text_for_pdf( f"- {indicator}: {percentage:.2f}% ({related_items} ítems relacionados)"), ln=True)
        if percentage < 50 and indicator in critical_advice:
            pdf.cell(0, 10, clean_text_for_pdf(f"  Consejos para {indicator}:"), ln=True)
            for tip in critical_advice[indicator]:
                pdf.cell(0, 10, f"    * {tip}", ln=True)
                
    pdf.ln(5)

    # Concordancia global
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Concordancia Global:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, clean_text_for_pdf(f"- Funciones del Cargo: {global_func_match:.2f}%"), ln=True)
    pdf.cell(0, 10, clean_text_for_pdf(f"- Perfil del Cargo: {global_profile_match:.2f}%"), ln=True)

    # Puntaje global
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Puntaje Global:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, clean_text_for_pdf(f"- Funciones del Cargo: {func_score}"), ln=True)
    pdf.cell(0, 10, clean_text_for_pdf(f"- Perfil del Cargo: {profile_score}"), ln=True)

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

    # Guardar el reporte
    describe_report_path = f"Reporte_Descriptivo_{candidate_name}_{position}.pdf"
    pdf.output(describe_report_path, 'F')

    # Descargar el reporte desde Streamlit
    with open(describe_report_path, "rb") as file:
        st.success("Reporte detallado generado exitosamente.")
        st.download_button(
            label="Descargar Reporte PDF",
            data=file,
            file_name=f"Reporte_Descriptivo_{candidate_name}_{position}.pdf",
            mime="application/pdf"
        )


# Interfaz en Streamlit
def home_page():
    st.title("Bienvenido a EvalHVUN")
    
    st.subheader("¿Qué tan listo estás para asumir un cargo de junta directiva Capitular? Descúbrelo aquí 🦁") 
    imagen_aneiap = 'Evaluador Hoja de Vida ANEIAP UNINORTE.jpg'
    st.image(imagen_aneiap, use_container_width=True)
    st.write("Esta herramienta analiza el contenido de la hoja de vida ANEIAP, comparandola con las funciones y perfil del cargo al que aspira, evaluando por medio de indicadores los aspectos puntuales en los cuales se hace necesario el aspirante enfatice para asegurar que este se encuentre preparado.") 
    st.write("Esta fue diseñada para apoyar en el proceso de convocatoria a los evaluadores para calificar las hojas de vida de los aspirantes.")
    st.write("Como resultado de este análisis se generará un reporte PDF descargable")
    
    st.write("---") 
    
    st.write("ℹ️ Aquí puedes encontrar información si quieres saber un poco más") 
    
    st.write("") 
    
    # Configuración del enlace CARGOS
    link_url_cargos = "https://drive.google.com/drive/folders/1hSUChvaYymUJ6g-IEfiY4hYqikePsQ9P?usp=drive_link"
    link_label_cargos = "Info cargos"
    
    # Configuración del enlace INDICADORES
    link_url_indicadores = "https://docs.google.com/document/d/1BM07wuVaXEWcdurTRr8xBzjsB1fiWt6wGqOzLiyQBs8/edit?usp=drive_link"
    link_label_indicadores = "Info indicadores"
    
    # Contenedor para centrar los botones
    st.markdown(f"""
        <div style="display: flex; justify-content: center; gap: 20px;">
            <a href="{link_url_cargos}" target="_blank" style="text-decoration:none;">
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
                    {link_label_cargos}
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
    La herramienta tiene disponible dos versiones, de modo se pueda evaluar la HV con el formato actual y una propuesta para incluir descripciones de los proyectos/cargos ocupados.
    </div>
    """,
    unsafe_allow_html=True
    )


def primary():
    imagen_primary= 'Analizador Versión Actual.jpg'
    st.title("Evaluador de Hoja de Vida ANEIAP")
    st.image(imagen_primary, use_container_width=True)
    st.subheader("Versión Actual Hoja de Vida ANEIAP▶️")
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
    - Es preferible que la HV no haya sido cambiada de formato varias veces, ya que esto puede complicar la lectura y extracción del texto.
    - La EXPERIENCIA EN ANEIAP debe estar enumerada para facilitar el análisis de la misma.
    - El análisis puede presentar inconsistencias si la HV no está debidamente separada en subtítulos.
    - Si la sección de EXPERIENCIA EN ANEIAP está dispuesta como tabla, la herramienta puede fallar.
    """)
    
    st.write("---")
    
    st.markdown(
    """
    <div style="text-align: center; font-weight: bold; font-size: 20px;">
    ⚠️ DISCLAIMER: LA INFORMACIÓN PROPORCIONADA POR ESTA HERRAMIENTA NO REPRESENTA NINGÚN TIPO DE DECISIÓN, SU FIN ES MERAMENTE ILUSTRATIVO
    </div>
    """,
    unsafe_allow_html=True
    )
    
def secondary():
    st.title("Evaluador de Hoja de Vida ANEIAP - Versión Descriptiva")
    st.write("Sube tu hoja de vida ANEIAP (en formato PDF) para evaluar tu perfil.")

    # Entrada de datos del usuario
    candidate_name = st.text_input("Nombre del candidato:")
    uploaded_file = st.file_uploader("Sube tu hoja de vida ANEIAP en formato PDF", type="pdf")
    position = st.selectbox("Selecciona el cargo al que aspiras:", [
        "DCA", "DCC", "DCD", "DCF", "DCM", "CCP", "IC", "PC"
    ])

    if st.button("Generar Reporte"):
        if uploaded_file is not None:
            with open("uploaded_cv.pdf", "wb") as f:
                f.write(uploaded_file.read())
            
            # Llamar a la nueva función unificada
            analyze_and_generate_descriptive_report("uploaded_cv.pdf", position, candidate_name, advice, indicators)


        else:
            st.error("Por favor, sube un archivo PDF para continuar.")


    st.write(f"---")

    st.subheader("Recomendaciones a tener en cuenta ✅")
    st.markdown("""
    - Organiza tu HV en formato descriptivo para cada cargo o proyecto.
    - Usa viñetas para detallar las acciones realizadas en cada ítem.
    - Evita usar tablas para la sección de experiencia, ya que esto dificulta la extracción de datos.
    """)
    st.write("---")

    st.markdown(
        """
        <div style="text-align: center; font-weight: bold; font-size: 20px;">
        Plantilla Propuesta HV 📑
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    imagen_plantilla = 'PLANTILLA PROPUESTA HV ANEIAP.jpg'
    st.image(imagen_plantilla, use_container_width=True)

    link_url_plantilla = "https://drive.google.com/drive/folders/16i35reQpBq9eC2EuZfy6E6Uul5XVDN8D?usp=sharing"
    link_label_plantilla = "Descargar plantilla"

    st.markdown(f"""
        <div style="display: flex; justify-content: center; gap: 20px;">
            <a href="{link_url_plantilla}" target="_blank" style="text-decoration:none;">
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
                    {link_label_plantilla}
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("---")

    st.markdown(
        """
        <div style="text-align: center; font-weight: bold; font-size: 20px;">
        ⚠️ DISCLAIMER: LA INFORMACIÓN PROPORCIONADA POR ESTA HERRAMIENTA NO REPRESENTA NINGÚN TIPO DE DECISIÓN, SU FIN ES MERAMENTE ILUSTRATIVO
        </div>
        """,
        unsafe_allow_html=True
    )

# Diccionario de páginas
pages = {
    "🏠 Inicio": home_page,
    "✳️ Versión actual": primary,
    "🚀 Analizador descriptivo": secondary,
}

# Sidebar para seleccionar página
st.sidebar.title("Menú")
selected_page = st.sidebar.radio("Ir a", list(pages.keys()))

# Renderiza la página seleccionada
pages[selected_page]()
