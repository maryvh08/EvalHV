import fitz
import numpy as np
import spacy
import pandas as pd
import streamlit as st
from io import BytesIO
from textstat import textstat
import requests
import tarfile
import io
import re
import json
import os
import pytesseract
from spellchecker import SpellChecker
from textblob import TextBlob
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageTemplate, Frame, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import statsmodels.api as sm
from spellchecker import SpellChecker
import re
from PIL import Image as PILImage
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

#Link de la página https://evalhv-uvgdqtpnuheurqmrzdnnnb.streamlit.app/

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

# Uso del código
background_path = "Fondo Comunicado.png"

def preprocess_image(image):
    """
    Preprocesa una imagen antes de aplicar OCR.
    :param image: Imagen a preprocesar.
    :return: Imagen preprocesada.
    """
    # Convertir la imagen a escala de grises
    image = image.convert("L")

    # Mejorar el contraste
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Aplicar umbral para binarización
    image = ImageOps.autocontrast(image)

    return image

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

# Definir función para añadir fondo
def add_background(canvas, background_path):
    """
    Dibuja una imagen de fondo en cada página del PDF.
    :param canvas: Lienzo de ReportLab.
    :param background_path: Ruta a la imagen de fondo.
    """
    canvas.saveState()
    canvas.drawImage(background_path, 0, 0, width=letter[0], height=letter[1])
    canvas.restoreState()

# FUNCIONES PARA PRIMARY
def extract_profile_section_with_ocr(pdf_path):
    """
    Extrae la sección 'Perfil' de un archivo PDF con soporte de OCR.
    :param pdf_path: Ruta del archivo PDF.
    :return: Texto de la sección 'Perfil'.
    """
    text = extract_text_with_ocr(pdf_path)  # Extraer texto completo del PDF utilizando OCR

    # Palabras clave para identificar el inicio y fin de la sección
    start_keyword = "Perfil"
    end_keywords = [
        "Asistencia a eventos",
        "Actualización profesional",
    ]

    # Encontrar índice de inicio
    start_idx = text.lower().find(start_keyword.lower())
    if start_idx == -1:
        return None  # No se encontró la sección "Perfil"

    # Encontrar índice más cercano de fin basado en palabras clave
    end_idx = len(text)  # Por defecto, tomar hasta el final
    for keyword in end_keywords:
        idx = text.lower().find(keyword.lower(), start_idx)
        if idx != -1:
            end_idx = min(end_idx, idx)

    # Extraer la sección entre inicio y fin
    profile_text = text[start_idx:end_idx].strip()

    # Limpieza adicional del texto
    cleaned_profile_text = re.sub(r"[^\w\s.,;:]", "", profile_text)  # Eliminar caracteres no deseados
    cleaned_profile_text = re.sub(r"\s+", " ", cleaned_profile_text)  # Normalizar espacios

    return cleaned_profile_text

def extract_experience_section_with_ocr(pdf_path):
    """
    Extrae la sección 'EXPERIENCIA EN ANEIAP' de un archivo PDF con soporte de OCR.
    :param pdf_path: Ruta del archivo PDF.
    :return: Texto de la sección 'EXPERIENCIA EN ANEIAP'.
    """
    text = extract_text_with_ocr(pdf_path)

    # Palabras clave para identificar inicio y fin de la sección
    start_keyword = "EXPERIENCIA EN ANEIAP"
    end_keywords = [
        "EVENTOS ORGANIZADOS",
        "Reconocimientos individuales",
        "Reconocimientos grupales",
        "Reconocimientos",
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

    # Filtrar y limpiar texto
    exclude_lines = [
        "a nivel capitular",
        "a nivel nacional",
        "a nivel seccional",
        "reconocimientos individuales",
        "reconocimientos grupales",
        "trabajo capitular",
        "trabajo nacional",
        "nacional 2024",
        "nacional 20212023",
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

def extract_event_section_with_ocr(pdf_path):
    """
    Extrae la sección 'EVENTOS ORGANIZADOS' de un archivo PDF con soporte de OCR.
    :param pdf_path: Ruta del archivo PDF.
    :return: Texto de la sección 'EVENTOS ORGANIZADOS'.
    """
    text = extract_text_with_ocr(pdf_path)

    # Palabras clave para identificar inicio y fin de la sección
    start_keyword = "EVENTOS ORGANIZADOS"
    end_keywords = [
        "EXPERIENCIA LABORAL",
        "FIRMA",
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
    org_text = text[start_idx:end_idx].strip()

    # Filtrar y limpiar texto
    org_exclude_lines = [
        "a nivel capitular",
        "a nivel nacional",
        "a nivel seccional",
        "capitular",
        "seccional",
        "nacional",
    ]
    org_lines = org_text.split("\n")
    org_cleaned_lines = []
    for line in org_lines:
        line = line.strip()
        line = re.sub(r"[^\w\s]", "", line)  # Eliminar caracteres no alfanuméricos excepto espacios
        normalized_org_line = re.sub(r"\s+", " ", line).lower()  # Normalizar espacios y convertir a minúsculas
        if (
            normalized_org_line
            and normalized_org_line not in org_exclude_lines
            and normalized_org_line != start_keyword.lower()
            and normalized_org_line not in [kw.lower() for kw in end_keywords]
        ):
            org_cleaned_lines.append(line)

    return "\n".join(org_cleaned_lines)
    
    # Debugging: Imprime líneas procesadas
    print("Líneas procesadas:")
    for line in org_cleaned_lines:
        print(f"- {line}")
    
    return "\n".join(org_cleaned_lines)

def evaluate_cv_presentation(pdf_path):
    """
    Evalúa la presentación de la hoja de vida en términos de redacción, ortografía,
    coherencia básica, y claridad.
    :param pdf_path: Ruta del archivo PDF.
    :return: Texto limpio y análisis detallado de la presentación.
    """
    # Extraer texto completo de la hoja de vida
    resume_text = extract_text_with_ocr(pdf_path)

    if not resume_text:
        return None, "No se pudo extraer el texto de la hoja de vida."

    # Limpiar y filtrar texto
    pres_cleaned_lines = []
    lines = resume_text.split("\n")
    for line in lines:
        line = line.strip()
        line = re.sub(r"[^\w\s.,;:!?-]", "", line)  # Eliminar caracteres no alfanuméricos excepto signos básicos
        line = re.sub(r"\s+", " ", line)  # Normalizar espacios
        if line:
            pres_cleaned_lines.append(line)

    # Evaluación de calidad de presentación
    total_lines = len(pres_cleaned_lines)
    if total_lines == 0:
        return None, "El documento está vacío o no contiene texto procesable."
        
    return "\n".join(pres_cleaned_lines)

def extract_attendance_section_with_ocr(pdf_path):
    """
    Extrae la sección 'Asistencia Eventos ANEIAP' de un archivo PDF con soporte de OCR.
    :param pdf_path: Ruta del archivo PDF.
    :return: Texto de la sección 'Asistencia Eventos ANEIAP'.
    """
    text = extract_text_with_ocr(pdf_path)

    # Palabras clave para identificar inicio y fin de la sección
    start_keyword = "ASISTENCIA A EVENTOS ANEIAP"
    end_keywords = [
        "ACTUALIZACIÓN PROFESIONAL",
        "EXPERIENCIA EN ANEIAP",
        "EVENTOS ORGANIZADOS",
        "RECONOCIMIENTOS",
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
    att_text = text[start_idx:end_idx].strip()

    # Filtrar y limpiar texto
    att_exclude_lines = [
        "a nivel capitular",
        "a nivel nacional",
        "a nivel seccional",
        "capitular",
        "seccional",
        "nacional",
    ]
    att_lines = att_text.split("\n")
    att_cleaned_lines = []
    for line in att_lines:
        line = line.strip()
        line = re.sub(r"[^\w\s]", "", line)  # Eliminar caracteres no alfanuméricos excepto espacios
        normalized_att_line = re.sub(r"\s+", " ", line).lower()  # Normalizar espacios y convertir a minúsculas
        if (
            normalized_att_line
            and normalized_att_line not in att_exclude_lines
            and normalized_att_line != start_keyword.lower()
            and normalized_att_line not in [kw.lower() for kw in end_keywords]
        ):
            att_cleaned_lines.append(line)

    return "\n".join(att_cleaned_lines)
    
    # Debugging: Imprime líneas procesadas
    print("Líneas procesadas:")
    for line in att_cleaned_lines:
        print(f"- {line}")
    
    return "\n".join(att_cleaned_lines)

def generate_report_with_background(pdf_path, position, candidate_name,background_path):
    """
    Genera un reporte con un fondo en cada página.
    :param pdf_path: Ruta del PDF.
    :param position: Cargo al que aspira.
    :param candidate_name: Nombre del candidato.
    :param background_path: Ruta de la imagen de fondo.
    """
    experience_text = extract_experience_section_with_ocr(pdf_path)
    if not experience_text:
        st.error("No se encontró la sección 'EXPERIENCIA EN ANEIAP' en el PDF.")
        return

    org_text = extract_event_section_with_ocr(pdf_path)
    if not org_text:
        st.error("No se encontró la sección 'EVENTOS ORGANIZADOS' en el PDF.")
        return

    att_text = extract_attendance_section_with_ocr(pdf_path)
    if not att_text:
        st.error("No se encontró la sección 'Asistencia a Eventos ANEIAP' en el PDF.")
        return

    resume_text= evaluate_cv_presentation(pdf_path)
    if not resume_text:
        st.error("No se encontró el texto de la hoja de vida")
        return
    
    # Dividir la experiencia en líneas
    lines = extract_cleaned_lines(experience_text)
    lines= experience_text.split("\n")
    lines = [line.strip() for line in lines if line.strip()]  # Eliminar líneas vacías

    # Dividir los eventos en líneas
    org_lines = extract_cleaned_lines(org_text)#Extraer texto del perfil
    profile_text = extract_profile_section_with_ocr(pdf_path)
    if not profile_text:
        st.warning("No se encontró la sección 'Perfil' en el PDF.")
        return None
    org_lines= org_text.split("\n")
    org_lines = [line.strip() for line in org_lines if line.strip()]  # Eliminar líneas vacías

    # Dividir la asistencia en líneas
    att_lines = extract_cleaned_lines(att_text)
    att_lines= att_text.split("\n")
    att_lines = [line.strip() for line in att_lines if line.strip()]  # Eliminar líneas vacías

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
    org_line_results = []
    att_line_results = []

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

    # Evaluación de renglones eventos organizados
    for line in org_lines:
        line = line.strip()
        if not line:  # Ignorar líneas vacías
            continue

        # Dividir los eventos en líneas
        org_lines = extract_cleaned_lines(org_text)
        org_lines= org_text.split("\n")
        org_lines = [line.strip() for line in org_lines if line.strip]

        # Evaluación general de concordancia
        if any(keyword.lower() in line.lower() for kw_set in position_indicators.values() for keyword in kw_set):
            org_func_match = 100.0
            org_profile_match = 100.0
        else:
            # Calcular similitud
            org_func_match = calculate_similarity(line, functions_text)
            org_profile_match = calculate_similarity(line, profile_text)
        
        # Solo agregar al reporte si no tiene 0% en ambas métricas
        if org_func_match > 0 or org_profile_match > 0:
            org_line_results.append((line, org_func_match, org_profile_match))

    # Evaluación de renglones asistencia a eventos
    for line in att_lines:
        line = line.strip()
        if not line:  # Ignorar líneas vacías
            continue

        # Dividir los eventos en líneas
        att_lines = extract_cleaned_lines(att_text)
        att_lines= att_text.split("\n")
        att_lines = [line.strip() for line in att_lines if line.strip]

        # Evaluación general de concordancia
        if any(keyword.lower() in line.lower() for kw_set in position_indicators.values() for keyword in kw_set):
            att_func_match = 100.0
            att_profile_match = 100.0
        else:
            # Calcular similitud
            att_func_match = calculate_similarity(line, functions_text)
            att_profile_match = calculate_similarity(line, profile_text)
        
        # Solo agregar al reporte si no tiene 0% en ambas métricas
        if att_func_match > 0 or att_profile_match > 0:
            att_line_results.append((line, att_func_match, att_profile_match))

    # Calcular concordancia de funciones y perfil del cargo con perfil de aspirante
    profile_func_match = calculate_similarity(profile_text, functions_text)
    profile_profile_match = calculate_similarity(profile_text, profile_text)
    
    # Calcular porcentajes parciales respecto a la Experiencia ANEIAP
    if line_results:  # Evitar división por cero si no hay ítems válidos
        parcial_exp_func_match = sum([res[1] for res in line_results]) / len(line_results)
        parcial_exp_profile_match = sum([res[2] for res in line_results]) / len(line_results)
    else:
        parcial_exp_func_match = 0
        parcial_exp_profile_match = 0
  
    # Calcular porcentajes parciales respecto a los Eventos ANEIAP
    if org_line_results:  # Evitar división por cero si no hay ítems válidos
        parcial_org_func_match = sum([res[1] for res in org_line_results]) / len(org_line_results)
        parcial_org_profile_match = sum([res[2] for res in org_line_results]) / len(org_line_results)
    else:
        parcial_org_func_match = 0
        parcial_org_profile_match = 0

    # Calcular porcentajes parciales respecto a la asistencia a eventos
    if att_line_results:  # Evitar división por cero si no hay ítems válidos
        parcial_att_func_match = sum([res[1] for res in att_line_results]) / len(att_line_results)
        parcial_att_profile_match = sum([res[2] for res in att_line_results]) / len(att_line_results)
    else:
        parcial_att_func_match = 0
        parcial_att_profile_match = 0

    resume_text= evaluate_cv_presentation(pdf_path)

    # Inicializar corrector ortográfico
    spell = SpellChecker(language='es')

    # Limpiar y dividir el texto en líneas
    pres_cleaned_lines = [line.strip() for line in resume_text.split("\n") if line.strip()]
    total_lines = len(pres_cleaned_lines)

    # Métricas
    total_words = 0
    spelling_errors = 0
    missing_capitalization = 0
    incomplete_sentences = 0
    punctuation_marks = 0
    grammar_errors = 0

    for line in pres_cleaned_lines:
        # Dividir en palabras y contar
        words = re.findall(r'\b\w+\b', line)
        total_words += len(words)

        # Ortografía
        misspelled = spell.unknown(words)
        spelling_errors += len(misspelled)

        # Verificar capitalización
        if line and not line[0].isupper():
            missing_capitalization += 1

        # Verificar que termine en signo de puntuación
        if not line.endswith((".", "!", "?", ":", ";")):
            incomplete_sentences += 1

        # Gramática básica: verificar patrones comunes (ejemplo)
        grammar_errors += len(re.findall(r'\b(?:es|está|son)\b [^\w\s]', line))  # Ejemplo: "es" sin continuación válida

    # Calcular métricas secundarias
    spelling = max(0, 100 - ((spelling_errors / total_words) * 100)) if total_words > 0 else 100
    capitalization_score = max(0, 100 - ((missing_capitalization / total_lines) * 100)) if total_lines > 0 else 100
    sentence_completion_score = max(0, 100 - ((incomplete_sentences / total_lines) * 100)) if total_lines > 0 else 100
    grammar = max(0, 100 - ((grammar_errors / total_lines) * 100)) if total_lines > 0 else 100

    #Calcular métricas principales
    grammar_score = round(((grammar+ sentence_completion_score)/2)/100)*5
    spelling_score= round(((spelling+ capitalization_score)/2)/100)*5

    if total_lines == 0:
        return 100  # Si no hay oraciones, asumimos coherencia perfecta.

    # Variables para análisis
    punctuation_errors = 0
    sentence_lengths = []

    for line in pres_cleaned_lines:
        # Verificar si la oración termina con puntuación válida
        if not line.endswith((".", "!", "?")):
            punctuation_errors += 1

        # Calcular longitud de la oración
        words = line.split()
        if words:
            sentence_lengths.append(len(words))

    # Calcular métricas    
    # 1. Errores de puntuación
    punctuation_error_rate = (punctuation_errors / total_lines) * 100 if total_lines > 0 else 0
    
    # 2. Longitud de las oraciones
    if sentence_lengths:
        length_std_dev = np.std(sentence_lengths)
        max_expected_length = 20  # Longitud promedio esperada de una oración
        length_deviation_score = min(100, (length_std_dev / max_expected_length) * 100)
    else:
        length_deviation_score = 0

    # Calcular coherencia como promedio de las métricas
    coherence_score = 100 - (punctuation_error_rate + length_deviation_score) / 2
     
    # Puntaje general ponderado
    overall_score = round((spelling_score + capitalization_score + sentence_completion_score + coherence_score + grammar_score) / 5, 2)

    # Calculo puntajes parciales
    parcial_exp_func_score = round((parcial_exp_func_match * 5) / 100, 2)
    parcial_exp_profile_score = round((parcial_exp_profile_match * 5) / 100, 2)
    parcial_org_func_score = round((parcial_org_func_match * 5) / 100, 2)
    parcial_org_profile_score = round((parcial_org_profile_match * 5) / 100, 2)
    parcial_att_func_score = round((parcial_att_func_match * 5) / 100, 2)
    parcial_att_profile_score = round((parcial_att_profile_match * 5) / 100, 2)
    profile_func_score= round((profile_func_match * 5) / 100, 2)
    profile_profile_score= round((profile_profile_match * 5) / 100, 2)

    #Calcular resultados globales
    global_func_match = (parcial_exp_func_match + parcial_att_func_match + parcial_org_func_match+ profile_func_match) / 4
    global_profile_match = (parcial_exp_profile_match + parcial_att_profile_match + parcial_org_profile_match + profile_profile_match) / 4
    func_score = round((global_func_match * 5) / 100, 2)
    profile_score = round((global_profile_match * 5) / 100, 2)

    #Calculo puntajes totales
    exp_score= (parcial_exp_func_score+ parcial_exp_profile_score)/2
    org_score= (parcial_org_func_score+ parcial_org_profile_score)/2
    att_score= (parcial_att_func_score+ parcial_att_profile_score)/2
    prof_score= (profile_func_score+ profile_profile_score)/2
    total_score= (overall_score+ exp_score+ org_score+ att_score+ profile_score)/5
    
    # Registrar la fuente personalizada
    pdfmetrics.registerFont(TTFont('CenturyGothic', 'Century_Gothic.ttf'))
    pdfmetrics.registerFont(TTFont('CenturyGothicBold', 'Century_Gothic_Bold.ttf'))

    # Estilos
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenturyGothic", fontName="CenturyGothic", fontSize=12, leading=14, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name="CenturyGothicBold", fontName="CenturyGothicBold", fontSize=12, leading=14, alignment=TA_JUSTIFY))

    # Crear el documento PDF
    report_path = f"Reporte_analisis_cargo_{candidate_name}_{position}.pdf"
    doc = SimpleDocTemplate(report_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=100, bottomMargin=72)

    # Lista de elementos para el reporte
    elements = []

    # Título del reporte centrado
    title_style = ParagraphStyle(name='CenteredTitle', fontName='CenturyGothicBold', fontSize=14, leading=16, alignment=1,  # 1 significa centrado, textColor=colors.black
                                )
    # Convertir texto a mayúsculas
    title_candidate_name = candidate_name.upper()
    title_position = position.upper()
    
    elements.append(Paragraph(f"REPORTE DE ANÁLISIS {title_candidate_name} CARGO {title_position}", title_style))

    elements.append(Spacer(1, 0.2 * inch))

    # Concordancia de items organizada en tabla con ajuste de texto
    elements.append(Paragraph("<b>Análisis de perfil de aspirante:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Encabezados de la tabla
    prof_table_data = [["Ítem", "Funciones del Cargo (%)", "Perfil del Cargo (%)"]]
    
    #Agregar resultados parciales
    prof_table_data.append([Paragraph("<b>Concordancia Parcial</b>", styles['CenturyGothicBold']), f"{profile_func_match:.2f}%", f"{profile_profile_match:.2f}%"])
    prof_table_data.append([Paragraph("<b>Puntaje Parcial</b>", styles['CenturyGothicBold']), f"{profile_func_score:.2f}", f"{profile_profile_score:.2f}"])   

    # Crear la tabla con ancho de columnas ajustado
    prof_item_table = Table(prof_table_data, colWidths=[3 * inch, 2 * inch, 2 * inch])
    
    # Estilos de la tabla con ajuste de texto
    prof_item_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),  # Fondo para encabezados
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Color de texto en encabezados
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Alinear texto al centro
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),  # Fuente para encabezados
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),  # Fuente para el resto de la tabla
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Tamaño de fuente
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),  # Padding inferior para encabezados
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # Líneas de la tabla
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Alinear texto verticalmente al centro
        ('WORDWRAP', (0, 0), (-1, -1)),  # Habilitar ajuste de texto
    ]))
    
    # Agregar tabla a los elementos
    elements.append(prof_item_table)

    elements.append(Spacer(1, 0.2 * inch))

    # Concordancia de items organizada en tabla con ajuste de texto
    elements.append(Paragraph("<b>Análisis de ítems de asistencia a eventos:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Encabezados de la tabla
    att_table_data = [["Ítem", "Funciones del Cargo (%)", "Perfil del Cargo (%)"]]
    
    # Agregar datos de line_results a la tabla
    for line, att_func_match, att_profile_match in att_line_results:
        att_table_data.append([Paragraph(line, styles['CenturyGothic']), f"{att_func_match:.2f}%", f"{att_profile_match:.2f}%"])

    #Agregar resultados parciales
    att_table_data.append([Paragraph("<b>Concordancia Parcial</b>", styles['CenturyGothicBold']), f"{parcial_att_func_match:.2f}%", f"{parcial_att_profile_match:.2f}%"])
    att_table_data.append([Paragraph("<b>Puntaje Parcial</b>", styles['CenturyGothicBold']), f"{parcial_att_func_score:.2f}", f"{parcial_att_profile_score:.2f}"])   

    # Crear la tabla con ancho de columnas ajustado
    att_item_table = Table(att_table_data, colWidths=[3 * inch, 2 * inch, 2 * inch])
    
    # Estilos de la tabla con ajuste de texto
    att_item_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),  # Fondo para encabezados
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Color de texto en encabezados
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Alinear texto al centro
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),  # Fuente para encabezados
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),  # Fuente para el resto de la tabla
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Tamaño de fuente
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),  # Padding inferior para encabezados
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # Líneas de la tabla
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Alinear texto verticalmente al centro
        ('WORDWRAP', (0, 0), (-1, -1)),  # Habilitar ajuste de texto
    ]))
    
    # Agregar tabla a los elementos
    elements.append(att_item_table)

    elements.append(Spacer(1, 0.1 * inch))

    # Total de líneas analizadas en ASISTENCIA A EVENTOS ANEIAP
    att_total_lines = len(att_line_results)
    elements.append(Paragraph(f"• Total de asistencias a eventos analizadas: {att_total_lines}", styles['CenturyGothicBold']))

    elements.append(Spacer(1, 0.2 * inch))

    # Concordancia de items organizada en tabla con ajuste de texto
    elements.append(Paragraph("<b>Análisis de ítems de eventos organizados:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Encabezados de la tabla
    org_table_data = [["Ítem", "Funciones del Cargo (%)", "Perfil del Cargo (%)"]]
    
    # Agregar datos de line_results a la tabla
    for line, org_func_match, org_profile_match in org_line_results:
        org_table_data.append([Paragraph(line, styles['CenturyGothic']), f"{org_func_match:.2f}%", f"{org_profile_match:.2f}%"])

    #Agregar resultados parciales
    org_table_data.append([Paragraph("<b>Concordancia Parcial</b>", styles['CenturyGothicBold']), f"{parcial_org_func_match:.2f}%", f"{parcial_org_profile_match:.2f}%"])
    org_table_data.append([Paragraph("<b>Puntaje Parcial</b>", styles['CenturyGothicBold']), f"{parcial_org_func_score:.2f}", f"{parcial_org_profile_score:.2f}"])   

    # Crear la tabla con ancho de columnas ajustado
    org_item_table = Table(org_table_data, colWidths=[3 * inch, 2 * inch, 2 * inch])
    
    # Estilos de la tabla con ajuste de texto
    org_item_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),  # Fondo para encabezados
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Color de texto en encabezados
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Alinear texto al centro
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),  # Fuente para encabezados
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),  # Fuente para el resto de la tabla
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Tamaño de fuente
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),  # Padding inferior para encabezados
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # Líneas de la tabla
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Alinear texto verticalmente al centro
        ('WORDWRAP', (0, 0), (-1, -1)),  # Habilitar ajuste de texto
    ]))
    
    # Agregar tabla a los elementos
    elements.append(org_item_table)

    elements.append(Spacer(1, 0.1 * inch))

    # Total de líneas analizadas en ASISTENCIA A EVENTOS ANEIAP
    org_total_lines = len(org_line_results)
    elements.append(Paragraph(f"• Total de eventos analizados: {org_total_lines}", styles['CenturyGothicBold']))

    elements.append(Spacer(1, 0.2 * inch))

    # Concordancia de items organizada en tabla con ajuste de texto
    elements.append(Paragraph("<b>Análisis de ítems:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Encabezados de la tabla
    table_data = [["Ítem", "Funciones del Cargo (%)", "Perfil del Cargo (%)"]]
    
    # Agregar datos de line_results a la tabla
    for line, exp_func_match, exp_profile_match in line_results:
        table_data.append([Paragraph(line, styles['CenturyGothic']), f"{exp_func_match:.2f}%", f"{exp_profile_match:.2f}%"])

    #Agregar resultados parciales
    table_data.append([Paragraph("<b>Concordancia Parcial</b>", styles['CenturyGothicBold']), f"{parcial_exp_func_match:.2f}%", f"{parcial_exp_profile_match:.2f}%"])
    table_data.append([Paragraph("<b>Puntaje Parcial</b>", styles['CenturyGothicBold']), f"{parcial_exp_func_score:.2f}", f"{parcial_exp_profile_score:.2f}"])   

    # Crear la tabla con ancho de columnas ajustado
    item_table = Table(table_data, colWidths=[3 * inch, 2 * inch, 2 * inch])
    
    # Estilos de la tabla con ajuste de texto
    item_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),  # Fondo para encabezados
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Color de texto en encabezados
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Alinear texto al centro
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),  # Fuente para encabezados
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),  # Fuente para el resto de la tabla
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Tamaño de fuente
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),  # Padding inferior para encabezados
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # Líneas de la tabla
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Alinear texto verticalmente al centro
        ('WORDWRAP', (0, 0), (-1, -1)),  # Habilitar ajuste de texto
    ]))
    
    # Agregar tabla a los elementos
    elements.append(item_table)
    
    elements.append(Spacer(1, 0.2 * inch))
    
    # Total de líneas analizadas en EXPERIENCIA EN ANEIAP
    total_lines = len(line_results)
    elements.append(Paragraph(f"• Total de experiencias analizadas: {total_lines}", styles['CenturyGothicBold']))

    elements.append(Spacer(1, 0.2 * inch))

    # Añadir resultados al reporte
    elements.append(Paragraph("<b>Evaluación de la Presentación:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Crear tabla de evaluación de presentación
    presentation_table = Table(
        [
            ["Criterio", "Puntaje"],
            ["Coherencia", f"{coherence_score:.2f}"],
            ["Ortografía", f"{spelling_score:.2f}"],
            ["Gramática", f"{grammar_score:.2f}"],
            ["Puntaje Total", f"{overall_score:.2f}"]
        ],
        colWidths=[3 * inch, 2 * inch]
    )
    
    # Estilo de la tabla
    presentation_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(presentation_table)

    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("<b>Consejos para mejorar la presentación de la hoja de vida:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Consejos para coherencia de frases
    if coherence_score < 3:
        elements.append(Paragraph(
            "• Mejora la redacción de las frases en tu hoja de vida. Asegúrate de que sean completas, coherentes y claras.",
            styles['CenturyGothic']
        ))
    elif 3 <= coherence_score <= 4:
        elements.append(Paragraph(
            "• La redacción de tus frases es adecuada, pero revisa la fluidez entre oraciones para mejorar la coherencia general.",
            styles['CenturyGothic']
        ))
    else:
        elements.append(Paragraph(
            "• La redacción de las frases en tu hoja de vida es clara y coherente. Excelente trabajo.",
            styles['CenturyGothic']
        ))
    elements.append(Spacer(1, 0.1 * inch))
    # Consejos para ortografía
    if spelling_score < 3:
        elements.append(Paragraph(
            "• Revisa cuidadosamente la ortografía de tu hoja de vida. Considera utilizar herramientas automáticas para detectar errores de escritura.",
            styles['CenturyGothic']
        ))
    elif 3 <= spelling_score <= 4:
        elements.append(Paragraph(
            "• Tu ortografía es buena, pero aún puede mejorar. Lee tu hoja de vida en voz alta para identificar errores menores.",
            styles['CenturyGothic']
        ))
    else:
        elements.append(Paragraph(
            "• Tu ortografía es excelente. Continúa manteniendo este nivel de detalle en tus documentos.",
            styles['CenturyGothic']
        ))
    elements.append(Spacer(1, 0.1 * inch))
    
    # Consejos para gramática
    if grammar_score < 3:
        elements.append(Paragraph(
            "• Corrige el uso de mayúsculas. Asegúrate de que nombres propios, títulos y principios de frases estén correctamente capitalizados.",
            styles['CenturyGothic']
        ))
    elif 3 <= grammar_score <= 4:
        elements.append(Paragraph(
            "• Tu uso de mayúsculas es aceptable, pero puede perfeccionarse. Revisa los encabezados y títulos para asegurarte de que estén bien escritos.",
            styles['CenturyGothic']
        ))
    else:
        elements.append(Paragraph(
            "• El uso de mayúsculas en tu hoja de vida es excelente. Continúa aplicando este estándar.",
            styles['CenturyGothic']
        ))
    
    elements.append(Spacer(1, 0.2 * inch))
    # Concordancia de items organizada en tabla con ajuste de texto
    elements.append(Paragraph("<b>Resultados de indicadores:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))

    # Encabezados de la tabla
    table_indicator = [["Indicador", "Concordancia (%)"]]
    
    # Agregar datos de line_results a la tabla
    for indicator, data in indicator_results.items():
        relevant_lines = sum(
            any(keyword.lower() in line.lower() for keyword in keywords) for line in lines
        )
        total_lines = len(line_results)
        percentage = (relevant_lines / total_lines) * 100 if total_lines > 0 else 0
        if isinstance(percentage, (int, float)):  # Validar que sea un número
            table_indicator.append([Paragraph(indicator, styles['CenturyGothic']), f"{percentage:.2f}%"])

    # Crear la tabla con ancho de columnas ajustado
    indicator_table = Table(table_indicator, colWidths=[3 * inch, 2 * inch, 2 * inch])
    
    # Estilos de la tabla con ajuste de texto
    indicator_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),  # Fondo para encabezados
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Color de texto en encabezados
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Alinear texto al centro
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),  # Fuente para encabezados
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),  # Fuente para el resto de la tabla
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Tamaño de fuente
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),  # Padding inferior para encabezados
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # Líneas de la tabla
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Alinear texto verticalmente al centro
        ('WORDWRAP', (0, 0), (-1, -1)),  # Habilitar ajuste de texto
    ]))
    
    # Agregar tabla a los elementos
    elements.append(indicator_table)

    elements.append(Spacer(1, 0.2 * inch))
    
    # Consejos para mejorar indicadores con baja presencia
    low_performance_indicators = {k: v for k, v in indicator_results.items() if (relevant_lines/ total_lines) * 100 < 60.0}
    if low_performance_indicators:
        elements.append(Paragraph("<b>Consejos para Mejorar:</b>", styles['CenturyGothicBold']))
        for indicator, result in low_performance_indicators.items():
            percentage = (relevant_lines/ total_lines)*100
            elements.append(Paragraph(f" {indicator}: ({percentage:.2f}%)", styles['CenturyGothicBold']))
            elements.append(Spacer(1, 0.05 * inch))
            for tip in advice[position].get(indicator, []):
                elements.append(Paragraph(f"  • {tip}", styles['CenturyGothic']))
                elements.append(Spacer(1, 0.1 * inch))

    elements.append(Spacer(1, 0.2 * inch))

   # Concordancia de items organizada en tabla global con ajuste de texto
    elements.append(Paragraph("<b>Resultados globales:</b>", styles['CenturyGothicBold']))

    elements.append(Spacer(1, 0.2 * inch))

    # Encabezados de la tabla global
    global_table_data = [["Criterio","Funciones del Cargo", "Perfil del Cargo"]]
    
    # Agregar datos de global_results a la tabla
    global_table_data.append([Paragraph("<b>Concordancia Global</b>", styles['CenturyGothicBold']), f"{global_func_match:.2f}%", f"{global_profile_match:.2f}%"])
    global_table_data.append([Paragraph("<b>Puntaje Global</b>", styles['CenturyGothicBold']), f"{func_score:.2f}", f"{profile_score:.2f}"])

    # Crear la tabla con ancho de columnas ajustado
    global_table = Table(global_table_data, colWidths=[3 * inch, 2 * inch, 2 * inch])
    
    # Estilos de la tabla con ajuste de texto
    global_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),  # Fondo para encabezados
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Color de texto en encabezados
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Alinear texto al centro
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),  # Fuente para encabezados
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),  # Fuente para el resto de la tabla
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Tamaño de fuente
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),  # Padding inferior para encabezados
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # Líneas de la tabla
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Alinear texto verticalmente al centro
        ('WORDWRAP', (0, 0), (-1, -1)),  # Habilitar ajuste de texto
    ]))
    
    # Agregar tabla a los elementos
    elements.append(global_table)
    
    elements.append(Spacer(1, 0.2 * inch))
    
    # Interpretación de resultados
    elements.append(Paragraph("<b>Interpretación de resultados globales:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.1 * inch))
    if global_profile_match > 75 and global_func_match > 75:
        elements.append(Paragraph(
            f" Alta Concordancia (> 0.75): El análisis revela que {candidate_name} tiene una excelente adecuación con las funciones del cargo de {position} y el perfil buscado. La experiencia detallada en su hoja de vida está estrechamente alineada con las responsabilidades y competencias requeridas para este rol crucial en la prevalencia del Capítulo. La alta concordancia indica que {candidate_name} está bien preparado para asumir este cargo y contribuir significativamente al éxito y la misión del Capítulo. Se recomienda proceder con el proceso de selección y considerar a {candidate_name} como una opción sólida para el cargo.",
            styles['CenturyGothic']
        ))
    elif 60 <= global_profile_match <= 75 or 60 <= global_func_match <= 75:
        elements.append(Paragraph(
            f" Buena Concordancia (> 0.60): El análisis muestra que {candidate_name} tiene una buena correspondencia con las funciones del cargo de {position} y el perfil deseado. Aunque su experiencia en la asociación es relevante, existe margen para mejorar. {candidate_name} muestra potencial para cumplir con el rol crucial en la prevalencia del Capítulo, pero se recomienda que continúe desarrollando sus habilidades y acumulando más experiencia relacionada con el cargo objetivo. Su candidatura debe ser considerada con la recomendación de enriquecimiento adicional.",
            styles['CenturyGothic']
        ))
    elif 60 < global_profile_match and 60 < global_func_match:
        elements.append(Paragraph(
            f" Baja Concordancia (< 0.60): El análisis indica que {candidate_name} tiene una baja concordancia con los requisitos del cargo de {position} y el perfil buscado. Esto sugiere que aunque el aspirante posee algunas experiencias relevantes, su historial actual no cubre adecuadamente las competencias y responsabilidades necesarias para este rol crucial en la prevalencia del Capítulo. Se aconseja a {candidate_name} enfocarse en mejorar su perfil profesional y desarrollar las habilidades necesarias para el cargo. Este enfoque permitirá a {candidate_name} alinear mejor su perfil con los requisitos del puesto en futuras oportunidades.",
            styles['CenturyGothic']
        ))

    elements.append(Spacer(1, 0.2 * inch))

    # Añadir resultados al reporte
    elements.append(Paragraph("<b>Puntajes totales:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Crear tabla de evaluación de presentación
    total_table = Table(
        [
            ["Criterio", "Puntaje"],
            ["Experiencia en ANEIAP", f"{exp_score:.2f}"],
            ["Asistencia a eventos", f"{att_score:.2f}"],
            ["Eventos organizados", f"{org_score:.2f}"],
            ["Perfil", f"{prof_score:.2f}"],
            ["Presentación", f"{overall_score:.2f}"],
            ["Puntaje Total", f"{total_score:.2f}"]
        ],
        colWidths=[3 * inch, 2 * inch]
    )
    
    # Estilo de la tabla
    total_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(total_table)

    elements.append(Spacer(1, 0.2 * inch))

    # Generar comentarios para los resultados
    comments = []
    
    if exp_score >= 4:
        comments.append("Tu experiencia en ANEIAP refleja un nivel destacado, lo que demuestra un conocimiento sólido de la organización y tus contribuciones en actividades clave. Continúa fortaleciendo tu participación para mantener este nivel y destacar aún más.")
    elif exp_score >= 3:
        comments.append("Tu experiencia en ANEIAP es buena, pero podrías enfocarte en profundizar tus contribuciones y participación en actividades clave.")
    else:
        comments.append("Es importante fortalecer tu experiencia en ANEIAP. Considera involucrarte en más actividades y proyectos para adquirir una mayor comprensión y relevancia.")
    
    if att_score >= 4:
        comments.append("Tu puntuación en asistencia a eventos es excelente. Esto muestra tu compromiso con el aprendizaje y el desarrollo profesional. Mantén esta consistencia participando en eventos relevantes que sigan ampliando tu red de contactos y conocimientos.")
    elif att_score >= 3:
        comments.append("Tu asistencia a eventos es adecuada, pero hay margen para participar más en actividades que refuercen tu aprendizaje y crecimiento profesional.")
    else:
        comments.append("Debes trabajar en tu participación en eventos. La asistencia regular a actividades puede ayudarte a desarrollar habilidades clave y expandir tu red de contactos.")
    
    if org_score >= 4:
        comments.append("¡Perfecto! Tu desempeño en la organización de eventos es ejemplar. Esto indica habilidades destacadas de planificación, liderazgo y ejecución. Considera compartir tus experiencias con otros miembros para fortalecer el impacto organizacional.")
    elif org_score >= 3:
        comments.append("Tu desempeño en la organización de eventos es bueno, pero podrías centrarte en mejorar la planificación y la ejecución para alcanzar un nivel más destacado.")
    else:
        comments.append("Es importante trabajar en tus habilidades de organización de eventos. Considera involucrarte en proyectos donde puedas asumir un rol de liderazgo y planificación.")
    
    if prof_score >= 4:
        comments.append("Tu perfil presenta una buena alineación con las expectativas del cargo, destacando competencias clave. Mantén este nivel y continúa fortaleciendo áreas relevantes.")
    elif prof_score >= 3:
        comments.append("El perfil presenta una buena alineación con las expectativas del cargo, aunque hay margen de mejora. Podrías enfocar tus esfuerzos en reforzar áreas específicas relacionadas con las competencias clave del puesto.")
    else:
        comments.append("Tu perfil necesita mejoras para alinearse mejor con las expectativas del cargo. Trabaja en desarrollar habilidades y competencias clave.")
    
    if overall_score >= 4:
        comments.append("La presentación de tu hoja de vida es excelente. Refleja profesionalismo y claridad. Continúa aplicando este enfoque para mantener un alto estándar.")
    elif overall_score >= 3:
        comments.append("La presentación de tu hoja de vida es buena, pero puede mejorar en aspectos como coherencia, ortografía o formato general. Dedica tiempo a revisar estos detalles.")
    else:
        comments.append("La presentación de tu hoja de vida necesita mejoras significativas. Asegúrate de revisar la ortografía, la gramática y la coherencia para proyectar una imagen más profesional.")
    
    if total_score >= 4:
        comments.append("Tu puntaje total indica un desempeño destacado en la mayoría de las áreas. Estás bien posicionado para asumir el rol. Mantén este nivel y busca perfeccionar tus fortalezas.")
    elif total_score >= 3:
        comments.append("Tu puntaje total es sólido, pero hay aspectos que podrían mejorarse. Enfócate en perfeccionar la presentación y el perfil para complementar tus fortalezas en experiencia, eventos y asistencia.")
    else:
        comments.append("El puntaje total muestra áreas importantes por mejorar. Trabaja en fortalecer cada criterio para presentar un perfil más competitivo y completo.")
    
    # Añadir comentarios al reporte
    elements.append(Paragraph("<b>Comentarios sobre los Resultados:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))
    for comment in comments:
        elements.append(Paragraph(comment, styles['CenturyGothic']))
        elements.append(Spacer(1, 0.1 * inch))

    elements.append(Spacer(1, 0.1 * inch))

    # Conclusión
    elements.append(Paragraph(
        f"Este análisis es generado debido a que es crucial tomar medidas estratégicas para garantizar que  los candidatos estén bien preparados para el rol de {position}. Los aspirantes con alta concordancia deben ser considerados seriamente para el cargo, ya que están en una posición favorable para asumir responsabilidades significativas y contribuir al éxito del Capítulo. Aquellos con buena concordancia deberían continuar desarrollando su experiencia, mientras que los aspirantes con  baja concordancia deberían recibir orientación para mejorar su perfil profesional y acumular más  experiencia relevante. Estas acciones asegurarán que el proceso de selección se base en una evaluación completa y precisa de las capacidades de cada candidato, fortaleciendo la gestión y el  impacto del Capítulo.",
        styles['CenturyGothic']
    ))

    elements.append(Spacer(1, 0.2 * inch))

    # Mensaje de agradecimiento
    elements.append(Paragraph(
        f"Gracias, {candidate_name}, por tu interés en el cargo de {position} ¡Éxitos en tu proceso!",
        styles['CenturyGothic']
    ))

     # Configuración de funciones de fondo
    def on_first_page(canvas, doc):
        add_background(canvas, background_path)

    def on_later_pages(canvas, doc):
        add_background(canvas, background_path)

    # Construcción del PDF
    doc.build(elements, onFirstPage=on_first_page, onLaterPages=on_later_pages)

    # Descargar el reporte desde Streamlit
    with open(report_path, "rb") as file:
        st.success("Reporte PDF generado exitosamente.")
        st.download_button(
            label="Descargar Reporte PDF",
            data=file,
            file_name= report_path,
            mime="application/pdf"
        )

# FUNCIONES PARA SECUNDARY
def extract_text_with_headers_and_details(pdf_path):
    """
    Extrae encabezados (en negrita) y detalles del texto de un archivo PDF.
    :param pdf_path: Ruta del archivo PDF.
    :return: Diccionario con encabezados como claves y detalles como valores.
    """
    items = {}
    current_header = None

    with fitz.open(pdf_path) as doc:
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue

                        # Detectar encabezados (negrita)
                        if "bold" in span["font"].lower() and not text.startswith("-"):
                            current_header = text
                            items[current_header] = []
                        elif current_header:
                            # Agregar detalles al encabezado actual
                            items[current_header].append(text)

    return items

def extract_experience_items_with_details(pdf_path):
    """
    Extrae encabezados (en negrita) y sus detalles de la sección 'EXPERIENCIA EN ANEIAP'.
    """
    items = {}
    current_item = None
    in_experience_section = False

    with fitz.open(pdf_path) as doc:
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue

                        # Detectar inicio y fin de la sección
                        if "experiencia en aneiap" in text.lower():
                            in_experience_section = True
                            continue
                        elif any(key in text.lower() for key in ["reconocimientos", "eventos organizados"]):
                            in_experience_section = False
                            break

                        if not in_experience_section:
                            continue

                        # Detectar encabezados (negrita) y detalles
                        if "bold" in span["font"].lower() and not text.startswith("-"):
                            current_item = text
                            items[current_item] = []
                        elif current_item:
                            items[current_item].append(text)

    return items

def extract_event_items_with_details(pdf_path):
    """
    Extrae encabezados (en negrita) y sus detalles de la sección 'EVENTOS ORGANIZADOS'.
    """
    items = {}
    current_item = None
    in_eventos_section = False

    with fitz.open(pdf_path) as doc:
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue

                        # Detectar inicio y fin de la sección
                        if "eventos organizados" in text.lower():
                            in_eventos_section = True
                            continue
                        elif any(key in text.lower() for key in ["firma", "experiencia laboral"]):
                            in_eventos_section = False
                            break

                        if not in_eventos_section:
                            continue

                        # Detectar encabezados (negrita) y detalles
                        if "bold" in span["font"].lower() and not text.startswith("-"):
                            current_item = text
                            items[current_item] = []
                        elif current_item:
                            items[current_item].append(text)

    return items

def extract_asistencia_items_with_details(pdf_path):
    """
    Extrae encabezados (en negrita) y sus detalles de la sección 'Asistencia a eventos ANEIAP'.
    """
    items = {}
    current_item = None
    in_asistencia_section = False

    with fitz.open(pdf_path) as doc:
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue

                        # Detectar inicio y fin de la sección
                        if "asistencia a eventos aneiap" in text.lower():
                            in_asistencia_section = True
                            continue
                        elif any(key in text.lower() for key in ["actualización profesional", "firma"]):
                            in_asistencia_section = False
                            break

                        if not in_asistencia_section:
                            continue

                        # Detectar encabezados (negrita) y detalles
                        if "bold" in span["font"].lower() and not text.startswith("-"):
                            current_item = text
                            items[current_item] = []
                        elif current_item:
                            items[current_item].append(text)

    return items

def extract_profile_section_with_details(pdf_path):
    """
    Extrae el texto de la sección 'Perfil' de un archivo PDF, incluyendo encabezados y detalles.
    :param pdf_path: Ruta del archivo PDF.
    :return: Texto completo de la sección 'Perfil'.
    """
    profile_text = ""
    in_profile_section = False

    with fitz.open(pdf_path) as doc:
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue

                        # Detectar inicio y fin de la sección
                        if "perfil" in text.lower():
                            in_profile_section = True
                            continue
                        elif any(key in text.lower() for key in ["asistencia a eventos aneiap", "actualización profesional"]):
                            in_profile_section = False
                            break

                        if in_profile_section:
                            profile_text += text + " "

    return profile_text.strip

def evaluate_cv_presentation_with_headers(pdf_path):
    """
    Evalúa la presentación de la hoja de vida en términos de redacción, ortografía,
    coherencia básica, y claridad, considerando encabezados y detalles.
    :param pdf_path: Ruta del archivo PDF.
    :return: Resultados del análisis de presentación por encabezados y detalles.
    """
    # Cargar texto del PDF
    text = extract_text_with_headers_and_details(pdf_path)  # Asegúrate de tener esta función definida

    if not text:
        return None, "No se pudo extraer texto del archivo PDF."

    # Instanciar SpellChecker
    spell = SpellChecker()

    # Función para evaluar ortografía
    def evaluate_spelling(text):
        """Evalúa la ortografía del texto y retorna un puntaje."""
        if not text or not isinstance(text, str):
            return 0  # Devuelve un puntaje de 0 si el texto no es válido
        words = text.split()
        misspelled = spell.unknown(words)
        if not words:
            return 100
        return ((len(words) - len(misspelled)) / len(words)) * 100


    # Función para evaluar capitalización
    def evaluate_capitalization(text):
        sentences = re.split(r'[.!?]\s*', text.strip())  # Dividir en oraciones usando signos de puntuación
        sentences = [sentence for sentence in sentences if sentence]  # Filtrar oraciones vacías
        correct_caps = sum(1 for sentence in sentences if sentence and sentence[0].isupper())
        if not sentences:
            return 100  # Si no hay oraciones, asumimos puntaje perfecto
        return (correct_caps / len(sentences)) * 100

    # Función para evaluar coherencia de las frases
    def evaluate_sentence_coherence(text):
        try:
            return max(0, min(100, 100 - textstat.flesch_kincaid_grade(text) * 10))  # Normalizar entre 0 y 100
        except Exception:
            return 50  # Puntaje intermedio en caso de error

    # Función para evaluar la calidad del texto
    spelling_score = evaluate_spelling(text)
    capitalization_score = evaluate_capitalization(text)
    coherence_score = evaluate_sentence_coherence(text)
    overall_score = (spelling_score + capitalization_score + coherence_score) / 3
    return {
        "spelling_score": spelling_score,
        "capitalization_score": capitalization_score,
        "coherence_score": coherence_score,
        "overall_score": overall_score,
    }

    # Evaluación de encabezados y detalles
    presentation_results = {}
    for header, details in text.items():
        header_score = evaluate_text_quality(header)  # Evaluar encabezado
        details_score = evaluate_text_quality(" ".join(details))  # Evaluar detalles combinados

        # Guardar resultados en un diccionario
        presentation_results[header] = {
            "header_score": header_score,
            "details_score": details_score,
        }

# Función principal para generar el reporte descriptivo
def analyze_and_generate_descriptive_report_with_background(pdf_path, position, candidate_name, advice, indicators, background_path):
    """
    Analiza un CV descriptivo y genera un reporte PDF con un fondo en cada página.
    :param pdf_path: Ruta del PDF.
    :param position: Cargo al que aspira.
    :param candidate_name: Nombre del candidato.
    :param advice: Diccionario con consejos.
    :param indicators: Diccionario con indicadores y palabras clave.
    :param background_path: Ruta de la imagen de fondo.
    """

    # Extraer la sección 'Perfil'
    profile_text = extract_profile_section_with_details(pdf_path)
    if not profile_text:
        st.error("No se encontró la sección 'Perfil' en el PDF.")
        return

    # Extraer texto de la sección EXPERIENCIA EN ANEIAP
    items = extract_experience_items_with_details(pdf_path)
    if not items:
        st.error("No se encontraron encabezados y detalles de experiencia para analizar.")
        return

    # Extraer texto de la sección EVENTOS ORGANIZADOS
    org_items = extract_event_items_with_details(pdf_path)
    if not org_items:
        st.error("No se encontraron encabezados y detalles de eventos para analizar.")
        return

    # Extraer texto de la sección EVENTOS ORGANIZADOS
    att_items = extract_asistencia_items_with_details(pdf_path)
    if not att_items:
        st.error("No se encontraron encabezados y detalles de asistencias para analizar.")
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
    org_item_results = {}
    att_item_results = {}

    # Calcular la cantidad de ítems relacionados para cada indicador
    related_items_count = {indicator: 0 for indicator in position_indicators}

    # Análisis de la sección de perfil
    profile_func_match = calculate_similarity(profile_text, functions_text)
    profile_profile_match = calculate_similarity(profile_text, profile_text)

    #EXPERIENCIA EN ANEIAP
    for header, details in items.items():
        header_and_details = f"{header} {' '.join(details)}"  # Combinar encabezado y detalles

        # Revisar palabras clave en el encabezado
        header_contains_keywords = any(
            keyword.lower() in header.lower() for keywords in position_indicators.values() for keyword in keywords
        )

        # Revisar palabras clave en los detalles
        details_contains_keywords = any(
            keyword.lower() in detail.lower() for detail in details for keywords in position_indicators.values() for keyword in keywords
        )

        # Determinar concordancia en funciones y perfil
        if header_contains_keywords or details_contains_keywords:
            exp_func_match = 100
            exp_profile_match = 100
        else:
            exp_func_match = calculate_similarity(header_and_details, functions_text)
            exp_profile_match = calculate_similarity(header_and_details, profile_text)

        # Ignorar ítems con 0% en funciones y perfil
        if exp_func_match == 0 and exp_profile_match == 0:
            continue

        # Evaluar indicadores únicamente para el cargo seleccionado
        for indicator, keywords in position_indicators.items():
            # Identificar si el encabezado o detalles contienen palabras clave del indicador
            if any(keyword.lower() in header_and_details.lower() for keyword in keywords):
                related_items_count[indicator] += 1

        item_results[header] = {
            "Funciones del Cargo": exp_func_match,
            "Perfil del Cargo": exp_profile_match,
        }

    # Calcular porcentajes de indicadores
    total_items = len(items)
    indicator_percentages = {
        indicator: (count / total_items) * 100 if total_items > 0 else 0 for indicator, count in related_items_count.items()
    }

    # Consejos para indicadores críticos (<60% de concordancia)
    critical_advice = {
        indicator: advice.get(position, {}).get(indicator, ["No hay consejos disponibles para este indicador."])
        for indicator, percentage in indicator_percentages.items() if percentage < 60
    }

    #EVENTOS ORGANIZADOS
    for header, details in org_items.items():
        header_and_details = f"{header} {' '.join(details)}"  # Combinar encabezado y detalles

        # Revisar palabras clave en el encabezado
        header_contains_keywords = any(
            keyword.lower() in header.lower() for keywords in position_indicators.values() for keyword in keywords
        )

        # Revisar palabras clave en los detalles
        details_contains_keywords = any(
            keyword.lower() in detail.lower() for detail in details for keywords in position_indicators.values() for keyword in keywords
        )

        # Determinar concordancia en funciones y perfil
        if header_contains_keywords or details_contains_keywords:
            org_func_match = 100
            org_profile_match = 100
        else:
            org_func_match = calculate_similarity(header_and_details, functions_text)
            org_profile_match = calculate_similarity(header_and_details, profile_text)

        # Ignorar ítems con 0% en funciones y perfil
        if org_func_match == 0 and org_profile_match == 0:
            continue

        org_item_results[header] = {
                "Funciones del Cargo": org_func_match,
                "Perfil del Cargo": org_profile_match,
            }

    #ASISTENCIA A EVENTOS
    for header, details in att_items.items():
        header_and_details = f"{header} {' '.join(details)}"  # Combinar encabezado y detalles

        # Revisar palabras clave en el encabezado
        header_contains_keywords = any(
            keyword.lower() in header.lower() for keywords in position_indicators.values() for keyword in keywords
        )

        # Revisar palabras clave en los detalles
        details_contains_keywords = any(
            keyword.lower() in detail.lower() for detail in details for keywords in position_indicators.values() for keyword in keywords
        )

        # Determinar concordancia en funciones y perfil
        if header_contains_keywords or details_contains_keywords:
            att_func_match = 100
            att_profile_match = 100
        else:
            att_func_match = calculate_similarity(header_and_details, functions_text)
            att_profile_match = calculate_similarity(header_and_details, profile_text)

        # Ignorar ítems con 0% en funciones y perfil
        if att_func_match == 0 and att_profile_match == 0:
            continue

        att_item_results[header] = {
                "Funciones del Cargo": att_func_match,
                "Perfil del Cargo": att_profile_match,
            }

    #Calcular concordancia parcial para Experiencia ANEIAP
    if item_results:
        parcial_exp_func_match = sum(res["Funciones del Cargo"] for res in item_results.values()) / len(item_results)
        parcial_exp_profile_match = sum(res["Perfil del Cargo"] for res in item_results.values()) / len(item_results)
    else:
        parcial_exp_func_match = 0
        parcial_exp_profile_match = 0

    #Calcular concordancia parcial para Eventos Organizados
    if item_results:
        parcial_org_func_match = sum(res["Funciones del Cargo"] for res in org_item_results.values()) / len(org_item_results)
        parcial_org_profile_match = sum(res["Perfil del Cargo"] for res in org_item_results.values()) / len(org_item_results)
    else:
        parcial_org_func_match = 0
        parcial_org_profile_match = 0

    #Calcular concordancia parcial para Asistencia a eventos
    if item_results:
        parcial_att_func_match = sum(res["Funciones del Cargo"] for res in att_item_results.values()) / len(att_item_results)
        parcial_att_profile_match = sum(res["Perfil del Cargo"] for res in att_item_results.values()) / len(att_item_results)
    else:
        parcial_att_func_match = 0
        parcial_att_profile_match = 0

    # Extraer texto del PDF con encabezados y detalles
    text_data = extract_text_with_headers_and_details(pdf_path)  # Asegúrate de tener esta función definida

    if not text_data:
        st.error("No se pudo extraer texto del archivo PDF.")
        return None

    # Instanciar el corrector ortográfico
    spell = SpellChecker()

    # Definir funciones para evaluación avanzada de presentación
    def evaluate_spelling(text):
        """Evalúa la ortografía del texto y retorna un puntaje."""
        if not text or not isinstance(text, str):
            return 0  # Texto inválido devuelve 0
        words = text.split()
        misspelled = spell.unknown(words)
        if not words:
            return 100  # Si no hay palabras, asumimos puntaje perfecto
        return ((len(words) - len(misspelled)) / len(words)) * 100

    def evaluate_capitalization(text):
        """Evalúa si las frases comienzan con mayúscula."""
        if not text or not isinstance(text, str):
            return 0  # Texto inválido devuelve 0
        sentences = re.split(r'[.!?]\s*', text.strip())  # Dividir en oraciones usando signos de puntuación
        sentences = [sentence for sentence in sentences if sentence]  # Filtrar oraciones vacías
        correct_caps = sum(1 for sentence in sentences if sentence and sentence[0].isupper())
        if not sentences:
            return 100  # Si no hay oraciones, asumimos puntaje perfecto
        return (correct_caps / len(sentences)) * 100

    def evaluate_sentence_coherence(text):
        """Evalúa la coherencia y legibilidad del texto."""
        if not text or not isinstance(text, str):
            return 0  # Texto inválido devuelve 0
        try:
            # Normalizar entre 0 y 100 usando Flesch-Kincaid Grade
            return max(0, min(100, 100 - textstat.flesch_kincaid_grade(text) * 10))
        except Exception:
            return 50  # Puntaje intermedio en caso de error

    # Evaluación por encabezado y detalles
    presentation_results = {}
    for header, details in text_data.items():
        # Convertir detalles en texto completo
        details_text = " ".join(details)

        # Evaluar encabezado
        header_spelling = evaluate_spelling(header)
        header_capitalization = evaluate_capitalization(header)
        header_coherence = evaluate_sentence_coherence(header)
        header_overall = (header_spelling + header_capitalization + header_coherence) / 3

        # Evaluar detalles
        details_spelling = evaluate_spelling(details_text)
        details_capitalization = evaluate_capitalization(details_text)
        details_coherence = evaluate_sentence_coherence(details_text)
        details_overall = (details_spelling + details_capitalization + details_coherence) / 3

        # Guardar resultados para cada encabezado y sus detalles
        presentation_results[header] = {
            "header_score": {
                "spelling_score": header_spelling,
                "capitalization_score": header_capitalization,
                "coherence_score": header_coherence,
                "overall_score": header_overall,
            },
            "details_score": {
                "spelling_score": details_spelling,
                "capitalization_score": details_capitalization,
                "coherence_score": details_coherence,
                "overall_score": details_overall,
            },
        }

    # Calculo puntajes parciales
    exp_func_score = round((parcial_exp_func_match * 5) / 100, 2)
    exp_profile_score = round((parcial_exp_profile_match * 5) / 100, 2)
    org_func_score = round((parcial_org_func_match * 5) / 100, 2)
    org_profile_score = round((parcial_org_profile_match * 5) / 100, 2)
    att_func_score = round((parcial_att_func_match * 5) / 100, 2)
    att_profile_score = round((parcial_att_profile_match * 5) / 100, 2)
    profile_func_score = round((profile_func_match * 5) / 100, 2)
    profile_profile_score = round((profile_profile_match * 5) / 100, 2)

    # Calcular concordancia global para funciones y perfil
    global_func_match = (parcial_exp_func_match + parcial_att_func_match + parcial_org_func_match + profile_func_match) / 4
    global_profile_match = (parcial_exp_profile_match + parcial_att_profile_match + parcial_org_profile_match + profile_profile_match) / 4
    
    # Calcular puntaje global
    func_score = round((global_func_match * 5) / 100, 2)
    profile_score = round((global_profile_match * 5) / 100, 2)

    #Calculo de puntajes totales
    exp_score= (exp_func_score+ exp_profile_score)/2
    org_score= (org_func_score+ org_profile_score)/2
    att_score= (att_func_score+ att_profile_score)/2
    prof_score= (profile_func_score+ profile_profile_score)/2
    
    # Registrar la fuente personalizada
    pdfmetrics.registerFont(TTFont('CenturyGothic', 'Century_Gothic.ttf'))
    pdfmetrics.registerFont(TTFont('CenturyGothicBold', 'Century_Gothic_Bold.ttf'))

    # Estilos
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenturyGothic", fontName="CenturyGothic", fontSize=12, leading=14, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name="CenturyGothicBold", fontName="CenturyGothicBold", fontSize=12, leading=14, alignment=TA_JUSTIFY))

    # Crear el documento PDF
    output_path = f"Reporte_descriptivo_cargo_{candidate_name}_{position}.pdf"
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=100, bottomMargin=72)

    # Lista de elementos para el reporte
    elements = []
    
    # Título del reporte centrado
    title_style = ParagraphStyle(name='CenteredTitle', fontName='CenturyGothicBold', fontSize=14, leading=16, alignment=1,  # 1 significa centrado, textColor=colors.black
                                )
    
    # Convertir texto a mayúsculas
    title_candidate_name = candidate_name.upper()
    title_position = position.upper()

    elements.append(Paragraph(f"REPORTE DE ANÁLISIS DESCRIPTIVO {title_candidate_name} CARGO {title_position}", title_style))

    elements.append(Spacer(1, 0.2 * inch))

    # Concordancia de items organizada en tabla con ajuste de texto
    elements.append(Paragraph("<b>Análisis de perfil de aspirante:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Encabezados de la tabla
    prof_table_data = [["Ítem", "Funciones del Cargo (%)", "Perfil del Cargo (%)"]]
    
    #Agregar resultados parciales
    prof_table_data.append([Paragraph("<b>Concordancia Parcial</b>", styles['CenturyGothicBold']), f"{profile_func_match:.2f}%", f"{profile_profile_match:.2f}%"])
    prof_table_data.append([Paragraph("<b>Puntaje Parcial</b>", styles['CenturyGothicBold']), f"{profile_func_score:.2f}", f"{profile_profile_score:.2f}"])   

    # Crear la tabla con ancho de columnas ajustado
    prof_item_table = Table(prof_table_data, colWidths=[3 * inch, 2 * inch, 2 * inch])
    
    # Estilos de la tabla con ajuste de texto
    prof_item_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),  # Fondo para encabezados
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Color de texto en encabezados
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Alinear texto al centro
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),  # Fuente para encabezados
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),  # Fuente para el resto de la tabla
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Tamaño de fuente
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),  # Padding inferior para encabezados
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # Líneas de la tabla
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Alinear texto verticalmente al centro
        ('WORDWRAP', (0, 0), (-1, -1)),  # Habilitar ajuste de texto
    ]))
    
    # Agregar tabla a los elementos
    elements.append(prof_item_table)

    elements.append(Spacer(1, 0.2 * inch))

    # Encabezados de la tabla
    org_table_data = [["Ítem", "Funciones del Cargo (%)", "Perfil del Cargo (%)"]]  # Encabezados
    
    # Iterar sobre los resultados por ítem y construir las filas de la tabla
    for header, result in org_item_results.items():
        org_func_match = result.get("Funciones del Cargo", 0)
        org_profile_match = result.get("Perfil del Cargo", 0)
        
        # Ajustar texto del encabezado para que no desborde
        header_paragraph = Paragraph(header, styles['CenturyGothic'])
    
        # Agregar una fila a la tabla
        org_table_data.append([
            header_paragraph,         # Ítem
            f"{org_func_match:.2f}%",    # Funciones del Cargo
            f"{org_profile_match:.2f}%"  # Perfil del Cargo
        ])

    #Agregar resultados parciales
    org_table_data.append([Paragraph("<b>Concordancia Parcial</b>", styles['CenturyGothicBold']), f"{parcial_org_func_match:.2f}%", f"{parcial_org_profile_match:.2f}%"])
    org_table_data.append([Paragraph("<b>Puntaje Parcial</b>", styles['CenturyGothicBold']), f"{org_func_score:.2f}", f"{org_profile_score:.2f}"])
        
    # Crear la tabla
    org_table = Table(org_table_data, colWidths=[3 * inch, 2 * inch, 2 * inch])
    
    # Aplicar estilos a la tabla
    org_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),  # Fondo de encabezados
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),                 # Color de texto de encabezados
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),                        # Alinear texto al centro
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),           # Fuente para encabezados
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),              # Fuente para celdas
        ('FONTSIZE', (0, 0), (-1, -1), 10),                           # Tamaño de fuente
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),                        # Padding inferior de encabezados
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),                 # Líneas de la tabla
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),                       # Alinear texto verticalmente
        ('WORDWRAP', (0, 0), (-1, -1))                                # Ajustar texto dentro de celdas
    ]))
    
    # Agregar la tabla al reporte
    elements.append(Paragraph("<b>Análisis de eventos organizados:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(org_table)
    elements.append(Spacer(1, 0.2 * inch))
    
    # Total de líneas analizadas
    org_total_items = len(org_item_results)
    elements.append(Paragraph(f"• Total de eventos analizados: {org_total_items}", styles['CenturyGothicBold']))

    elements.append(Spacer(1, 0.2 * inch))

    # Encabezados de la tabla
    att_table_data = [["Ítem", "Funciones del Cargo (%)", "Perfil del Cargo (%)"]]  # Encabezados
    
    # Iterar sobre los resultados por ítem y construir las filas de la tabla
    for header, result in att_item_results.items():
        att_func_match = result.get("Funciones del Cargo", 0)
        att_profile_match = result.get("Perfil del Cargo", 0)
        
        # Ajustar texto del encabezado para que no desborde
        header_paragraph = Paragraph(header, styles['CenturyGothic'])
    
        # Agregar una fila a la tabla
        att_table_data.append([
            header_paragraph,         # Ítem
            f"{att_func_match:.2f}%",    # Funciones del Cargo
            f"{att_profile_match:.2f}%"  # Perfil del Cargo
        ])

    #Agregar resultados parciales
    att_table_data.append([Paragraph("<b>Concordancia Parcial</b>", styles['CenturyGothicBold']), f"{parcial_att_func_match:.2f}%", f"{parcial_att_profile_match:.2f}%"])
    att_table_data.append([Paragraph("<b>Puntaje Parcial</b>", styles['CenturyGothicBold']), f"{att_func_score:.2f}", f"{att_profile_score:.2f}"])
        
    # Crear la tabla
    att_table = Table(att_table_data, colWidths=[3 * inch, 2 * inch, 2 * inch])
    
    # Aplicar estilos a la tabla
    att_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),  # Fondo de encabezados
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),                 # Color de texto de encabezados
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),                        # Alinear texto al centro
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),           # Fuente para encabezados
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),              # Fuente para celdas
        ('FONTSIZE', (0, 0), (-1, -1), 10),                           # Tamaño de fuente
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),                        # Padding inferior de encabezados
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),                 # Líneas de la tabla
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),                       # Alinear texto verticalmente
        ('WORDWRAP', (0, 0), (-1, -1))                                # Ajustar texto dentro de celdas
    ]))
    
    # Agregar la tabla al reporte
    elements.append(Paragraph("<b>Análisis de eventos asistidos:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(att_table)
    elements.append(Spacer(1, 0.2 * inch))
    
    # Total de líneas analizadas
    att_total_items = len(att_item_results)
    elements.append(Paragraph(f"• Total de asistencias analizadas: {att_total_items}", styles['CenturyGothicBold']))

    elements.append(Spacer(1, 0.2 * inch))

    # Encabezados de la tabla
    item_table_data = [["Ítem", "Funciones del Cargo (%)", "Perfil del Cargo (%)"]]  # Encabezados
    
    # Iterar sobre los resultados por ítem y construir las filas de la tabla
    for header, result in item_results.items():
        exp_func_match = result.get("Funciones del Cargo", 0)
        exp_profile_match = result.get("Perfil del Cargo", 0)
        
        # Ajustar texto del encabezado para que no desborde
        header_paragraph = Paragraph(header, styles['CenturyGothic'])
    
        # Agregar una fila a la tabla
        item_table_data.append([
            header_paragraph,         # Ítem
            f"{exp_func_match:.2f}%",    # Funciones del Cargo
            f"{exp_profile_match:.2f}%"  # Perfil del Cargo
        ])

    #Agregar resultados parciales
    item_table_data.append([Paragraph("<b>Concordancia Parcial</b>", styles['CenturyGothicBold']), f"{parcial_exp_func_match:.2f}%", f"{parcial_exp_profile_match:.2f}%"])
    item_table_data.append([Paragraph("<b>Puntaje Parcial</b>", styles['CenturyGothicBold']), f"{exp_func_score:.2f}", f"{exp_profile_score:.2f}"])
        
    # Crear la tabla
    item_table = Table(item_table_data, colWidths=[3 * inch, 2 * inch, 2 * inch])
    
    # Aplicar estilos a la tabla
    item_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),  # Fondo de encabezados
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),                 # Color de texto de encabezados
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),                        # Alinear texto al centro
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),           # Fuente para encabezados
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),              # Fuente para celdas
        ('FONTSIZE', (0, 0), (-1, -1), 10),                           # Tamaño de fuente
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),                        # Padding inferior de encabezados
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),                 # Líneas de la tabla
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),                       # Alinear texto verticalmente
        ('WORDWRAP', (0, 0), (-1, -1))                                # Ajustar texto dentro de celdas
    ]))
    
    # Agregar la tabla al reporte
    elements.append(Paragraph("<b>Análisis de Ítems:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(item_table)
    elements.append(Spacer(1, 0.2 * inch))
    
    # Total de líneas analizadas
    total_items = len(item_results)
    elements.append(Paragraph(f"• Total de experiencias analizadas: {total_items}", styles['CenturyGothicBold']))

    elements.append(Spacer(1, 0.2 * inch))

    # Crear tabla con la estructura deseada
    presentation_table_data = [["Criterio", "Puntaje"]]
    
    # Inicializar variables para combinar encabezados y detalles
    total_spelling_score = 0
    total_capitalization_score = 0
    total_coherence_score = 0
    total_overall_score = 0
    total_sections = 0
    
    # Combinar puntajes de encabezados y detalles
    for header, scores in presentation_results.items():
        header_scores = scores["header_score"]
        details_scores = scores["details_score"]
    
        total_spelling_score += (header_scores["spelling_score"] + details_scores["spelling_score"])
        total_capitalization_score += (header_scores["capitalization_score"] + details_scores["capitalization_score"])
        total_coherence_score += (header_scores["coherence_score"] + details_scores["coherence_score"])
        total_overall_score += (header_scores["overall_score"] + details_scores["overall_score"])
        total_sections += 2  # Sumar encabezado y detalle como secciones separadas
    
    # Calcular promedios generales
    average_spelling_score = total_spelling_score / total_sections if total_sections > 0 else 0
    average_capitalization_score = total_capitalization_score / total_sections if total_sections > 0 else 0
    average_coherence_score = total_coherence_score / total_sections if total_sections > 0 else 0
    average_overall_score = total_overall_score / total_sections if total_sections > 0 else 0

    # Calcular puntajes ajustados
    round_spelling_score = round((average_spelling_score / 100) * 5, 2) 
    round_capitalization_score = round((average_capitalization_score / 100) * 5, 2) 
    round_coherence_score = round((average_coherence_score / 100) * 5, 2) 
    round_overall_score = round((average_overall_score / 100) * 5, 2) 
    
    # Agregar los puntajes combinados a la tabla
    presentation_table_data.append(["Coherencia", f"{round_spelling_score:.2f}"])
    presentation_table_data.append(["Ortografía", f"{round_capitalization_score:.2f}"])
    presentation_table_data.append(["Gramática", f"{round_coherence_score:.2f}"])
    presentation_table_data.append(["Puntaje Total", f"{round_overall_score:.2f}"])
    
    # Crear la tabla con ancho ajustado para las columnas
    presentation_table = Table(presentation_table_data, colWidths=[4 * inch, 2 * inch])
    
    # Estilo de la tabla
    presentation_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),  # Fondo para encabezados
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Color de texto en encabezados
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Centrar texto
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),  # Fuente en encabezados
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),  # Fuente en datos
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Tamaño de fuente
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),  # Espaciado inferior en encabezados
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # Líneas de la tabla
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Centrar texto verticalmente
    ]))
    
    # Agregar la tabla a los elementos
    elements.append(presentation_table)
    elements.append(Spacer(1, 0.2 * inch))

    # Generar consejos basados en puntajes
    elements.append(Paragraph("<b>Consejos para Mejorar la Presentación:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Consejos para coherencia
    if round_coherence_score < 3:
        elements.append(Paragraph(
            "• La coherencia de las frases necesita atención. Asegúrate de conectar las ideas claramente y evitar frases fragmentadas.",
            styles['CenturyGothic']
        ))
    elif 3 <= round_coherence_score <= 4:
        elements.append(Paragraph(
            "• La coherencia es aceptable, pero hay margen de mejora. Revisa las transiciones entre ideas para lograr un flujo más natural.",
            styles['CenturyGothic']
        ))
    else:
        elements.append(Paragraph(
            "• La coherencia de las frases es excelente. Sigue organizando las ideas de manera clara y lógica.",
            styles['CenturyGothic']
        ))
    
    # Consejos para ortografía
    if round_spelling_score < 3:
        elements.append(Paragraph(
            "• Revisa cuidadosamente la ortografía. Utiliza herramientas como correctores automáticos para identificar y corregir errores.",
            styles['CenturyGothic']
        ))
    elif 3 <= round_spelling_score <= 4:
        elements.append(Paragraph(
            "• La ortografía es buena, pero se pueden corregir errores menores. Dedica tiempo a revisar cada palabra detenidamente.",
            styles['CenturyGothic']
        ))
    else:
        elements.append(Paragraph(
            "• La ortografía es impecable. Sigue prestando atención a los detalles en tus documentos.",
            styles['CenturyGothic']
        ))
    
    # Consejos para gramática
    if round_capitalization_score < 3:
        elements.append(Paragraph(
            "• El uso de mayúsculas y la gramática necesitan mejoras. Asegúrate de que los nombres propios y los títulos estén correctamente capitalizados.",
            styles['CenturyGothic']
        ))
    elif 3 <= round_capitalization_score <= 4:
        elements.append(Paragraph(
            "• El uso de mayúsculas es correcto, pero puede perfeccionarse. Revisa los títulos y encabezados para asegurarte de que sean consistentes.",
            styles['CenturyGothic']
        ))
    else:
        elements.append(Paragraph(
            "• El uso de mayúsculas es excelente. Mantén este nivel de precisión en la gramática y los detalles del documento.",
            styles['CenturyGothic']
        ))
    
    elements.append(Spacer(1, 0.2 * inch))

    # Concordancia de items organizada en tabla con ajuste de texto
    elements.append(Paragraph("<b>Resultados de indicadores:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))

    # Encabezados de la tabla
    table_indicator = [["Indicador", "Concordancia (%)"]]
    
    # Agregar datos de line_results a la tabla
    for indicator, percentage in indicator_percentages.items():
        if isinstance(percentage, (int, float)):
            table_indicator.append([Paragraph(indicator, styles['CenturyGothic']), f"{percentage:.2f}%"])

    # Crear la tabla con ancho de columnas ajustado
    indicator_table = Table(table_indicator, colWidths=[3 * inch, 2 * inch, 2 * inch])
    
    # Estilos de la tabla con ajuste de texto
    indicator_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),  # Fondo para encabezados
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Color de texto en encabezados
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Alinear texto al centro
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),  # Fuente para encabezados
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),  # Fuente para el resto de la tabla
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Tamaño de fuente
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),  # Padding inferior para encabezados
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # Líneas de la tabla
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Alinear texto verticalmente al centro
        ('WORDWRAP', (0, 0), (-1, -1)),  # Habilitar ajuste de texto
    ]))
    
    # Agregar tabla a los elementos
    elements.append(indicator_table)

    elements.append(Spacer(1, 0.2 * inch))
    
    # Mostrar consejos para indicadores con porcentaje menor al 50%
    elements.append(Paragraph("<b>Consejos para Indicadores Críticos:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.05 * inch))
    for indicator, percentage in indicator_percentages.items():
        if percentage < 60: 
            elements.append(Paragraph(f"  Indicador: {indicator} ({percentage:.2f}%)", styles['CenturyGothicBold']))
            for tip in critical_advice.get(indicator, ["No hay consejos disponibles para este indicador."]):
                elements.append(Paragraph(f"    • {tip}", styles['CenturyGothic']))
                elements.append(Spacer(1, 0.1 * inch))

    elements.append(Spacer(1, 0.2 * inch))

    # Concordancia de items organizada en tabla global con ajuste de texto
    elements.append(Paragraph("<b>Resultados globales:</b>", styles['CenturyGothicBold']))

    elements.append(Spacer(1, 0.2 * inch))

    # Encabezados de la tabla global
    global_table_data = [["Criterio","Funciones del Cargo", "Perfil del Cargo"]]
    
    # Agregar datos de global_results a la tabla
    global_table_data.append([Paragraph("<b>Concordancia Global</b>", styles['CenturyGothicBold']), f"{global_func_match:.2f}%", f"{global_profile_match:.2f}%"])
    global_table_data.append([Paragraph("<b>Puntaje Global</b>", styles['CenturyGothicBold']), f"{func_score:.2f}", f"{profile_score:.2f}"])

    # Crear la tabla con ancho de columnas ajustado
    global_table = Table(global_table_data, colWidths=[3 * inch, 2 * inch, 2 * inch])
    
    # Estilos de la tabla con ajuste de texto
    global_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),  # Fondo para encabezados
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Color de texto en encabezados
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Alinear texto al centro
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),  # Fuente para encabezados
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),  # Fuente para el resto de la tabla
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Tamaño de fuente
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),  # Padding inferior para encabezados
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # Líneas de la tabla
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Alinear texto verticalmente al centro
        ('WORDWRAP', (0, 0), (-1, -1)),  # Habilitar ajuste de texto
    ]))
    
    # Agregar tabla a los elementos
    elements.append(global_table)
    
    elements.append(Spacer(1, 0.2 * inch))
    
    # Interpretación de resultados
    elements.append(Paragraph("<b>Interpretación de resultados globales:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.1 * inch))
    if global_profile_match > 75 and global_func_match > 75:
        elements.append(Paragraph(
            f" Alta Concordancia (> 0.75): El análisis revela que {candidate_name} tiene una excelente adecuación con las funciones del cargo de {position} y el perfil buscado. La experiencia detallada en su hoja de vida está estrechamente alineada con las responsabilidades y competencias requeridas para este rol crucial en la prevalencia del Capítulo. La alta concordancia indica que {candidate_name} está bien preparado para asumir este cargo y contribuir significativamente al éxito y la misión del Capítulo. Se recomienda proceder con el proceso de selección y considerar a {candidate_name} como una opción sólida para el cargo.",
            styles['CenturyGothic']
        ))
    elif 60 < global_profile_match <= 75 or 60 < global_func_match <= 75:
        elements.append(Paragraph(
            f" Buena Concordancia (> 0.60): El análisis muestra que {candidate_name} tiene una buena correspondencia con las funciones del cargo de {position} y el perfil deseado. Aunque su experiencia en la asociación es relevante, existe margen para mejorar. {candidate_name} muestra potencial para cumplir con el rol crucial en la prevalencia del Capítulo, pero se recomienda que continúe desarrollando sus habilidades y acumulando más experiencia relacionada con el cargo objetivo. Su candidatura debe ser considerada con la recomendación de enriquecimiento adicional.",
            styles['CenturyGothic']
        ))
    elif 60 < global_profile_match and 60 < global_func_match:
        elements.append(Paragraph(
            f" Baja Concordancia (< 0.60): El análisis indica que {candidate_name} tiene una baja concordancia con los requisitos del cargo de {position} y el perfil buscado. Esto sugiere que aunque el aspirante posee algunas experiencias relevantes, su historial actual no cubre adecuadamente las competencias y responsabilidades necesarias para este rol crucial en la prevalencia del Capítulo. Se aconseja a {candidate_name} enfocarse en mejorar su perfil profesional y desarrollar las habilidades necesarias para el cargo. Este enfoque permitirá a {candidate_name} alinear mejor su perfil con los requisitos del puesto en futuras oportunidades.",
            styles['CenturyGothic']
        ))

    elements.append(Spacer(1, 0.2 * inch))

    # Añadir resultados al reporte
    elements.append(Paragraph("<b>Puntajes totales:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))

    total_score= (exp_score+ att_score+ org_score+ round_overall_score+ profile_score)/5
    
    # Crear tabla de evaluación de presentación
    total_table = Table(
        [
            ["Criterio", "Puntaje"],
            ["Experiencia en ANEIAP", f"{exp_score:.2f}"],
            ["Asistencia a eventos", f"{att_score:.2f}"],
            ["Eventos organizados", f"{org_score:.2f}"],
            ["Perfil", f"{prof_score:.2f}"],
            ["Presentación", f"{round_overall_score:.2f}"],
            ["Puntaje Total", f"{total_score:.2f}"]
        ],
        colWidths=[3 * inch, 2 * inch]
    )
    
    # Estilo de la tabla
    total_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'CenturyGothicBold'),
        ('FONTNAME', (0, 1), (-1, -1), 'CenturyGothic'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(total_table)

    elements.append(Spacer(1, 0.2 * inch))

    # Generar comentarios para los resultados
    comments = []
    
    if exp_score >= 4:
        comments.append("Tu experiencia en ANEIAP refleja un nivel destacado, lo que demuestra un conocimiento sólido de la organización y tus contribuciones en actividades clave. Continúa fortaleciendo tu participación para mantener este nivel y destacar aún más.")
    elif exp_score >= 3:
        comments.append("Tu experiencia en ANEIAP es buena, pero podrías enfocarte en profundizar tus contribuciones y participación en actividades clave.")
    else:
        comments.append("Es importante fortalecer tu experiencia en ANEIAP. Considera involucrarte en más actividades y proyectos para adquirir una mayor comprensión y relevancia.")
    
    if att_score >= 4:
        comments.append("Tu puntuación en asistencia a eventos es excelente. Esto muestra tu compromiso con el aprendizaje y el desarrollo profesional. Mantén esta consistencia participando en eventos relevantes que sigan ampliando tu red de contactos y conocimientos.")
    elif att_score >= 3:
        comments.append("Tu asistencia a eventos es adecuada, pero hay margen para participar más en actividades que refuercen tu aprendizaje y crecimiento profesional.")
    else:
        comments.append("Debes trabajar en tu participación en eventos. La asistencia regular a actividades puede ayudarte a desarrollar habilidades clave y expandir tu red de contactos.")
    
    if org_score >= 4:
        comments.append("¡Perfecto! Tu desempeño en la organización de eventos es ejemplar. Esto indica habilidades destacadas de planificación, liderazgo y ejecución. Considera compartir tus experiencias con otros miembros para fortalecer el impacto organizacional.")
    elif org_score >= 3:
        comments.append("Tu desempeño en la organización de eventos es bueno, pero podrías centrarte en mejorar la planificación y la ejecución para alcanzar un nivel más destacado.")
    else:
        comments.append("Es importante trabajar en tus habilidades de organización de eventos. Considera involucrarte en proyectos donde puedas asumir un rol de liderazgo y planificación.")
    
    if prof_score >= 4:
        comments.append("Tu perfil presenta una buena alineación con las expectativas del cargo, destacando competencias clave. Mantén este nivel y continúa fortaleciendo áreas relevantes.")
    elif prof_score >= 3:
        comments.append("El perfil presenta una buena alineación con las expectativas del cargo, aunque hay margen de mejora. Podrías enfocar tus esfuerzos en reforzar áreas específicas relacionadas con las competencias clave del puesto.")
    else:
        comments.append("Tu perfil necesita mejoras para alinearse mejor con las expectativas del cargo. Trabaja en desarrollar habilidades y competencias clave.")
    
    if round_overall_score >= 4:
        comments.append("La presentación de tu hoja de vida es excelente. Refleja profesionalismo y claridad. Continúa aplicando este enfoque para mantener un alto estándar.")
    elif round_overall_score >= 3:
        comments.append("La presentación de tu hoja de vida es buena, pero puede mejorar en aspectos como coherencia, ortografía o formato general. Dedica tiempo a revisar estos detalles.")
    else:
        comments.append("La presentación de tu hoja de vida necesita mejoras significativas. Asegúrate de revisar la ortografía, la gramática y la coherencia para proyectar una imagen más profesional.")
    
    if total_score >= 4:
        comments.append("Tu puntaje total indica un desempeño destacado en la mayoría de las áreas. Estás bien posicionado para asumir el rol. Mantén este nivel y busca perfeccionar tus fortalezas.")
    elif total_score >= 3:
        comments.append("Tu puntaje total es sólido, pero hay aspectos que podrían mejorarse. Enfócate en perfeccionar la presentación y el perfil para complementar tus fortalezas en experiencia, eventos y asistencia.")
    else:
        comments.append("El puntaje total muestra áreas importantes por mejorar. Trabaja en fortalecer cada criterio para presentar un perfil más competitivo y completo.")
    
    # Añadir comentarios al reporte
    elements.append(Paragraph("<b>Comentarios sobre los Resultados:</b>", styles['CenturyGothicBold']))
    elements.append(Spacer(1, 0.2 * inch))
    for comment in comments:
        elements.append(Paragraph(comment, styles['CenturyGothic']))
        elements.append(Spacer(1, 0.1 * inch))

    elements.append(Spacer(1, 0.1 * inch))
    
    # Conclusión
    elements.append(Paragraph(
        f"Este análisis es generado debido a que es crucial tomar medidas estratégicas para garantizar que  los candidatos estén bien preparados para el rol de {position}. Los aspirantes con alta concordancia deben ser considerados seriamente para el cargo, ya que están en una posición favorable para asumir responsabilidades significativas y contribuir al éxito del Capítulo. Aquellos con buena concordancia deberían continuar desarrollando su experiencia, mientras que los aspirantes con  baja concordancia deberían recibir orientación para mejorar su perfil profesional y acumular más  experiencia relevante. Estas acciones asegurarán que el proceso de selección se base en una evaluación completa y precisa de las capacidades de cada candidato, fortaleciendo la gestión y el  impacto del Capítulo.",
        styles['CenturyGothic']
    ))

    elements.append(Spacer(1, 0.2 * inch))

    # Mensaje de agradecimiento
    elements.append(Paragraph(
        f"Gracias, {candidate_name}, por tu interés en el cargo de {position} ¡Éxitos en tu proceso!",
        styles['CenturyGothic']
    ))

    # Configuración de funciones de fondo
    def on_first_page(canvas, doc):
        add_background(canvas, background_path)

    def on_later_pages(canvas, doc):
        add_background(canvas, background_path)

    # Construcción del PDF
    doc.build(elements, onFirstPage=on_first_page, onLaterPages=on_later_pages)

    # Descargar el reporte desde Streamlit
    with open(output_path, "rb") as file:
        st.success("Reporte detallado PDF generado exitosamente.")
        st.download_button(
            label="Descargar Reporte PDF",
            data=file,
            file_name=output_path,
            mime="application/pdf",
        )


# Interfaz en Streamlit
def home_page():
    st.title("Bienvenido a EvalHVAN")
    
    st.subheader("¿Qué tan listo estás para asumir un cargo de junta directiva Capitular? Descúbrelo aquí 🦁") 
    imagen_aneiap = 'Evaluador Hoja de Vida ANEIAP.jpg'
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
    if st.button("Generar Reporte PDF"):
        if uploaded_file is not None:
            with open("uploaded_cv.pdf", "wb") as f:
                f.write(uploaded_file.read())
            
            # Llamar a la función para generar el reporte con fondo
            generate_report_with_background("uploaded_cv.pdf", position, candidate_name, background_path)
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
    imagen_secundary= 'Analizador Versión Descriptiva.jpg'
    st.title("Evaluador de Hoja de Vida ANEIAP")
    st.image(imagen_secundary, use_container_width=True)
    st.subheader("Versión Descriptiva Hoja de Vida ANEIAP⏩")
    st.write("Sube tu hoja de vida ANEIAP (en formato PDF) para evaluar tu perfil.")

    # Entrada de datos del usuario
    candidate_name = st.text_input("Nombre del candidato:")
    detailed_uploaded_file = st.file_uploader("Sube tu hoja de vida ANEIAP en formato PDF", type="pdf")
    position = st.selectbox("Selecciona el cargo al que aspiras:", [
        "DCA", "DCC", "DCD", "DCF", "DCM", "CCP", "IC", "PC"
    ])

    if st.button("Generar Reporte PDF"):
        if detailed_uploaded_file is not None:
            with open("detailed_uploaded_cv.pdf", "wb") as f:
                f.write(detailed_uploaded_file.read())
            
            # Llamar a la nueva función unificada
            analyze_and_generate_descriptive_report_with_background("detailed_uploaded_cv.pdf", position, candidate_name, advice, indicators, background_path)


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
    link_label_plantilla = "Explorar plantilla"

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
