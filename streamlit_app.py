import fitz  
import tensorflow_hub as hub
import numpy as np
import streamlit as st
from fpdf import FPDF

# Cargar Universal Sentence Encoder
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Función para calcular similitud con USE
def calculate_similarity_use(text1, text2):
    """Calcula la similitud entre dos textos usando Universal Sentence Encoder."""
    embeddings = model([text1, text2])
    similarity = np.inner(embeddings[0], embeddings[1])  # Producto interno
    return similarity * 100  # Escala a porcentaje

# Función para extraer la sección "EXPERIENCIA EN ANEIAP"
def extract_experience_section(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    start_keyword = "EXPERIENCIA EN ANEIAP"
    end_keywords = [
        "CONDICIONES ECONÓMICAS PARA VIAJAR",
        "EVENTOS ORGANIZADOS",
        "Asistencia a eventos ANEIAP",
        "Firma"
    ]
    start_idx = text.find(start_keyword)
    if start_idx == -1:
        return None

    end_idx = len(text)
    for keyword in end_keywords:
        idx = text.find(keyword, start_idx)
        if idx != -1:
            end_idx = min(end_idx, idx)

    return text[start_idx:end_idx].strip()

# Generar reporte en PDF
def generate_report(pdf_path, position, candidate_name):
    experience_text = extract_experience_section(pdf_path)
    if not experience_text:
        st.error("No se pudo extraer la sección 'EXPERIENCIA EN ANEIAP' del PDF.")
        return

    # Cargar archivos de funciones y perfil
    functions_path = f"Funciones//F{position}.pdf"
    profile_path = f"Perfiles/P{position}.pdf"

    try:
        with fitz.open(functions_path) as func_doc:
            functions_text = func_doc[0].get_text()

        with fitz.open(profile_path) as profile_doc:
            profile_text = profile_doc[0].get_text()
    except Exception as e:
        st.error(f"No se pudieron cargar los archivos de funciones o perfil: {e}")
        return

    # Evaluación renglón por renglón
    lines = experience_text.split("\n")
    line_results = []
    for line in lines:
        func_match = calculate_similarity_use(line, functions_text)
        profile_match = calculate_similarity_use(line, profile_text)
        line_results.append((line, func_match, profile_match))

    # Cálculo de concordancia global
    global_func_match = sum([res[1] for res in line_results]) / len(line_results)
    global_profile_match = sum([res[2] for res in line_results]) / len(line_results)

    # Crear reporte en PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Times", size=12)

    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(200, 10, txt=f"Reporte de Concordancia de {candidate_name} para el cargo de {position}", ln=True, align='C')
    pdf.ln(5)

    for line, func_match, profile_match in line_results:
        pdf.set_font("Arial", style="", size=12)
        pdf.multi_cell(0, 10, f"Item: {line}")
        pdf.multi_cell(0, 10, f"- Concordancia con funciones: {func_match:.2f}%")
        pdf.multi_cell(0, 10, f"- Concordancia con perfil: {profile_match:.2f}%")

    pdf.set_font("Arial", style="B", size=12)
    pdf.multi_cell(0, 10, "\nConcordancia Global:")
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"- Funciones: {global_func_match:.2f}%")
    pdf.multi_cell(0, 10, f"- Perfil: {global_profile_match:.2f}%")

    report_path = f"reporte_analisis_{position}_{candidate_name}.pdf"
    pdf.output(report_path, 'F')

    st.success("Reporte generado exitosamente.")
    st.download_button(
        label="Descargar Reporte", data=open(report_path, "rb"), file_name=report_path, mime="application/pdf"
    )

# Interfaz en Streamlit
imagen_aneiap = 'Evaluador Hoja de Vida ANEIAP UNINORTE.jpg'
st.title("Evaluador de Hoja de Vida ANEIAP")
st.image(imagen_aneiap, use_container_width=True)
st.subheader("¿Qué tan listo estás para asumir un cargo de junta directiva Capitular? Descúbrelo aquí 🦁")
st.write("Sube tu hoja de vida ANEIAP (en formato PDF) para evaluar tu perfil.")

# Interfaz de usuario
candidate_name = st.text_input("Nombre del candidato:")
uploaded_file = st.file_uploader("Sube tu hoja de vida ANEIAP en formato PDF", type="pdf")
position = st.selectbox("Selecciona el cargo al que aspiras:", [
    "DCA", "DCC", "DCD", "DCF", "DCM", "CCP", "IC", "PC"
])

if st.button("Generar Reporte"):
    if uploaded_file is not None:
        with open("uploaded_cv.pdf", "wb") as f:
            f.write(uploaded_file.read())
        generate_report("uploaded_cv.pdf", position, candidate_name)
    else:
        st.error("Por favor, sube un archivo PDF para continuar.")
