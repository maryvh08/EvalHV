import streamlit as st
import os
import requests
import PyPDF2
from llama_index import SimpleDirectoryReader, ServiceContext, LLMPredictor
from llama_index.prompts import Prompt
from llama_index.schema import Document
from llama_index.llms import LlamaAPI
import tempfile
from fpdf import FPDF

# Configuración inicial
LLAMA_API_KEY = "TU_LLAMA_API_KEY"  # Reemplaza con tu clave
GITHUB_REPO_URL = "https://github.com/usuario/repositorio"  # Reemplaza con tu repo

# Función para descargar archivos desde GitHub
def download_file_from_github(file_path):
    file_url = f"{GITHUB_REPO_URL}/raw/main/{file_path}"  # Ajusta si tu repositorio tiene otra estructura
    response = requests.get(file_url)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"No se pudo descargar el archivo {file_path}")
        return None

# Función para extraer texto de un PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name
    with open(tmp_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Función para comparar EXPERIENCIA ANEIAP con funciones y perfil
def analyze_resume(experiencia_text, funciones_text, perfil_text, llama_api):
    prompts = [
        Prompt(f"Compara la siguiente EXPERIENCIA ANEIAP: {experiencia_text} con las FUNCIONES: {funciones_text}"),
        Prompt(f"Compara la siguiente EXPERIENCIA ANEIAP: {experiencia_text} con el PERFIL: {perfil_text}")
    ]
    porcentajes = []
    for prompt in prompts:
        response = llama_api.complete(prompt.prompt_text)
        # Simulación: Parsear porcentaje de respuesta (realiza un ajuste más específico)
        porcentaje = float(response.text.split("%")[0].strip())
        porcentajes.append(porcentaje)
    return porcentajes

# Generar el reporte en PDF
def generate_pdf_report(candidate_name, position, experiencia_aneiap, analysis_results, global_func, global_prof):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"ANALISIS DE HOJA DE VIDA - {position}", ln=True, align="C")
    pdf.cell(200, 10, txt="Análisis Experiencia ANEIAP", ln=True)

    for item, (func, prof) in zip(experiencia_aneiap, analysis_results):
        pdf.multi_cell(0, 10, f"{item}\nPorcentaje de concordancia con funciones del cargo: {func}%\n"
                              f"Porcentaje de concordancia con perfil del cargo: {prof}%\n")

    pdf.cell(200, 10, txt=f"Porcentaje global de concordancia con funciones del cargo: {global_func}%", ln=True)
    pdf.cell(200, 10, txt=f"Porcentaje global de concordancia con perfil del cargo: {global_prof}%", ln=True)

    pdf.multi_cell(0, 10, "Este análisis es generado debido a que es crucial tomar medidas estratégicas para garantizar "
                          f"que los candidatos estén bien preparados para el rol de {position}...")

    pdf.cell(200, 10, txt=f"Muchas gracias {candidate_name} por tu interés en convertirte en {position}. ¡Éxitos en tu proceso!", ln=True)
    file_name = f"Reporte_Analisis_{position}_{candidate_name}.pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        return tmp_file.name, file_name

# Streamlit App
def main():
    st.title("Evaluador de Hoja de Vida ANEIAP")
    st.image(f"{GITHUB_REPO_URL}/raw/main/banner.png", use_column_width=True)
    st.subheader("¿Qué tan listo estás para asumir un cargo de junta directiva Capitular? Descúbrelo aquí 🦁")
    st.write("Con solo tu hoja de vida ANEIAP (en formato .pdf) podrás averiguar qué tan preparado te encuentras para asumir un cargo dentro de la JDC-IC-CCP.")

    # Formulario de entrada
    position = st.selectbox("Selecciona el cargo:", ["PC", "DCA", "DCC", "DCD", "DCF", "DCM", "CCP", "IC"])
    candidate_name = st.text_input("Ingresa tu nombre:")
    uploaded_resume = st.file_uploader("Sube tu hoja de vida en formato PDF:", type=["pdf"])

    if st.button("Analizar y Generar Reporte"):
        if position and candidate_name and uploaded_resume:
            # Descargar y procesar archivos desde GitHub
            funciones_file = download_file_from_github(f"F{position}.pdf")
            perfil_file = download_file_from_github(f"P{position}.pdf")

            if funciones_file and perfil_file:
                funciones_text = extract_text_from_pdf(funciones_file)
                perfil_text = extract_text_from_pdf(perfil_file)
                resume_text = extract_text_from_pdf(uploaded_resume)

                # Configurar Llama3 API
                llama_api = LlamaAPI(api_key=LLAMA_API_KEY)
                experiencia_aneiap = resume_text.split("\n")  # Dividir experiencia en ítems

                # Realizar análisis
                analysis_results = []
                for item in experiencia_aneiap:
                    porcentajes = analyze_resume(item, funciones_text, perfil_text, llama_api)
                    analysis_results.append(porcentajes)

                global_func = sum([x[0] for x in analysis_results]) / len(analysis_results)
                global_prof = sum([x[1] for x in analysis_results]) / len(analysis_results)

                # Generar reporte PDF
                pdf_path, file_name = generate_pdf_report(candidate_name, position, experiencia_aneiap, analysis_results, global_func, global_prof)
                st.success("¡Reporte generado exitosamente!")

                # Botón para descargar
                with open(pdf_path, "rb") as file:
                    st.download_button("Descargar Reporte", file, file_name)

        else:
            st.warning("Por favor, completa todos los campos antes de continuar.")

if __name__ == "__main__":
    main()

