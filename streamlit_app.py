import streamlit as st
import os
import requests

try:
    from llama_index.llms import OpenAI
    print("LlamaIndex está instalado correctamente.")
except ImportError as e:
    print(f"Error de importación: {e}")

# Configurar API Llama3
LLAMA3_API_KEY = "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"
llm = LlamaAPI(api_key=LLAMA3_API_KEY)

# Función para obtener rutas dinámicas
def get_file_paths(position):
    base_url = "https://raw.githubusercontent.com/EvalHV/main/"
    functions_path = f"{base_url}Funciones/F{position}.pdf"
    profile_path = f"{base_url}Perfiles/P{position}.pdf"
    return functions_path, profile_path

# Función para procesar PDF
def extract_text_from_pdf(pdf_path):
    response = requests.get(pdf_path)
    pdf_bytes = response.content
    pdf = PdfReader(pdf_bytes)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Función para generar reporte PDF
def generate_pdf_report(candidate_name, position, analysis_results, global_func_match, global_profile_match):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, f"ANALISIS DE HOJA DE VIDA - {position}", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, "Análisis Experiencia ANEIAP", ln=True)

    # Agregar análisis item por item
    for item, results in analysis_results.items():
        pdf.multi_cell(0, 10, f"{item}")
        pdf.multi_cell(0, 10, f"Porcentaje de concordancia con funciones del cargo: {results['func']}%")
        pdf.multi_cell(0, 10, f"Porcentaje de concordancia con perfil del cargo: {results['profile']}%\n")

    # Porcentajes globales
    pdf.cell(200, 10, f"Porcentaje global con funciones del cargo: {global_func_match}%", ln=True)
    pdf.cell(200, 10, f"Porcentaje global con perfil del cargo: {global_profile_match}%", ln=True)

    # Interpretación de resultados
    pdf.multi_cell(0, 10, f"Interpretación de resultados: ... [dependiendo del porcentaje]")

    # Conclusión
    pdf.multi_cell(0, 10, f"Este análisis es generado debido a que es crucial ... rol de {position}.")

    # Mensaje de agradecimiento
    pdf.multi_cell(0, 10, f"Muchas gracias {candidate_name} por tu interés en convertirte en {position}. ¡Éxitos en tu proceso!")

    # Guardar PDF
    output_file = f"Reporte_Analisis_{position}_{candidate_name}.pdf"
    pdf.output(output_file)
    return output_file

# UI en Streamlit
st.title("Evaluador de Hoja de Vida ANEIAP")
st.image("https://raw.githubusercontent.com/YourRepoName/main/assets/banner.jpg", use_column_width=True)
st.subheader("¿Qué tan listo estás para asumir un cargo de junta directiva Capitular? Descúbrelo aquí 🦁")
st.write("Con solo tu hoja de vida ANEIAP (en formato PDF) podrás averiguar qué tan preparado te encuentras para asumir un cargo dentro de la JDC-IC-CCP.")

# Entrada del usuario
position = st.selectbox("Selecciona el cargo:", ["PC", "DCA", "DCC", "DCD", "DCF", "DCM", "CCP", "IC"])
candidate_name = st.text_input("Ingresa tu nombre:")
uploaded_file = st.file_uploader("Sube tu hoja de vida (formato .pdf)", type=["pdf"])

if st.button("Generar Reporte"):
    if candidate_name and position and uploaded_file:
        # Obtener rutas dinámicas
        functions_path, profile_path = get_file_paths(position)

        # Extraer texto de los PDFs
        st.write("Analizando documentos...")
        job_functions_text = extract_text_from_pdf(functions_path)
        job_profile_text = extract_text_from_pdf(profile_path)

        # Simulación del análisis (aquí integras la API Llama3)
        resume_text = extract_text_from_pdf(uploaded_file)
        analysis_results = {"Item 1": {"func": 80, "profile": 75}, "Item 2": {"func": 65, "profile": 70}}  # Simulado
        global_func_match = 72
        global_profile_match = 73

        # Generar reporte PDF
        report_path = generate_pdf_report(candidate_name, position, analysis_results, global_func_match, global_profile_match)
        st.success("¡Reporte generado con éxito!")
        
        # Descargar PDF
        with open(report_path, "rb") as file:
            st.download_button("Descargar Reporte", file, file_name=report_path)
    else:
        st.error("Por favor completa todos los campos antes de generar el reporte.")
