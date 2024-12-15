import streamlit as st
from PIL import Image
import os

# Configura Streamlit
st.set_page_config(page_title="Evaluador de Hoja de Vida ANEIAP", layout="wide")

# Imagen para la interfaz
imagen_aneiap = 'Evaluador Hoja de Vida ANEIAP UNINORTE.jpg'

with st.container():
    st.subheader("¿Qué tan listo estás para asumir un cargo de junta directiva Capitular? Descúbrelo aquí :lion:")
    st.title("Evaluador de Hoja de Vida ANEIAP UNINORTE")
    st.write("Con solo tu hoja de vida ANEIAP (en formato .docx) podrás averiguar qué tan preparado te encuentras para asumir un cargo dentro de la JDC-IC-CCP.")
    st.image(imagen_aneiap, use_container_width=True)

# Función para cargar documentos de funciones y perfil
def load_job_docs(cargo):
    base_path = "/content/drive/MyDrive/HERRAMIENTAS EN COLLAB/EVALUADOR HOJA DE VIDA ANEIAP/"
    job_docs = {
        "PC": {"funciones": base_path + "CARGOS JUNTA/FPC.docx", "perfil": base_path + "CARGOS JUNTA/PPC.docx"},
        "DCA": {"funciones": base_path + "CARGOS JUNTA/FDCA.docx", "perfil": base_path + "CARGOS JUNTA/PDCA.docx"},
        "DCC": {"funciones": base_path + "CARGOS JUNTA/FDCC.docx", "perfil": base_path + "CARGOS JUNTA/PDCC.docx"},
        "DCD": {"funciones": base_path + "CARGOS JUNTA/FDCD.docx", "perfil": base_path + "CARGOS JUNTA/PDCD.docx"},
        "DCF": {"funciones": base_path + "CARGOS JUNTA/FDCF.docx", "perfil": base_path + "CARGOS JUNTA/PDCF.docx"},
        "DCM": {"funciones": base_path + "CARGOS JUNTA/FDCM.docx", "perfil": base_path + "CARGOS JUNTA/PDCM.docx"},
        "CCP": {"funciones": base_path + "CARGOS JUNTA/FCCP.docx", "perfil": base_path + "CARGOS JUNTA/PCCP.docx"},
        "IC": {"funciones": base_path + "CARGOS JUNTA/FIC.docx", "perfil": base_path + "CARGOS JUNTA/PIC.docx"}
    }

# Extracción del contenido de interés (Experiencia ANEIAP)
def extract_experience_aneiap(doc):
    """Extrae el texto de la sección 'EXPERIENCIA EN ANEIAP'."""
    text = ""
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    extract = False
    for para in paragraphs:
        if "EXPERIENCIA EN ANEIAP" in para.upper():
            extract = True
        elif extract and any(keyword in para.upper() for keyword in ["EVENTOS ORGANIZADOS", "ASISTENCIA A EVENTOS ANEIAP"]):
            break
        elif extract:
            text += para + "\n"
    return text.strip()

# Comparación de textos usando Llama3
def analyze_similarity(texts, reference_text, api_key):
    """Analiza la similitud de una lista de textos con un texto de referencia utilizando Llama3."""
    similarities = []
    for text in texts:
        similarity = llama3_similarity(text, reference_text, api_key)
        similarities.append(similarity)
    return similarities

# Generación del reporte
def generate_report(experience_text, func_text, profile_text, cargo, candidate_name, api_key):
    """Genera un reporte de análisis y lo exporta como archivo .docx."""
    # Separar los ítems de experiencia en ANEIAP
    experience_items = [item.strip() for item in experience_text.split('\n') if item.strip()]

    # Crear documento
    doc = Document()

    # Título del reporte
    title = doc.add_paragraph("REPORTE ANÁLISIS HOJA DE VIDA")
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    run_title = title.runs[0]
    run_title.font.bold = True
    run_title.font.size = Pt(16)
    run_title.font.name = "Century Gothic"

    # Evaluar cada ítem de la sección "EXPERIENCIA EN ANEIAP"
    item_similarities_func = []
    item_similarities_prof = []

    for item in experience_items:
        # Usar Llama3 para comparar cada ítem con las funciones y perfil
        similarity_func = analyze_similarity([item], func_text, api_key)[0]
        similarity_prof = analyze_similarity([item], profile_text, api_key)[0]
        item_similarities_func.append(max(similarity_func, 0.0))
        item_similarities_prof.append(max(similarity_prof, 0.0))

    # Promediar las similitudes para las funciones y el perfil
    avg_similarity_func = sum(item_similarities_func) / len(item_similarities_func) if item_similarities_func else 0
    avg_similarity_prof = sum(item_similarities_prof) / len(item_similarities_prof) if item_similarities_prof else 0

    # Incluir los resultados en el reporte
    doc.add_paragraph(f"Similitud promedio con las funciones: {avg_similarity_func * 100:.2f}%")
    doc.add_paragraph(f"Similitud promedio con el perfil: {avg_similarity_prof * 100:.2f}%")

    # Conclusión
    if avg_similarity_func < 0.5 or avg_similarity_prof < 0.5:
        doc.add_paragraph(f"- Baja Concordancia (< 0.50): El análisis indica que {candidate_name} tiene una baja concordancia con los requisitos del cargo "
            f"de {cargo} y el perfil buscado. Esto sugiere que aunque el aspirante posee algunas experiencias relevantes, su historial actual "
            f"no cubre adecuadamente las competencias y responsabilidades necesarias para este rol crucial en la prevalencia del Capítulo. "
            f"Se aconseja a {candidate_name} enfocarse en mejorar su perfil profesional y desarrollar las habilidades necesarias para el cargo. "
            f"Este enfoque permitirá a {candidate_name} alinear mejor su perfil con los requisitos del puesto en futuras oportunidades.")
    elif avg_similarity_func < 0.75 or avg_similarity_prof < 0.75:
        doc.add_paragraph(f"- Buena Concordancia (≥ 0.50): El análisis muestra que {candidate_name} tiene una buena correspondencia con las funciones "
            f"del cargo de {cargo} y el perfil deseado. Aunque su experiencia en la asociación es relevante, existe margen para mejorar. "
            f"{candidate_name} muestra potencial para cumplir con el rol crucial en la prevalencia del Capítulo, pero se recomienda que continúe "
            f"desarrollando sus habilidades y acumulando más experiencia relacionada con el cargo objetivo. Su candidatura debe ser considerada "
            f"con la recomendación de enriquecimiento adicional.")
    else:
        doc.add_paragraph(f"- Alta Concordancia (≥ 0.75): El análisis revela que {candidate_name} tiene una excelente adecuación con las funciones "
            f"del cargo de {cargo} y el perfil buscado. La experiencia detallada en su hoja de vida está estrechamente alineada con "
            f"las responsabilidades y competencias requeridas para este rol crucial en la prevalencia del Capítulo. La alta concordancia "
            f"indica que {candidate_name} está bien preparado para asumir este cargo y contribuir significativamente al éxito y la misión "
            f"del Capítulo. Se recomienda proceder con el proceso de selección y considerar a {candidate_name} como una opción sólida para el "
            f"cargo.")
    doc.add_paragraph(
        f"Este análisis es generado debido a que es crucial tomar medidas estratégicas para garantizar que los candidatos estén bien preparados "
        f"para el rol de {cargo}. Los aspirantes con alta concordancia deben ser considerados seriamente para el cargo, ya que están en una "
        f"posición favorable para asumir responsabilidades significativas y contribuir al éxito del Capítulo. Aquellos con buena concordancia "
        f"deberían continuar desarrollando su experiencia, mientras que los aspirantes con baja concordancia deberían recibir orientación para "
        f"mejorar su perfil profesional y acumular más experiencia relevante. Estas acciones asegurarán que el proceso de selección se base en una "
        f"evaluación completa y precisa de las capacidades de cada candidato, fortaleciendo la gestión y el impacto del Capítulo.")

    # Mensaje de agradecimiento
    doc.add_paragraph(f"\nMuchas gracias {candidate_name} por tu interés en convertirte en {cargo}. ¡Éxitos en tu proceso!").runs[0].font.name = "Century Gothic"
    format_paragraph(doc.paragraphs[-1])

    # Guardar el archivo
    doc.save(f"Reporte_{candidate_name}_{cargo}.docx")

# Inputs del usuario
with st.container():
    st.header("Carga tu información")
    candidate_name = st.text_input("Nombre del candidato:")
    cargo = st.selectbox("Selecciona el cargo al que aspiras:", ["PC", "DCA", "DCC", "DCD", "DCF", "DCM", "IC", "CCP"])
    cv_file = st.file_uploader("Carga tu hoja de vida ANEIAP (.docx):", type=["docx"])

# Extracción del contenido de interés (Experiencia ANEIAP)
def extract_experience_aneiap(doc):
    """Extrae el texto de la sección 'EXPERIENCIA EN ANEIAP'."""
    text = ""
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    extract = False
    for para in paragraphs:
        if "EXPERIENCIA EN ANEIAP" in para.upper():
            extract = True
        elif extract and any(keyword in para.upper() for keyword in ["EVENTOS ORGANIZADOS", "ASISTENCIA A EVENTOS ANEIAP"]):
            break
        elif extract:
            text += para + "\n"
    return text.strip()

# Procesar y mostrar resultados
if st.button("Evaluar"):
    if not candidate_name or not cargo or not cv_file:
        st.error("Por favor, llena todos los campos y carga tu hoja de vida.")
    else:
        st.info("Procesando tu información, por favor espera...")
        job_docs = load_job_docs(cargo)
        if not job_docs:
            st.error(f"No se encontraron documentos para el cargo {cargo}.")
        else:
            nlp = spacy.load("es_core_news_md")  # Cargar modelo de SpaCy
            similarity_score = analyze_cv(cv_file, job_docs["funciones"], job_docs["perfil"], nlp)
            st.success(f"Porcentaje de afinidad: {similarity_score * 100:.2f}%")
            report = generate_report(candidate_name, cargo, similarity_score)
            report_name = f"Reporte_Analisis_Hoja_de_Vida_{cargo}_{candidate_name}.docx"
            report.save(report_name)

            # Botón para descargar el reporte
            with open(report_name, "rb") as file:
                btn = st.download_button(
                    label="Descargar Reporte",
                    data=file,
                    file_name=report_name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )


