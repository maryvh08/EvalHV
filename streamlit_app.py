import fitz  # PyMuPDF para trabajar con PDFs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from fpdf import FPDF

# Palabras clave por posición
pos_keywords = {
    "DCA": ["Académico", "Conocimiento", "Formación", "Integral", "I+D+I", "Consultoria", "Entorno", "Liderazgo", "Directiva", "Capítulo", "Innovación", "Escuela", "Olimpiadas", "Taller", "FIC", "Habilidades", "Ingeolimpiadas", "Capacitación", "ANEIAP DAY", "SÉ", "Institucional", "Subdirector", "Subdirectora"],
    "DCC": ["Comunicaciones", "Redes", "Data", "Publicidad", "MIC", "Documental", "Youtube", "Biblioteca", "Podcast", "Directiva", "Capítulo", "Escuela", "Induspod", "Web,", "Journal", "Boletín", "Diseño", "Contenido", "IGTV", "Subdirector", "Subdirectora", "Piezas", "Tiktok", "Audiovisual"],
    "DCD": ["Desarrollo", "Relaciones", "Gala", "Integraciones", "Directiva", "Capítulo", "ANEIAP DAY", "Expansión", "Cultura", "Reclutamiento", "SÉ", "SRA", "Responsabilidad", "Insignia", "RSA", "Saloneos", "Gestión", "Subdirector", "Subdirectora"],
    "DCF": ["Finanzas", "Financiero", "Riqueza", "Sostenibilidad", "Directiva", "Capítulo", "Subdirector", "Subdirectora", "Obtención", "Recursos", "Recaudación", "Fondos", "Fuente", "Financiero"],
    "DCM": ["Mercadeo", "Branding", "Tienda", "Buzón", "Negocio", "Directiva", "Capítulo", "ANEIAP DAY", "Subdirector", "Subdirectora"],
    "CCP": ["Proyecto", "Project", "Innovación", "Asesor", "Sponsor", "CNI", "GNP", "Directiva", "Innova", "ECP", "PEN", "COEC", "Capítulo", "Equipo", "Manager", "Fraternidad", "Cambio", "Reforma", "Gestión", "Vida", "ANEIAP DAY", "Subcoordinador", "Subcoordinadora"],
    "IC": ["Interventoría", "Transparencia", "Normativa", "ECI", "Directiva", "Auditor", "IC", "ENI", "Capítulo", "Interventor", "Datos", "Data", "Análisis", "Veeduría"],
    "PC": ["Presidencia", "Estrategia", "Directiva", "Capítulo", "Presidente", "Directivo", "Junta", "ANEIAP DAY", "ECAP"]
}

# Función para extraer la sección "EXPERIENCIA EN ANEIAP" de un archivo PDF
def extract_experience_section(pdf_path):
    """
    Extrae la sección 'EXPERIENCIA ANEIAP' de un archivo PDF.
    Identifica el inicio por el subtítulo 'EXPERIENCIA ANEIAP' y el final por 'EVENTOS ORGANIZADOS'.
    Excluye renglones vacíos, subtítulos y elimina viñetas de los renglones.
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    
    # Palabras clave para identificar el inicio y final de la sección
    start_keyword = "EXPERIENCIA EN ANEIAP"
    end_keyword = "EVENTOS ORGANIZADOS"
    
    # Encuentra los índices de inicio y fin
    start_idx = text.find(start_keyword)
    if start_idx == -1:
        return None  # No se encontró la sección de experiencia

    end_idx = text.find(end_keyword, start_idx)
    if end_idx == -1:
        end_idx = len(text)  # Si no encuentra el final, usa el resto del texto

    # Extrae la sección entre el inicio y el fin
    experience_text = text[start_idx:end_idx].strip()
    
    # Limpia el texto: elimina subtítulos, renglones vacíos y viñetas
    experience_lines = experience_text.split("\n")
    cleaned_lines = []
    for line in experience_lines:
        line = line.strip()  # Elimina espacios en blanco al inicio y final
        if line and line not in [start_keyword, end_keyword]:  # Omite subtítulos y renglones vacíos
            # Elimina posibles viñetas
            line = line.lstrip("•-–—*")  # Elimina viñetas comunes al inicio del renglón
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

# Función para calcular la similitud usando TF-IDF y similitud de coseno
def calculate_similarity(text1, text2):
    """Calcula la similitud entre dos textos usando TF-IDF y similitud de coseno."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity * 100

# Generar reporte en PDF
def generate_report(pdf_path, position, candidate_name):
    """
    Genera un reporte en PDF basado en la comparación de la hoja de vida con funciones y perfil del cargo.
    Excluye ítems con 0% de concordancia en funciones y perfil al mismo tiempo.
    """
    experience_text = extract_experience_section(pdf_path)
    if not experience_text:
        st.error("No se pudo extraer la sección 'EXPERIENCIA ANEIAP' del PDF.")
        return

    # Verificar palabras clave específicas del cargo
    keywords = pos_keywords.get(position, [])
    lines = experience_text.split("\n")
    line_results = []

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

    for line in lines:
        if any(keyword.lower() in line.lower() for keyword in keywords):
            # Si contiene una palabra clave, la concordancia es 100%
            func_match = 100.0
            profile_match = 100.0
        else:
            # Calcular similitud normalmente
            func_match = calculate_similarity(line, functions_text)
            profile_match = calculate_similarity(line, profile_text)
        
        # Solo agregar al reporte si no tiene 0% en ambas métricas
        if func_match > 0 or profile_match > 0:
            line_results.append((line, func_match, profile_match))

    # Cálculo de concordancia global
    if line_results:  # Evitar división por cero si no hay ítems válidos
        global_func_match = sum([res[1] for res in line_results]) / len(line_results)
        global_profile_match = sum([res[2] for res in line_results]) / len(line_results)
    else:
        global_func_match = 0
        global_profile_match = 0

    func_score= (global_func_match*5)/100
    profile_score= (global_profile_match*5)/100

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
    
    pdf.ln(5)

    pdf.set_font("Arial", style="", size=12)
    for line, func_match, profile_match in line_results:
        pdf.multi_cell(0, 10, clean_text(f"Item: {line}"))
        pdf.multi_cell(0, 10, clean_text(f"- Concordancia con funciones: {func_match:.2f}%"))
        pdf.multi_cell(0, 10, clean_text( f"- Concordancia con perfil: {profile_match:.2f}%"))

    pdf.ln(5)
    
    pdf.set_font("Arial", style="B", size=12)
    pdf.multi_cell(0, 10, "\nConcordancia Global:")
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"- La concordancia global respecto a las funciones es: {global_func_match:.2f}%")
    pdf.multi_cell(0, 10, f"- La concordancia global respecto al Perfil es: {global_profile_match:.2f}%")

    #Puntaje global
    pdf.ln(5)
    pdf.multi_cell(0,10, f"- El puntaje respecto a las funciones de cargo es: {func_score}")
    pdf.multi_cell(0,10, f"- El puntaje respecto al perfil de cargo es: {profile_score}")

    pdf.ln(5)

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

    pdf.ln(5)

    # Conclusión
    pdf.multi_cell(0, 10, f"Este análisis es generado debido a que es crucial tomar medidas estratégicas para garantizar que  los candidatos estén bien preparados para el rol de {position}. Los aspirantes con alta concordancia deben ser considerados seriamente para el cargo, ya que están en una posición favorable para asumir responsabilidades significativas y contribuir al éxito del Capítulo. Aquellos con buena concordancia deberían continuar desarrollando su experiencia, mientras que los aspirantes con  baja concordancia deberían recibir orientación para mejorar su perfil profesional y acumular más  experiencia relevante. Estas acciones asegurarán que el proceso de selección se base en una evaluación completa y precisa de las capacidades de cada candidato, fortaleciendo la gestión y el  impacto del Capítulo.")

    pdf.ln(5)
    
    # Mensaje de agradecimiento
    pdf.multi_cell(0, 10, f"Muchas gracias {candidate_name} por tu interés en convertirte en {position}. ¡Éxitos en tu proceso!")

    # Guardar PDF
    report_path = f"Reporte_analisis_cargo_{position}_{candidate_name}.pdf"
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

# Entrada de datos del usuario
candidate_name = st.text_input("Nombre del candidato:")
uploaded_file = st.file_uploader("Sube tu hoja de vida ANEIAP en formato PDF", type="pdf")
position = st.selectbox("Selecciona el cargo al que aspiras:", [
    "DCA", "DCC", "DCD", "DCF", "DCM", "CCP", "IC", "PC"
])

# Botón para generar reporte
if st.button("Generar Reporte"):
    if uploaded_file is not None:
        with open("uploaded_cv.pdf", "wb") as f:
            f.write(uploaded_file.read())
        generate_report("uploaded_cv.pdf", position, candidate_name)
    else:
        st.error("Por favor, sube un archivo PDF para continuar.")
