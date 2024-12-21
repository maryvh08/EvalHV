import fitz  # PyMuPDF para trabajar con PDFs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from fpdf import FPDF
from collections import Counter

# Datos extraídos del documento de planificación
indicators = {
    "DCC": {
        "Estrategia de comunicación": ["Comunicaciones", "Publicidad", "MIC", "Digital", "Campañas", "Promoción", "Difusión"],
        "Producción audiovisual": ["Redes", "Podcast", "Youtube", "Diseño", "Tiktok", "Audiovisual", "Contenido"],
        "Gestión de documental": ["Data", "Documental", "Biblioteca", "Documentación"]
    },
    # Agregar otros cargos con sus indicadores y palabras clave aquí...
}

advice = {
    "DCC": {
        "Estrategia de comunicación": [
            "Trabaja en tus habilidades de redacción y storytelling.",
            "Gestiona relaciones públicas para ampliar la visibilidad del capítulo."
        ],
        "Producción audiovisual": [
            "Domina herramientas de diseño gráfico y edición audiovisual.",
            "Fomenta la creación de contenido multimedia atractivo."
        ],
        "Gestión de documental": [
            "Participa en iniciativas relacionadas con la gestión documental.",
            "Infórmate acerca del manejo de datos y documentación."
        ]
    },
    # Agregar consejos para otros cargos aquí...
}

def calculate_presence(text, keywords):
    """Calcula el porcentaje de presencia de palabras clave en un texto."""
    words = text.split()
    count = sum(1 for word in words if word in keywords)
    return (count / len(keywords)) * 100 if keywords else 0

def extract_experience_section(pdf_path):
    """Extrae la sección EXPERIENCIA EN ANEIAP de un PDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    start_keyword = "EXPERIENCIA EN ANEIAP"
    end_keyword = "EVENTOS ORGANIZADOS"
    start_idx = text.find(start_keyword)
    end_idx = text.find(end_keyword, start_idx)
    return text[start_idx:end_idx].strip() if start_idx != -1 and end_idx != -1 else None

ddef generate_advice(pdf_path, position):
    """Genera consejos basados en la evaluación de indicadores."""
    experience_text = extract_experience_section(pdf_path)
    if not experience_text:
        st.error("No se encontró la sección 'EXPERIENCIA EN ANEIAP' en el PDF.")
        return

    # Obtener indicadores y palabras clave para el cargo seleccionado
    position_indicators = indicators.get(position, {})
    results = {}

    for indicator, keywords in position_indicators.items():
        results[indicator] = calculate_presence(experience_text, keywords)

    # Identificar el indicador con menor presencia
    lowest_indicator = min(results, key=results.get)
    st.write(f"Indicador con menor presencia: {lowest_indicator} ({results[lowest_indicator]:.2f}%)")

    # Mostrar los consejos asociados al indicador
    st.write("Consejos para mejorar:")
    for tip in advice[position][lowest_indicator]:
        st.write(f"- {tip}")

def generate_report(pdf_path, position, candidate_name):
    """
    Genera un reporte en PDF basado en la evaluación de indicadores, concordancia y consejos personalizados.
    """
    experience_text = extract_experience_section(pdf_path)
    if not experience_text:
        st.error("No se encontró la sección 'EXPERIENCIA EN ANEIAP' en el PDF.")
        return

    # Obtener indicadores y palabras clave para el cargo seleccionado
    position_indicators = indicators.get(position, {})
    indicator_results = {}

    # Calcular la presencia de palabras clave por indicador
    for indicator, keywords in position_indicators.items():
        indicator_results[indicator] = calculate_presence(experience_text, keywords)

    # Identificar el indicador con menor presencia
    lowest_indicator = min(indicator_results, key=indicator_results.get)
    lowest_indicator_percentage = indicator_results[lowest_indicator]

    # Verificar palabras clave específicas del cargo
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

    func_score= round((global_func_match*5)/100,2)
    profile_score= round((global_profile_match*5)/100,2)

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

        # Resultados de indicadores
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Resultados por Indicador:", ln=True)
    for indicator, percentage in indicator_results.items():
        pdf.cell(0, 10, f"- {indicator}: {percentage:.2f}%", ln=True)
    pdf.ln(5)

    # Indicador con menor presencia
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, f"Indicador con menor presencia: {lowest_indicator} ({lowest_indicator_percentage:.2f}%)", ln=True)
    pdf.ln(5)

    # Consejos personalizados
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Consejos para Mejorar:", ln=True)
    for tip in advice[position][lowest_indicator]:
        pdf.cell(0, 10, f"- {tip}", ln=True)

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
if st.button("Generar Consejos"):
    if uploaded_file is not None:
        with open("uploaded_cv.pdf", "wb") as f:
            f.write(uploaded_file.read())
        generate_advice("uploaded_cv.pdf", position)
    else:
        st.error("Por favor, sube un archivo PDF para continuar.")
