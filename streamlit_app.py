import fitz  # PyMuPDF para trabajar con PDFs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from fpdf import FPDF
from collections import Counter
import re

# Datos extraídos del documento de planificación
indicators = {
    "DCA": {
        "Diseño académico": ["Académico", "Conocimiento", "Integral", "Directiva", "Capítulo", "Habilidades", "ANEIAP DAY", "SÉ", "Institucional", "Subdirector", "Subdirectora", "Blandas", "Duras", "Skills", "Académica", "Desarrollo"],
        "Innovación e investigación": ["I+D+I", "Consultoría", "Entorno", "Innovación", "Mentoría", "Ciclo","COEXPRO","Herramienta"],
        "Formación y capacitación": ["Formación", "Escuela", "Liderazgo", "Olimpiadas", "Taller", "FIC", "Ingeolimpiadas", "Capacitación", "Seminario", "Entrenamiento", "Cursos","CEA", "Profesional", "Aplicado"]
    },
    "DCC": {
        "Estrategia de comunicación": ["Comunicaciones", "Publicidad", "MIC", "Digital", "Campañas", "Promoción", "Difusión"],
        "Producción audiovisual": ["Redes", "Podcast", "Youtube", "Diseño", "Tiktok", "Audiovisual", "Contenido"],
        "Gestión de documental": ["Data", "Documental", "Biblioteca", "Documentación"]
    },
    "DCD": {
        "Gestión de asociados": ["Desarrollo", "Directiva", "Capítulo", "ANEIAP DAY", "Expansión", "Cultura", "Reclutamiento", "SÉ", "SRA", "Insignia", "Gestión", "Subdirector", "Subdirectora", "Equipos", "Contacto", "Retención"],
        "Integración y bienestar": ["Relaciones", "Gala", "Integraciones", "Premios", "Cohesión", "Personal", "Interpersonal","Talento","Humano","Plan","Incentivo","Clima","RRHH"],
        "Sostenimiento y sociedad": ["Responsabilidad", "RSA", "Social", "Ambiental", "Comunitario"]
    },
    "DCF": {
        "Gestión financieras": ["Finanzas", "Financiero", "Recursos", "Fondos", "Fuente", "Gestión", "Egreso", "Ingreso", "Ahorro", "Dashboard", "Sustentable"],
        "Sostenibilidad económica": ["Riqueza", "Sostenibilidad", "Obtención", "Recaudación", "Sostenimiento", "Económica", "Rentabilidad"],
        "Análisis  y transparencia": ["Directiva", "Capítulo", "Subdirector", "Subdirectora", "Donaciones"]
    },
    "DCM": {
        "Estrategias de brandings": ["Mercadeo", "Branding", "Negocio", "Posicionamiento", "Promoción", "Plan", "Campaña","Stakeholders","SGA","CRM","NPS","Indicador"],
        "Promoción y visibilidad": ["Buzón", "Directiva", "Capítulo", "ANEIAP DAY", "Subdirector", "Subdirectora", "Relaciones", "Visibilidad", "Identidad", "Visualización","Saloneo","Red","Expansión"],
        "Gestión comercial": ["Tienda", "Públicas", "Cliente", "Externo", "Interno", "Modelo", "Servicio", "Venta", "Comercial","EMPRENDE-IAP","Modelo", "Producto", "Servicio","Posicionamiento", "Entorno", "Crecimiento"]
    },
    "PC": {
        "Liderazgo y estrategia": ["Estrategia", "Directivo", "Liderazgo", "Rendimiento", "Decisiones", "Supervisión", "Transformación"],
        "Gestión organizacional": ["Presidencia", "Presidente", "Directiva", "Capítulo", "Junta", "ECAP", "Gestión", "Gestor"],
        "Relaciones y representación": ["Representante", "ANEIAP DAY", "Legal"]
    },
    "CCP": {
        "Gestión de proyectos": ["Proyecto", "Project", "Asesor", "Sponsor", "Equipo", "Manager", "Gestión", "Vida", "Subcoordinador", "Subcoordinadora", "Viabilidad", "Planificación", "Implementación"],
        "Innovación y creatividad": ["Innovación", "Innova", "Cambio", "Reforma", "ALMA", "Estructura", "Modelo", "Gobierno"],
        "Colaboración estratégica": ["CNI", "GNP", "Directiva", "ECP", "PEN", "COEC", "Capítulo", "Fraternidad", "ANEIAP DAY", "Organización", "Asesoramiento", "Indicadores", "Colaboración"]
    },
    "IC": {
        "Auditoría y control": ["Interventoría", "Normativa", "Auditor", "Interventor", "Datos", "Data", "Análisis", "Ética", "Revisión","Asesoría"],
        "Normativa y transparencia": ["Transparencia", "Reglamento", "Interventor", "Análisis financiero", "Veeduría","Conducto","Conducta", "Regular"],
        "Seguimiento y evaluación": ["ECI", "360", "Estratégica", "Directiva", "IC", "ENI", "Capítulo", "Interventor", "Rúbrica", "Indicadores de desempeño", "Seguimiento"]
    }
}

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

# Función para extraer la sección "EXPERIENCIA EN ANEIAP" de un archivo PDF
def extract_experience_section(pdf_path):
    """
    Extrae la sección 'EXPERIENCIA ANEIAP' de un archivo PDF.
    Detiene el análisis si encuentra el subtítulo 'EVENTOS ORGANIZADOS' o renglones irrelevantes como 
    'Reconocimientos individuales', 'Reconocimientos', y 'Reconocimientos grupales'.
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    
    # Palabras clave para identificar el inicio y final de la sección
    start_keyword = "EXPERIENCIA EN ANEIAP"
    end_keywords = [
        "EVENTOS ORGANIZADOS", 
        "Reconocimientos individuales", 
        "Reconocimientos", 
        "Reconocimientos grupales"
        "Reconocimientos"
    ]
    
    # Encuentra el índice de inicio
    start_idx = text.find(start_keyword)
    if start_idx == -1:
        return None  # No se encontró la sección de experiencia

    # Encuentra el índice más cercano de fin basado en los términos en end_keywords
    end_idx = len(text)  # Por defecto, toma hasta el final
    for keyword in end_keywords:
        idx = text.find(keyword, start_idx)
        if idx != -1:  # Si se encuentra el término, actualiza end_idx con el menor índice encontrado
            end_idx = min(end_idx, idx)

    # Extrae la sección entre el inicio y el fin
    experience_text = text[start_idx:end_idx].strip()

    # Lista de renglones a excluir (normalizados a minúsculas y sin espacios)
    exclude_lines = [
        "a nivel capitular",
        "a nivel nacional",
        "a nivel seccional",
        "reconocimientos individuales",
        "reconocimientos grupales",
        "nacional 2024"
        "cargos"
    ]
    
    # Limpia el texto: elimina renglones vacíos, subtítulos y viñetas
    experience_lines = experience_text.split("\n")
    cleaned_lines = []
    for line in experience_lines:
        line = line.strip()
        line = re.sub(r"[^\w\s]", "", line)  # Elimina caracteres no alfanuméricos excepto espacios
        normalized_line = re.sub(r"\s+", " ", line).lower()  # Normaliza espacios y convierte a minúsculas
        
        # Verificar si la línea es relevante
        if (
            normalized_line  # Línea no vacía
            and normalized_line not in exclude_lines  # No está en la lista de exclusión
            and normalized_line != start_keyword.lower()  # No es subtítulo de inicio
            and normalized_line not in [kw.lower() for kw in end_keywords]  # No es subtítulo de fin
        ):
            cleaned_lines.append(line)
    
    # Debugging: Imprime líneas procesadas
    print("Líneas procesadas:")
    for line in cleaned_lines:
        print(f"- {line}")
    
    return "\n".join(cleaned_lines)
    
def generate_advice(pdf_path, position):
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

# Función para calcular la similitud usando TF-IDF y similitud de coseno
def calculate_similarity(text1, text2):
    """Calcula la similitud entre dos textos usando TF-IDF y similitud de coseno."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity * 100

def calculate_presence(text, keywords):
    """
    Calcula el porcentaje de presencia de palabras clave en un texto.
    :param text: Texto donde se buscan las palabras clave.
    :param keywords: Lista de palabras clave a buscar.
    :return: Porcentaje de presencia de palabras clave.
    """
    words = text.split()
    count = sum(1 for word in words if word.lower() in [kw.lower() for kw in keywords])
    return (count / len(keywords)) * 100 if keywords else 0

def generate_report(pdf_path, position, candidate_name):
    """Genera un reporte en PDF basado en la comparación de la hoja de vida con funciones, perfil e indicadores."""
    experience_text = extract_experience_section(pdf_path)
    if not experience_text:
        st.error("No se encontró la sección 'EXPERIENCIA EN ANEIAP' en el PDF.")
        return

    position_indicators = indicators.get(position, {})
    indicator_results = Counter()
    lines = experience_text.split("\n")

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

        # Evaluación por palabras clave de indicadores
        for indicator, keywords in position_indicators.items():
            indicator_results[indicator] += calculate_presence(line, keywords)

        # Evaluación general de concordancia
        if any(keyword.lower() in line.lower() for kw_set in position_indicators.values() for keyword in kw_set):
            func_match = 100.0
            profile_match = 100.0
        else:
            # Calcular similitud normalmente
            func_match = calculate_similarity(line, functions_text)
            profile_match = calculate_similarity(line, profile_text)
        
        # Solo agregar al reporte si no tiene 0% en ambas métricas
        if func_match > 0 or profile_match > 0:
            line_results.append((line, func_match, profile_match))

    # Normalización de los resultados de indicadores
    total_presence = sum(indicator_results.values())
    if total_presence > 0:
        for indicator in indicator_results:
            indicator_results[indicator] = (indicator_results[indicator] / total_presence) * 100
            
    # Cálculo de concordancia global
    if line_results:  # Evitar división por cero si no hay ítems válidos
        global_func_match = sum([res[1] for res in line_results]) / len(line_results)
        global_profile_match = sum([res[2] for res in line_results]) / len(line_results)
    else:
        global_func_match = 0
        global_profile_match = 0
        
    # Identificar indicador menos presente
    lowest_indicator = min(indicator_results, key=indicator_results.get)
    lowest_percentage = indicator_results[lowest_indicator]

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

    # Resultados de indicadores
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(200, 10, txt=f"Análisis por Indicadores:", ln=True)
    pdf.set_font("Arial", style="", size=12)
    for indicator, percentage in indicator_results.items():
        pdf.cell(0, 10, f"- {indicator}: {percentage:.2f}%", ln=True)
    low_performance_indicators = {k: v for k, v in indicator_results.items() if v < 50.0}
    if low_performance_indicators:
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, "Consejos para Mejorar:", ln=True)
        pdf.set_font("Arial", size=12)
        for indicator, percentage in low_performance_indicators.items():
            pdf.cell(0, 10, f"- {indicator}: ({percentage:.2f}%)", ln=True)
            for tip in advice[position].get(indicator, []):
                pdf.cell(0, 10, f"  * {tip}", ln=True)

    #Concordancia global
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Concordancia Global:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"La concordancia Global de Funciones es: {global_func_match:.2f}%", ln=True)
    pdf.cell(0, 10, f"La concordancia Global de Perfil es: {global_profile_match:.2f}%", ln=True)

    #Puntaje global
    pdf.ln(5)
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

# Configuración BOTÓN GENERARR REPORTE
if st.button("Generar Reporte"):
    if uploaded_file is not None:
        with open("uploaded_cv.pdf", "wb") as f:
            f.write(uploaded_file.read())
        generate_report("uploaded_cv.pdf", position, candidate_name)
    else:
        st.error("Por favor, sube un archivo PDF para continuar.")

st.write(f"---")

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
