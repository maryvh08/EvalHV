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

# Subir archivo
cv_file = st.file_uploader("Sube tu hoja de vida ANEIAP (formato .docx)", type="docx")

# Seleccionar cargo
cargo = st.selectbox("Selecciona el cargo al que aspiras:", ["PC", "DCA", "DCC", "DCD", "DCF", "DCM", "CCP", "IC"])

# Función para cargar documentos de funciones y perfil
def load_job_documents(cargo):
    base_path = "/content/drive/MyDrive/HERRAMIENTAS EN COLLAB/EVALUADOR HOJA DE VIDA ANEIAP/"
    documentos = {
        "PC": {"funciones": base_path + "funciones_pc.txt", "perfil": base_path + "perfil_pc.txt"},
        "DCA": {"funciones": base_path + "funciones_dca.txt", "perfil": base_path + "perfil_dca.txt"},
        "DCC": {"funciones": base_path + "funciones_dcc.txt", "perfil": base_path + "perfil_dcc.txt"},
        "DCD": {"funciones": base_path + "funciones_dcd.txt", "perfil": base_path + "perfil_dcd.txt"},
        "DCF": {"funciones": base_path + "funciones_dcf.txt", "perfil": base_path + "perfil_dcf.txt"},
        "DCM": {"funciones": base_path + "funciones_dcm.txt", "perfil": base_path + "perfil_dcm.txt"},
        "CCP": {"funciones": base_path + "funciones_ccp.txt", "perfil": base_path + "perfil_ccp.txt"},
        "IC": {"funciones": base_path + "funciones_ic.txt", "perfil": base_path + "perfil_ic.txt"},
    }

    job_document = documentos.get(cargo, None)
    
    # Verificar existencia de los archivos
    if job_document:
        funciones_path = job_document["funciones"]
        perfil_path = job_document["perfil"]
        
        # Verificar si los archivos existen
        if os.path.exists(funciones_path) and os.path.exists(perfil_path):
            print(f"Documentos encontrados para {cargo}")
            return job_document
        else:
            print(f"Error: Archivos no encontrados para el cargo {cargo}")
            return None
    else:
        print(f"No se encontró información para el cargo {cargo}")
        return None

# Función para analizar el CV (simulada)
def analyze_cv(cv_file, funciones, perfil, nlp=None):
    # Este es un espacio para la lógica de análisis del CV
    # Aquí se puede integrar cualquier análisis con NLP, como SpaCy
    return 85  # Ejemplo de puntuación de similitud (esto debería ser calculado)

# Función para generar el reporte (simulado)
def generate_report(candidate_name, cargo, similarity_score):
    # Aquí generas un reporte que puedes convertir en un archivo .docx
    report_content = f"Reporte de Evaluación\n\nCandidato: {candidate_name}\nCargo: {cargo}\nSimilitud: {similarity_score}%"
    return report_content

# Si el usuario sube su CV y selecciona un cargo
if cv_file and cargo:
    # Cargar documentos de funciones y perfil
    job_documents = load_job_documents(cargo)
    if job_documents:
        funciones_path = job_documents["funciones"]
        perfil_path = job_documents["perfil"]
        
        # Leer el contenido de los archivos
        try:
            with open(funciones_path, 'r') as f:
                funciones = f.read()
            with open(perfil_path, 'r') as p:
                perfil = p.read()

            # Mostrar contenido de funciones y perfil para verificar
            st.write("Contenido de Funciones:", funciones)
            st.write("Contenido de Perfil:", perfil)

            # Analizar hoja de vida
            similarity_score = analyze_cv(cv_file, funciones, perfil, nlp=None)

            # Generar y mostrar el reporte
            report = generate_report(candidate_name="Candidato", cargo=cargo, similarity_score=similarity_score)
            st.download_button("Descargar Reporte", report, file_name=f"reporte_{cargo}.txt")
        
        except Exception as e:
            st.error(f"Error al leer los archivos: {e}")
    else:
        st.error("No se encontraron documentos para el cargo seleccionado.")
