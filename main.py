import streamlit as st
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Configuração da página
st.set_page_config(
    page_title="Classificador de Gatos e Cachorros",
    page_icon="😺🐶",
    layout="centered"
)

# Estilo CSS para melhorias visuais
st.markdown(
    """
    <style>
        .stApp {
            background-color: #F9F9F9;
        }
        .title {
            text-align: center;
            color: #4B9CD3;
        }
        .stFileUploader > label {
            color: #4B9CD3;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Centralizar título com multilíngue
lang = st.selectbox("Escolha o idioma | Choose your language", ["Português", "English"])

if lang == "English":
    st.markdown("<h1 class='title'>Cat 😺 and Dog 🐶 Classifier</h1>", unsafe_allow_html=True)
    st.write("<br>Upload an image to classify as Cat or Dog!", unsafe_allow_html=True)
    reset_text = "Reset"
    confidence_text = "Confidence"
    tips_header = "Tips for Better Results"
    tips = """
    - Upload clear images for best results.  
    - This model was trained only on Cat and Dog images.  
    - If the result is unclear, try a different image.
    """
else:
    st.markdown("<h1 class='title'>Classificador de Gatos 😺 e Cachorros 🐶</h1>", unsafe_allow_html=True)
    st.write("<br>Faça o upload de uma imagem para classificar como Gato ou Cachorro!", unsafe_allow_html=True)
    reset_text = "Reiniciar"
    confidence_text = "Confiança"
    tips_header = "Dicas de Uso"
    tips = """
    - Carregue imagens nítidas para obter melhores resultados.  
    - O modelo foi treinado apenas com imagens de Gatos e Cachorros.  
    - Caso o resultado não seja claro, tente outra imagem.
    """



# Carregar o modelo treinado com cache
@st.cache_resource
def load_classification_model():
    model_path = "modelo_CatsAndDogs.h5"
    if not os.path.exists(model_path):
        st.error("Erro: O modelo não foi encontrado. Verifique o caminho do arquivo.")
        st.stop()
    return load_model(model_path)

model = load_classification_model()

# Histórico de previsões
if "history" not in st.session_state:
    st.session_state["history"] = []

# Função para processar a imagem
def process_image(uploaded_image):
    img = Image.open(uploaded_image).convert('RGB')
    img = img.resize((128, 128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.write("---")
with st.expander(tips_header, expanded=False):
    st.write(f"""
    <div style="font-size: 14px; color:rgb(0, 0, 0);">
        {tips}
    </div>
    """, unsafe_allow_html=True)

# Upload de imagem
uploaded_image = st.file_uploader(
    "Envie uma imagem (JPG, JPEG ou PNG)", type=["jpg", "jpeg", "png"]
)

# Inferência e resultados
if uploaded_image is not None:
    st.image(uploaded_image, caption="Imagem Carregada", use_container_width=True)
    with st.spinner("Processando a imagem... 🚀"):
        time.sleep(1)

        # Processar imagem e realizar predição
        img_array = process_image(uploaded_image)
        prediction = model.predict(img_array)
        class_name = "Cachorro 🐶" if prediction[0][0] >= 0.5 else "Gato 😺"
        confidence = prediction[0][0] if prediction[0][0] >= 0.5 else 1 - prediction[0][0]

        # Salvar histórico
        st.session_state["history"].append(
            {"nome_arquivo": uploaded_image.name, "resultado": class_name, "confiança": confidence}
        )

        # Resultado
        emoji = "🐶" if class_name == "Cachorro 🐶" else "😺"
        st.success(f"### {class_name} {emoji}")
        st.info(f"**{confidence_text}:** {confidence * 100:.2f}%")

        # Gráfico de barras
        fig, ax = plt.subplots()
        labels = ["Gato", "Cachorro"]
        probs = [1 - prediction[0][0], prediction[0][0]]
        ax.bar(labels, probs, color=["#FF9999", "#66B3FF"])
        ax.set_ylim(0, 1)
        ax.set_ylabel(confidence_text)
        ax.set_title("Probabilidade de Predição")
        st.pyplot(fig)

# Histórico de previsões
if len(st.session_state["history"]) > 0:
    st.write("---")
    st.subheader("📜 Histórico de Predições")
    for item in st.session_state["history"]:
        st.write(f"**{item['nome_arquivo']}** - {item['resultado']} ({item['confiança']*100:.2f}%)")


# Rodapé centralizado e estilizado
st.write("---")
st.markdown(
    """
    <div style="text-align: center; font-family: 'Arial', sans-serif; padding: 15px; border-radius: 10px;">
        <p style="font-size: 16px; color: #333333; font-weight: bold;">Desenvolvido com ❤️ por 
            <a href="https://github.com/pablosmolak" target="_blank" style="color: #FF6F61; font-weight: bold;">Pablo Smolak</a> 
            usando Streamlit e TensorFlow.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
