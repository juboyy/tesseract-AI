import os
from groq import Groq
import streamlit as st
from PIL import Image
import tempfile
import base64
import pypdfium2 as pdfium
from io import BytesIO
from pytesseract import image_to_string
import pytesseract
import cv2
import numpy as np
import json
from streamlit_ace import st_ace  # Importando o componente Ace
import re  # Importado para regex no parsing

# Configurar o caminho do Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Função para codificar bytes de imagem em base64
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# Função para inverter as cores da imagem
def invert_image_color(image_bytes):
    # Converter bytes de imagem em um array NumPy
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decodificar a imagem para o formato OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Inverter as cores
    inverted_image = 255 - image
    # Codificar a imagem invertida de volta para bytes
    _, buffer = cv2.imencode('.jpg', inverted_image)
    inverted_image_bytes = buffer.tobytes()
    return inverted_image_bytes

# Função para converter PDF em imagens (sem aplicar inversão de cores)
def convert_pdf_to_images(file_path, scale=300/72):
    pdf_file = pdfium.PdfDocument(file_path)
    page_indices = [i for i in range(len(pdf_file))]
    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )
    list_final_images = []
    for i, image in zip(page_indices, renderer):
        # Converter imagem PIL para bytes
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='JPEG', optimize=True)
        image_bytes = image_byte_array.getvalue()
        list_final_images.append({i: image_bytes})
    return list_final_images

# Função para extrair texto usando o Tesseract OCR (usando imagens originais)
def extract_text_with_pytesseract(list_dict_final_images):
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []
    for index, image_bytes in enumerate(image_list):
        # Abrir imagem a partir dos bytes
        image = Image.open(BytesIO(image_bytes))
        # Realizar OCR
        raw_text = str(image_to_string(image, lang='por'))
        image_content.append(raw_text)
    return "\n".join(image_content)

def main():
    st.set_page_config(layout="wide")  # Ajusta o layout para largura total
    st.title('OFM Extractor | llama 3.2')

    # Inicializar o estado do Streamlit para armazenar o JSON gerado e o último arquivo
    if 'generated_json' not in st.session_state:
        st.session_state.generated_json = ""
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    if 'images_list' not in st.session_state:
        st.session_state.images_list = []
    if 'ocr_text' not in st.session_state:
        st.session_state.ocr_text = ""

    # Obter a chave de API da variável de ambiente
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        st.error('A chave de API da Groq não foi encontrada. Verifique as configurações de segredos no Streamlit Cloud.')
        return

    # Inicializar o cliente Groq com a chave de API
    client = Groq(api_key=api_key)

    uploaded_file = st.file_uploader("Escolha uma imagem ou PDF...", type=["jpg", "jpeg", "png", "pdf"])

    # Verificar se um novo arquivo foi carregado
    if uploaded_file is not None:
        # Verificar se o arquivo atual é diferente do último carregado
        if st.session_state.last_uploaded_file != uploaded_file.name:
            # Atualizar o último arquivo carregado
            st.session_state.last_uploaded_file = uploaded_file.name
            # Limpar o JSON gerado anteriormente e outros estados
            st.session_state.generated_json = ""
            st.session_state.images_list = []
            st.session_state.ocr_text = ""
    else:
        # Se nenhum arquivo estiver carregado, limpar todos os estados
        if st.session_state.generated_json != "":
            st.session_state.generated_json = ""
        if st.session_state.last_uploaded_file is not None:
            st.session_state.last_uploaded_file = None
        if st.session_state.images_list != []:
            st.session_state.images_list = []
        if st.session_state.ocr_text != "":
            st.session_state.ocr_text = ""

    if uploaded_file is not None:
        try:
            with st.spinner('Processando o arquivo...'):
                if uploaded_file.type == "application/pdf":
                    # É um PDF
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_file_path = tmp_file.name

                    # Converter PDF em imagens (sem inversão de cores)
                    st.session_state.images_list = convert_pdf_to_images(tmp_file_path)

                    # Extrair texto usando OCR das imagens originais
                    st.session_state.ocr_text = extract_text_with_pytesseract(st.session_state.images_list)

                    # Obter a primeira imagem (página) do PDF
                    first_image_bytes = list(st.session_state.images_list[0].values())[0]

                    # Aplicar inversão de cores à primeira imagem para o LLM
                    enhanced_image_bytes = invert_image_color(first_image_bytes)

                    # Codificar as imagens (original e invertida) em base64
                    base64_image_original = encode_image(first_image_bytes)
                    base64_image_inverted = encode_image(enhanced_image_bytes)
                else:
                    # É uma imagem
                    image_bytes = uploaded_file.getvalue()

                    # Realizar OCR na imagem original
                    image = Image.open(BytesIO(image_bytes))
                    st.session_state.ocr_text = image_to_string(image, lang='por')

                    # Aplicar inversão de cores à imagem para o LLM
                    enhanced_image_bytes = invert_image_color(image_bytes)

                    # Codificar as imagens (original e invertida) em base64
                    base64_image_original = encode_image(image_bytes)
                    base64_image_inverted = encode_image(enhanced_image_bytes)

            # Definições dos campos do JSON (mantido igual)
            field_definitions = (
                "Definições dos campos do JSON a serem preenchidos:\n\n"
                "1. numeroRps (String) - Número do RPS que gerou a nota fiscal de saída de serviço. Obrigatório: Sim\n"
                "2. numeroNota (String) - Número da nota fiscal de saída de serviço. Obrigatório: Sim\n"
                "3. dataEmissao (String) - Data de emissão da nota fiscal de saída (Formato: DD/MM/YYYY HH24:MI:SS). Obrigatório: Sim\n"
                "4. codigoSerie (String) - Código da série da nota fiscal de serviço. Obrigatório: Não\n"
                "5. descricaoSerie (String) - Descrição da série da nota fiscal de serviço. Obrigatório: Não\n"
                "6. codigoModelo (String) - Código do modelo da nota fiscal de serviço. Obrigatório: Sim\n"
                "7. descricaoModelo (String) - Descrição do modelo da nota fiscal de serviço. Obrigatório: Não\n"
                "8. cnpjCliente (String) - CNPJ do cliente da nota fiscal de serviço. Obrigatório: Não\n"
                "9. razaoCliente (String) - Razão social do cliente da nota fiscal de serviço. Obrigatório: Não\n"
                "10. codIbgeEstadoServico (String) - Código IBGE do estado da execução do serviço. Obrigatório: Sim\n"
                "11. codIbgeCidadeServico (String) - Código IBGE da cidade da execução do serviço. Obrigatório: Sim\n"
                "12. tipoTributacaoIss (String) - Tipo de Tributação do ISS (1 a 9). Obrigatório: Sim\n"
                "   Valores possíveis:\n"
                "     1. Tributado no Município\n"
                "     2. Tributado fora do Município\n"
                "     3. Tributado no Município Isento\n"
                "     4. Tributado fora do Município Isento\n"
                "     5. Tributado no Município Imune\n"
                "     6. Tributado fora do Município Imune\n"
                "     7. Tributado no Município Suspensa\n"
                "     8. Tributado fora do Município Suspensa\n"
                "     9. Exp Servicos\n"
                "13. valorNotaFiscal (BigDecimal) - Valor da nota fiscal de serviço. Obrigatório: Sim\n"
                "14. valorMulta (BigDecimal) - Valor da multa na nota fiscal de serviço. Obrigatório: Não\n"
                "15. valorDesconto (BigDecimal) - Valor do desconto na nota fiscal de serviço. Obrigatório: Não\n"
                "16. termoRecebimento (String) - Descrição do termo de recebimento da nota fiscal de serviço integrado com o Fusion (Oracle). Obrigatório: Não\n"
                "17. observacao (String) - Descrição da observação da nota fiscal de serviço. Obrigatório: Sim\n\n"
                "Servicos:\n"
                "1. codigoTipoServico (String) - Código do Tipo de Serviço na nota fiscal. Obrigatório: Sim\n"
                "2. descricaoTipoServico (String) - Descrição do Tipo de Serviço na nota fiscal. Obrigatório: Sim\n"
                "3. codigoServico (String) - Código do Serviço na nota fiscal. Obrigatório: Sim\n"
                "4. descricaoServico (String) - Descrição do Serviço na nota fiscal. Obrigatório: Sim\n"
                "5. quantidadeServico (BigDecimal) - Quantidade do serviço na nota fiscal. Obrigatório: Sim\n"
                "6. valorServico (BigDecimal) - Valor do serviço na nota fiscal. Obrigatório: Sim\n"
                "7. valorTotalServico (BigDecimal) - Valor total do serviço na nota fiscal. Obrigatório: Sim\n\n"
                "ImpostosRetido (Se houver):\n"
                "1. indicadorImposto (String) - Tipo de imposto (ex: COFINS, PIS/PASEP, ISS, INSS-PJ, INSS-PF, IRRF-PF, IRRF-PJ, CSLL). Obrigatório se houver informação no documento.\n"
                "2. codigoReceita (String) - Código da Receita. Obrigatório: Não\n"
                "3. indicadorRetencao (String) - Indica se o imposto possui Retenção. Obrigatório se houver informação no documento.\n"
                "4. vlrBaseImposto (BigDecimal) - Valor base do Imposto. Obrigatório se houver informação no documento.\n"
                "5. aliquotaImposto (BigDecimal) - Alíquota do Imposto. Obrigatório se houver informação no documento.\n"
                "6. vlrImposto (BigDecimal) - Valor do Imposto. Obrigatório se houver informação no documento.\n\n"
                "Titulos:\n"
                "1. numeroTitulo (String) - Informar o número do título. Obrigatório: Não\n"
                "2. dataVencimento (Data) - Data de vencimento do título (Formato: DD/MM/YYYY). Obrigatório: Não\n"
                "3. cnpjCpfCredorTitulo (String) - CNPJ/CPF do credor do título. Obrigatório: Não\n"
                "4. valorTitulo (BigDecimal) - Valor do título. Obrigatório: Não\n"
                "5. indicadorTipoTitulo (String) - Tipo do título ('P' - Título do credor principal, 'R' - Título de retenção). Obrigatório: Não\n"
            )

            # Construir o conteúdo da mensagem com o entendimento de que os arquivos podem não seguir um padrão
            message_content = [
                {
                    "type": "text",
                    "text": (
                        "Por favor, analise o documento fornecido e extraia todas as informações relevantes necessárias para preencher o JSON abaixo. "
                        "Note que os arquivos fornecidos podem não seguir um padrão específico, portanto, é importante buscar as informações pertinentes para preencher o JSON, independentemente do formato do documento. "
                        "Utilize tanto o conteúdo das imagens (original e tratada) quanto o texto extraído via OCR fornecido abaixo. "
                        "Corrija todas as incongruências entre o OCR e as imagens para chegar ao melhor resultado possível. "
                        "Preencha o JSON abaixo com os dados extraídos. "
                        "Não inclua descrições adicionais, apenas preencha o JSON seguindo exatamente a estrutura apresentada. "
                        "Sempre responda **apenas com o JSON**, sem incluir qualquer texto adicional ou explicações.\n\n"
                        "Certifique-se de preencher todos os campos com as informações extraídas do documento fornecido e siga a estrutura exata para garantir a compatibilidade com o sistema de integração. "
                        "Por favor, concentre-se apenas nas informações do arquivo; não invente dados. Para os campos que não tiverem informação no arquivo, deixe vazio como \"\".\n\n"
                        f"{field_definitions}\n"
                        "Conteúdo extraído via OCR (pode conter erros):\n"
                        f"{st.session_state.ocr_text}\n\n"
                        "Estrutura do JSON a ser preenchido:\n"
                        "[\n"
                        "  {\n"
                        "    \"numeroRps\": \"\",\n"
                        "    \"numeroNota\": \"\",\n"
                        "    \"dataEmissao\": \"\",\n"
                        "    \"codigoSerie\": \"\",\n"
                        "    \"descricaoSerie\": \"\",\n"
                        "    \"codigoModelo\": \"\",\n"
                        "    \"descricaoModelo\": \"\",\n"
                        "    \"cnpjCliente\": \"\",\n"
                        "    \"razaoCliente\": \"\",\n"
                        "    \"codIbgeEstadoServico\": \"\",\n"
                        "    \"codIbgeCidadeServico\": \"\",\n"
                        "    \"tipoTributacaoIss\": \"\",\n"
                        "    \"valorNotaFiscal\": 0,\n"
                        "    \"valorMulta\": 0,\n"
                        "    \"valorDesconto\": 0,\n"
                        "    \"termoRecebimento\": \"\",\n"
                        "    \"observacao\": \"\",\n"
                        "    \"Servicos\": [\n"
                        "      {\n"
                        "        \"codigoTipoServico\": \"\",\n"
                        "        \"descricaoTipoServico\": \"\",\n"
                        "        \"codigoServico\": \"\",\n"
                        "        \"descricaoServico\": \"\",\n"
                        "        \"quantidadeServico\": 0,\n"
                        "        \"valorServico\": 0,\n"
                        "        \"valorTotalServico\": 0,\n"
                        "        \"cstSpedEfdSaida\": 0,\n"
                        "        \"aliqPisSpedEfdSaida\": 0.0,\n"
                        "        \"aliqCofinsSpedEfdSaida\": 0.0\n"
                        "      }\n"
                        "    ],\n"
                        "    \"CodigoReceita\": [\n"
                        "      {\n"
                        "        \"codigoReceita\": \"\"\n"
                        "      },\n"
                        "      {\n"
                        "        \"codigoReceita\": \"\"\n"
                        "      }\n"
                        "    ],\n"
                        "    \"ImpostosRetido\": [\n"
                        "      {\n"
                        "        \"indicadorImposto\": \"\",\n"
                        "        \"codigoReceita\": \"\",\n"
                        "        \"indicadorRetencao\": \"\",\n"
                        "        \"vlrBaseImposto\": 0,\n"
                        "        \"aliquotaImposto\": 0.0,\n"
                        "        \"vlrImposto\": 0.0\n"
                        "      }\n"
                        "    ],\n"
                        "    \"Titulos\": [\n"
                        "      {\n"
                        "        \"numeroTitulo\": \"\",\n"
                        "        \"dataVencimento\": \"\",\n"
                        "        \"cnpjCpfCredorTitulo\": \"\",\n"
                        "        \"valorTitulo\": 0,\n"
                        "        \"indicadorTipoTitulo\": \"\"\n"
                        "      }\n"
                        "    ]\n"
                        "  }\n"
                        "]"
                    )
                }
            ]

            # Adicionar as imagens (original e invertida) ao conteúdo da mensagem
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image_original}"
                    }
                }
            )
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image_inverted}"
                    }
                }
            )

            # Função para gerar o JSON usando a API Groq
            def generate_json():
                completion = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": message_content
                        }
                    ],
                    temperature=0,  # Definir como 0 para saída determinística
                    max_tokens=8000,  # Ajuste conforme necessário
                    top_p=1,
                    stream=False,
                    stop=None,
                )
                raw_response = completion.choices[0].message.content

                # Extrair apenas o JSON da resposta usando regex
                json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
                if json_match:
                    raw_json = json_match.group(0)
                else:
                    raw_json = raw_response  # Fallback se não encontrar padrão

                try:
                    # Parsear o JSON retornado para garantir que está válido
                    parsed_json = json.loads(raw_json)
                    # Reformatar o JSON com indentação
                    pretty_json = json.dumps(parsed_json, indent=4, ensure_ascii=False)
                    return pretty_json
                except json.JSONDecodeError as e:
                    st.error(f"Erro ao parsear o JSON retornado pela API: {e}")
                    return raw_json  # Retorna o JSON bruto mesmo que inválido

            # Se o JSON ainda não foi gerado, faça a geração inicial
            if st.session_state.generated_json == "":
                with st.spinner('Gerando o JSON...'):
                    st.session_state.generated_json = generate_json()

            # Exibir a imagem e o JSON lado a lado
            col1, col2 = st.columns([1, 1])  # Ajuste as proporções conforme necessário

            with col1:
                st.subheader('Documento')
                if uploaded_file.type == "application/pdf":
                    num_pages = len(st.session_state.images_list)
                    page_number = st.selectbox(
                        "Selecione a página do PDF para visualizar:",
                        options=range(1, num_pages + 1),
                        index=0
                    )
                    selected_image_bytes = list(st.session_state.images_list[page_number - 1].values())[0]
                    st.image(selected_image_bytes, caption=f'Página {page_number} do PDF carregado.', use_column_width=True)
                else:
                    st.image(image_bytes, caption='Imagem carregada.', use_column_width=True)

            with col2:
                st.subheader('JSON Editor')
                st.markdown('Para aplicar as alterações, utilize **CTRL+ENTER**.')  # Descrição adicionada

                # Botões lado a lado após a descrição
                col_buttons = st.columns(2)
                with col_buttons[0]:
                    # Botão para regenerar o JSON
                    if st.button('Regenerar JSON'):
                        with st.spinner('Regenerando o JSON...'):
                            st.session_state.generated_json = generate_json()
                            st.success('JSON regenerado com sucesso!')

                with col_buttons[1]:
                    # Botão para baixar o JSON
                    try:
                        parsed_json = json.loads(st.session_state.generated_json)
                        json_valid = True
                    except json.JSONDecodeError as e:
                        json_valid = False
                        st.error(f"O JSON fornecido não é válido: {e}")

                    if json_valid:
                        st.download_button(
                            label="Baixar JSON",
                            data=json.dumps(parsed_json, indent=4, ensure_ascii=False),
                            file_name='extracted_data.json',
                            mime='application/json'
                        )
                    else:
                        st.warning("Corrija os erros no JSON para habilitar o download.")

                # Editor de código Ace para edição do JSON com destaque de sintaxe
                edited_json = st_ace(
                    value=st.session_state.generated_json,
                    language='json',
                    theme='twilight',  # Você pode escolher outros temas disponíveis
                    key='ace_json_editor',
                    height=900,  # Aumentei a altura para melhor visualização
                    font_size=18,
                    show_gutter=True,
                    show_print_margin=True,
                    wrap=True,
                )

                # Atualizar o JSON no session_state se houver alterações
                if edited_json and edited_json != st.session_state.generated_json:
                    try:
                        parsed_json = json.loads(edited_json)
                        # Reformatar o JSON com indentação
                        st.session_state.generated_json = json.dumps(parsed_json, indent=4, ensure_ascii=False)
                    except json.JSONDecodeError as e:
                        # Se o JSON for inválido, manter o texto editado como está
                        st.session_state.generated_json = edited_json

        except Exception as e:
            st.error('Ocorreu um erro no processamento. Por favor, tente novamente.')
            st.write(str(e))

if __name__ == '__main__':
    main()
