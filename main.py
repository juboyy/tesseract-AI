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
    st.title('OFM Extractor | llama 3.2')

    uploaded_file = st.file_uploader("Escolha uma imagem ou PDF...", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                # É um PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name

                # Converter PDF em imagens (sem inversão de cores)
                images_list = convert_pdf_to_images(tmp_file_path)

                # Extrair texto usando OCR das imagens originais
                ocr_text = extract_text_with_pytesseract(images_list)

                # Exibir a primeira página para visualização
                first_image_bytes = list(images_list[0].values())[0]
                st.image(first_image_bytes, caption='Primeira página do PDF carregado.', use_column_width=True)

                # Aplicar inversão de cores apenas à primeira imagem para o LLM
                enhanced_image_bytes = invert_image_color(first_image_bytes)

                # Codificar a imagem invertida em base64
                base64_image = encode_image(enhanced_image_bytes)

            else:
                # É uma imagem
                image_bytes = uploaded_file.getvalue()

                # Exibir a imagem original
                st.image(image_bytes, caption='Imagem recebida com sucesso.', use_column_width=True)

                # Realizar OCR na imagem original
                image = Image.open(BytesIO(image_bytes))
                ocr_text = image_to_string(image, lang='por')

                # Aplicar inversão de cores à imagem para o LLM
                enhanced_image_bytes = invert_image_color(image_bytes)

                # Codificar a imagem invertida em base64
                base64_image = encode_image(enhanced_image_bytes)

            # Definições dos campos do JSON
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
                        "Utilize tanto o conteúdo da imagem quanto o texto extraído via OCR fornecido abaixo. "
                        "Note que o OCR pode conter erros, então use a imagem como referência para validar as informações. "
                        "Preencha o JSON abaixo com os dados extraídos. "
                        "Não inclua descrições adicionais, apenas preencha o JSON seguindo exatamente a estrutura apresentada. "
                        "Sempre responda com o JSON, mesmo para os campos que não houverem correspondência. "
                        "Você não é um chat conversacional; irá apenas executar a tarefa mencionada e entregar um resultado.\n\n"
                        "Certifique-se de preencher todos os campos com as informações extraídas do documento fornecido e siga a estrutura exata para garantir a compatibilidade com o sistema de integração. "
                        "Por favor, concentre-se apenas nas informações do arquivo; não invente dados. Para os campos que não tiverem informação no arquivo, deixe vazio como \"\".\n\n"
                        f"{field_definitions}\n"
                        "Conteúdo extraído via OCR (pode conter erros):\n"
                        f"{ocr_text}\n\n"
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
                        "      },\n"
                        "      {\n"
                        "        \"indicadorImposto\": \"\",\n"
                        "        \"codigoReceita\": \"\",\n"
                        "        \"indicadorRetencao\": \"\",\n"
                        "        \"vlrBaseImposto\": 0,\n"
                        "        \"aliquotaImposto\": 0.0,\n"
                        "        \"vlrImposto\": 0.0\n"
                        "      },\n"
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

            # Adicionar a imagem invertida ao conteúdo da mensagem
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            )

            # Usar a API Groq para processar o conteúdo da mensagem
            with st.spinner('Processando a imagem...'):
                client = Groq()
                completion = client.chat.completions.create(
                    model="llama-3.2-90b-vision-preview",
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

            # Indicar que o processamento foi concluído
            st.success('Processamento concluído com sucesso!')

            # Exibir o JSON extraído
            st.write('Conteúdo extraído:')
            st.code(completion.choices[0].message.content, language='json')

        except Exception as e:
            st.error('Ocorreu um erro no processamento. Por favor, tente novamente.')
            st.write(str(e))

if __name__ == '__main__':
    main()
