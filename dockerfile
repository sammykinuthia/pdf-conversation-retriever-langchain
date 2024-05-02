# generate dockerfile for this python project

FROM python:3.10

WORKDIR /usr/src/app

RUN pip install --no-cache-dir langchain langchain-openai langchain-community pypdf langchain-chroma streamlit
RUN pip install --no-cache-dir  unstructured

COPY . .

EXPOSE 8501

CMD ["streamlit","run", "main.py"]
