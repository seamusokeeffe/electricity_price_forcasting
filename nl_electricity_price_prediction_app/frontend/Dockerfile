FROM python:3.9

RUN mkdir /frontend

COPY ./requirements.txt /frontend/requirements.txt

WORKDIR /frontend

RUN pip install -r requirements.txt

COPY . /frontend

EXPOSE 8501

CMD ["streamlit","run", "frontend.py"]