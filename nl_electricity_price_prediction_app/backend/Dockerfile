FROM python:3.9

RUN mkdir /backend

COPY ./requirements.txt /backend/requirements.txt

WORKDIR /backend

RUN pip install -r requirements.txt

COPY . /backend

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000" ,"--reload"]