FROM python:3.11.2-slim-buster
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD [ "python", "test.py"]