FROM python:3.6.8
ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
EXPOSE 3002
CMD ["python", "./serve.py"]
