FROM python:3.10
WORKDIR /workdir
COPY . /workdir
RUN pip install dependencies/botocore-1.29.162-py3-none-any.whl dependencies/boto3-1.26.162-py3-none-any.whl
RUN pip install -r requirements.txt
EXPOSE 7860
ENTRYPOINT [ "python" , "app.py" ]

