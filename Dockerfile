FROM python:3.10
RUN pip install gradio boto3 sagemaker ai21[SM]
COPY app.py /workdir/app.py
WORKDIR /workdir
EXPOSE 7860
ENTRYPOINT [ "python" , "app.py" ]

