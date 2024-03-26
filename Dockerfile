FROM --platform=linux/amd64 python:3.10 as build
WORKDIR /workdir
COPY . /workdir
RUN pip install -r requirements.txt
EXPOSE 7860
ENTRYPOINT [ "python" , "app.py" ]

