FROM tiangolo/uvicorn-gunicorn-starlette:python3.7


RUN pip install fastai aiohttp
RUN pip install jinja2
RUN pip install starlette
RUN pip install requests
RUN pip install aiofiles
RUN pip install uvicorn
RUN pip install python-multipart


COPY ./app /app

WORKDIR /app

EXPOSE 80
