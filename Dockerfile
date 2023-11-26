FROM python:3.10-slim

ARG QUANT_METHOD
ENV QUANT_METHOD=${QUANT_METHOD}

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

#CMD python download_model.py $QUANT_METHOD -v
# Keep it running for login and testing
CMD python download_model.py $QUANT_METHOD -v && tail -f /dev/null

# docker cp /path/to/local/file your_container_id:/path/to/container/directory