FROM quay.io/codait/max-base:v1.5.1

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD python app.py
