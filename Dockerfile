FROM codait/max-base:v1.1.1
	
COPY requirements.txt /workspace

RUN pip install -r requirements.txt
 
COPY . /workspace
    
EXPOSE 5000

CMD python app.py
