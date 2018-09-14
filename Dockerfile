FROM codait/max-base
	
COPY requirements.txt /workspace

RUN pip install -r requirements.txt
 
COPY . /workspace
    
EXPOSE 5000

CMD python app.py
