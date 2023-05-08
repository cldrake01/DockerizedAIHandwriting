FROM ubuntu:latest
LABEL authors="collindrake"

ENTRYPOINT ["top", "-b"]

EXPOSE 5000

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "main.py"]