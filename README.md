**Install Docker**
then docker-compose up -build  


**Terminal Mistral AI Run**
docker-compose up -d

docker exec -it ollama bash

ollama pull mistral

ollama run mistral


_This is just to see Mistral LLM model is working properly or not._



**POST(Document Upload)**
http://localhost:8000/ingest

form-data
file = file.pdf

**GET (Search information)**
http://localhost:8000/query_stream?q=who is vicky chhetri
