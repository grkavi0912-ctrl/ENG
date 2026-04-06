#Use a minimal base image with python 3.9 installed
FROM python:3.9

#set the working directory inside the container to /app
WORKDIR /app

#copy all the files from the current directory on the host to the container's /app directory
COPY . .

#Install python dependencies listed in requirements.text
RUN pip install -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
   PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

#Define the command to run the Streamlit app on port "8501" and make itaccessible externally
CMD ["streamlit","run","app.py","--server.port=8501", "--server.address=0.0.0.0","--server.enableXsrfProtection=False"]
