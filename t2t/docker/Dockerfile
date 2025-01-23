FROM tensorflow/tensorflow:1.10.0-devel-gpu-py3

RUN pip install -U pip && pip install --no-cache-dir \
        tensor2tensor==1.7.0

RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && apt-get clean

RUN pip install --no-cache-dir \
    setuptools jupyterhub jupyterlab notebook jupyter_server

COPY start-notebook.py start-notebook.sh start-singleuser.py start-singleuser.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start-notebook.py \
    /usr/local/bin/start-notebook.sh \
    /usr/local/bin/start-singleuser.py \
    /usr/local/bin/start-singleuser.sh

ENV PATH="/usr/local/bin:${PATH}"

CMD ["start-notebook.py", "--allow-root"]
