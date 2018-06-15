FROM python:latest

ARG BUILD_ENV=release
ARG WORKDIR="/var/task"
ARG VENVDIR="/var/venv"

WORKDIR ${WORKDIR}

RUN /bin/bash -c "mkdir -p ${VENVDIR} && python3 -m venv ${VENVDIR}"

COPY . ${WORKDIR}

RUN /bin/bash -c "source ${VENVDIR}/bin/activate && \
  pip install --upgrade pip && \
  pip install -r requirements/${BUILD_ENV}.txt && \
  python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger && \
  deactivate"

ENTRYPOINT [ "./.docker/scripts/entrypoint.sh" ]
