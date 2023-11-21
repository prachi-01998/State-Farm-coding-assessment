FROM continuumio/miniconda3

WORKDIR /MLE_26

COPY . /MLE_26

RUN conda create --name my_envi --file requirements.txt --yes

EXPOSE 1313

ENV NAME my_envi

# Set Docker API version
ENV DOCKER_API_VERSION=1.43

CMD ["conda", "run", "--no-capture-output", "-n", "my_envi", "python", "api.py"]