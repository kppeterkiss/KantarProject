FROM ubuntu:latest
LABEL authors="Administrador"


RUN conda env create -f environment.yml
CMD jupyter nbconvert --to notebook --inplace --execute data_to_pg.ipynb
ENTRYPOINT ["top", "-b"]