 

docker build --tag "python_script" .

https://micromamba-docker.readthedocs.io/en/latest/advanced_usage.html#multiple-environments
description of env: 
envirronment.yml

for using in jupyter:
python -m ipykernel install --user --name=kantar_data_env2

For trust run, only  pg:
docker run -d -i --name postgres -p 5432:5432 -e POSTGRES_HOST_AUTH_METHOD=trust postgres