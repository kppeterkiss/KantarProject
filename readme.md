# outlier detection in data + relational scheme creation in ```.dta``` files:
Create python env:
```conda env create  --file=environments.yml```

create ```./analysis/figures``` and ```./analysis/results``` directories if needed

Add input files to ```./FullKantarData/``` 

run ```./analysis/simple.ipynb``` for outlier detection, and data separation.



# Kanatar database infrastructure 

Setting up database infrastructure:

```docker compose up```


- Service *python* 
  - creates conda env
  - reads input csv
  - creates relational scheme
  - load new data incrementally in postgres

- Service *[metabase](https://www.metabase.com)*: 
   
  - http://localhost:3000
- Service *adminer*
  - http://localhost:8081
- Service *db* 
  - target db of the transformation
  - postgres:postgres for demo


# notes
 

docker build --tag "python_script" .

https://micromamba-docker.readthedocs.io/en/latest/advanced_usage.html#multiple-environments
description of env: 
envirronment.yml

for using in jupyter:
python -m ipykernel install --user --name=kantar_data_env2

For trust run, only  pg:
docker run -d -i --name postgres -p 5432:5432 -e POSTGRES_HOST_AUTH_METHOD=trust postgres
