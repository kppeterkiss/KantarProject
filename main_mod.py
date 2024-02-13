# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# %%
import os
import pandas as pd
import numpy as np
import psycopg2

result_dir = './results/'
data_dir = './KantarData/'

figure_dir = "./figures/"
# os.listdir(data_dir)

test = False
local_test_no_db = False
local_test_no_auth=False

# %%
read_able_columns = pd.read_excel(data_dir + 'Explicacion_VariablesV3.xlsx', engine='openpyxl')

#data_dir = './test_data/'
# %%

# %%

# codevf vs coproduct??
product_attributes = ['CodProducto', 'CodigoBarras', 'CodVF', 'VF', 'SA7_BPL', 'SA4_Fabricante', 'SA3_Marca',
                      'SA2_Submarca',
                      'X102_Aditivos', 'X108_Usuario', 'X109_Estado', 'X112_Niv_Calorias',
                      'X124_Estilo', 'X127_Tipo_Bañado_Choc', 'X129_Tipo_Pasteleria',
                      'X135_Uso', 'X143_Tipo_Corte', 'X163_Info_Cafeina',
                      'X167_Grado_Curacion', 'X169_Cont_Materia_Grasa', 'X174_Despiece.Corte',
                      'X182_Tipo_Queso', 'X188_Tipo_Pan', 'X189_Tipo_Viena',
                      'X206_Tipo_Pescado', 'X230_Niv_Concentracion',
                      'X314_Tipo_Cena.Entrante', 'X315_Tipo_Carne', 'X323_Tipo_Sazonamiento',
                      'X328_Tipo_Aceite', 'X329_Pureness_Niveau', 'X360_Tipo_Verdura',
                      'X366_Tipo_Postre', 'X383_Tipo_Yogurt', 'X384_Tipo_Cena',
                      'X489_Como_Compro.', 'X490_Fresca.Congelada', 'X491_Presentacion',
                      'X497_Tipo_Queso', 'X498_Marcas_Queso', 'X500_Tipo_Embutido',
                      'X514_Tipo_Carne', 'X527_Tipo_Cafe', 'X529_Tipo_Producto',
                      'X531_Info_Biologica', 'X558_Variedad_Queso', 'X571_Peso_Bebe',
                      'X674_Tipo_Helado', 'X901_Localizacion', 'X902_Tipo_Producto',
                      'X903_Envase', 'X904_Variedad.Sabor', 'X5_Num_Tot_Unidades',
                      'X18_Num_Tot_Paquetes.Botes', 'X66_Preferred_Unit_Measure', ]

purchase_attributes = ['idCompra', 'CodPanelista', 'CodigoBarras', 'CodProducto', 'FechaCesta', 'PrecioCompra',
                       'Cantidad', 'Precio', 'CodLugarCompra', 'CodIndividuo', 'TipoMaquina', 'Promo_Folleto',
                       'Promo_Envase', 'Promo_TPR', 'promo']

shop_attributes = ['CodLugarCompra', 'Lugar_Compra', 'Canal', 'CodIndividuo']

customer_attributes = ['CodPanelista', 'CC_AA', 'Edad_Ama', 'NF', 'Numero_Gatos', 'Numero_Perros', 'Presencia_Niños',
                       'CodIMC', 'IMC', 'Ponderacion', 'Ciclo_Vida', 'Clase_EGM', 'Clase_Social',
                       'Habitat_Metropolitano',
                       'Habitat_Municipal_Std', 'Inmigrante', 'Provincia', 'Region', ]

tables = {'customers': customer_attributes, 'products': product_attributes, 'shops': shop_attributes,
          'purchases': purchase_attributes}
# %%
# define mapping of types
type_dict = dict(zip(read_able_columns['Variable'].str.strip(), read_able_columns['Tipo Variable'].str.strip()))

type_translation = {'ID': 'int64', 'Fecha': 'datetime', 'Numérica': 'float64', 'Categórica': 'category',
                    'Dicotómica': 'bool'}
type_dict_new = {k: type_translation[v] for k, v in type_dict.items()}
type_dict_new = {('X' + k if k[0].isdigit() else k): v for k, v in type_dict_new.items()}
type_dict_new['CodigoBarras'] = 'str'

type_dict_new['Presencia_Niños'] = 'category'

# when reading, should be given...
del type_dict_new['FechaCesta']
# because stata cannot handle too many categories with long names:
for cn in product_attributes:
    if type_dict_new[cn] == 'category':
        type_dict_new[cn] = 'str'


# %%
def general_model(df):
    if df.shape[0] > 30:
        # for throwing out outliers iteratively if we like, to have a better regression
        cleaning_rounds = 1
        Y = np.array(df['price_per_unit'])
        X = np.array(df['t'])
        # flag=np.zeros(y.size)
        idx = df[['price_per_unit']].index.copy()
        x = X.copy()
        y = Y.copy()
        print(x)
        for i in range(cleaning_rounds):
            p = np.polyfit(x, y, deg=5)
            ps = np.polyval(p, x)
            diff = abs(y - ps)
            std = np.std(diff)
            mean = np.mean(diff)
            good = abs(diff - mean) / std < 3
            # good = abs(y - ps) < 1  # Here we will only remove positive outliers

            x_bad, y_bad = x[~good], y[~good]
            x, y = x[good], y[good]

            if (~good).sum() == 0:
                break
        ps = np.polyval(p, X)
        diff = abs(Y - ps)
        std = np.std(diff)
        mean = np.mean(diff)
        z = abs(diff - mean) / std
        good = z < 3
        flag = np.logical_not(good)

        ret = pd.DataFrame(index=idx, data=list(zip(ps, z, flag)), columns=['1', '2', '3'])
    else:
        Y = np.array(df['price_per_unit'])
        # flag=np.zeros(y.size)
        idx = df[['price_per_unit']].index.copy()
        y = Y.copy()

        std = np.std(Y)
        mean = np.mean(Y)
        ps = np.full(y.shape, mean)

        z = abs(Y - mean) / std
        good = z < 3
        flag = np.logical_not(good)

        ret = pd.DataFrame(index=idx, data=list(zip(ps, z, flag)), columns=['1', '2', '3'])

    return ret


def get_oultiers(odf1):
    odf1['t'] = odf1['FechaCesta'].astype('datetime64').dt.dayofyear
    # calculate price per unit of sales
    odf1['price_per_unit'] = odf1['Precio'] / odf1['Cantidad']
    x = odf1.groupby(['CodProducto'])[['t', 'price_per_unit']].apply(general_model)
    x1 = x.reset_index().set_index('level_1').drop(['CodProducto'], axis=1)
    odf1[['predicted', 'z', 'outlier']] = x1
    return odf1


import traceback
from sqlalchemy import create_engine, MetaData
import numbers

all_products = None
all_customers = None
all_shops = None

input_enc = "ISO-8859-1"  # 'latin_1 '



def fix_enc(source_df_test, bad_enc=False):
    replace_chars = {"Ã.": "ñ", "Ã³": "ó", "Ã": "í"}
    orig_names = source_df_test.columns.to_list()
    re_encoded_names = [s.encode(input_enc) for s in orig_names]
    if bad_enc:
        #print(orig_names)
        re_encoded_names = orig_names
        # re_encoded_names = [codecs.decode(w.encode(input_enc), "utf-8") for w in orig_names]
        for k, v in replace_chars.items():
            re_encoded_names = [w.replace(k, v) for w in re_encoded_names]

        # codecs.decode(w.encode(input_enc), "utf-8").replace("Ã.", "ñ")

        # re_encoded_names = [s.decode("") for s in re_encoded_names]

        column_rename = dict(zip(source_df_test.columns.to_list()
                                 , re_encoded_names))

        # for messed up column names
        # column_rename['promo_envase']='Promo_Envase'
        # column_rename['promo_folleto']='Promo_Folleto'
        #print(column_rename)

        # print("EXC -- ",e,k)

        # source_df_test= source_df_test.replace(replace_chars)
        # column_names=source_df_test.columns
        # for k,v in replace_chars.items():
        #    column_names=[cn.replace(k,v) for cn in column_names]
        # column_rename = dict(zip(source_df_test.columns.to_list()
        #                         , column_names))
        #print(source_df_test.columns)

        source_df_test = source_df_test.rename(columns=column_rename)
        #print(source_df_test.columns)

    for col in source_df_test.columns:
        #print(f"-----------processing {col}-----------")
        if col not in type_dict_new:
            #print('skipping ', col)
            continue  # for FechaCesta
        elem = source_df_test[col].iloc[0]
        if source_df_test[col].dtype == "object" and (
                (type_dict_new[col] == "str") or (type_dict_new[col] == "category")):
            source_df_test[col] = source_df_test[col].fillna("")
            if isinstance(elem, numbers.Number):
                #print('Numeric, skipping transform: ', elem)
                continue
            #print('Trying transformation: ', elem)
            if bad_enc:
                try:
                    #print("String  ", col, ' t: ', source_df_test[col].dtype)
                    source_df_test[col] = source_df_test[col].str.encode(input_enc)
                    source_df_test[col] = source_df_test[col].str.decode('utf-8')
                    for k, v in replace_chars.items():
                        source_df_test[col] = [w.replace(k, v) for w in source_df_test[col]]
                except Exception as e:
                    print(f"Exception at processing column {col} as a String, declared as {type_dict_new[col]}", e, k)
                    traceback.print_exc()
        #print(" Converting ", col, ' -> ', type_dict_new[col])
        if type_dict_new[col] == 'float64':

            #print('Example element:', elem, type(elem))
            if type(elem) == str:
                #print(col, " -> removing ','")
                source_df_test[col] = source_df_test[col].str.replace(",", ".")
        source_df_test[col] = source_df_test[col].astype(type_dict_new[col])
        #print(col, " -- ", source_df_test[col].dtype)

    return source_df_test

def load_only_purchase_per_year(year,engine):
    def process_and_upload(source_df_test, engine):

        source_df_test = fix_enc(source_df_test, bad_enc)

        # transform
        # if nrows:
        #    print(source_df_test.iloc[0][['idCompra','Promo_Envase','Promo_Folleto','Promo_TPR']])

        source_df_test['Promo_Envase'] = source_df_test['Promo_Envase'].map({'No': False, 'Si': True})
        source_df_test['Promo_Folleto'] = source_df_test['Promo_Folleto'].map({'No': False, 'Si': True})
        source_df_test['Promo_TPR'] = source_df_test['Promo_TPR'].map({'No': False, 'Si': True})

        source_df_test['promo'] = np.where(
            source_df_test['Promo_Folleto'] | source_df_test['Promo_Envase'] | source_df_test['Promo_TPR'], True, False)
        append_to_table('purchases', source_df_test, engine)

    p1=purchase_attributes
    p1.remove("promo")
    chunksize = 10000
    i=1
    if year < 2018:
        bad_enc = True
        input_fn = f'{year} data.csv'
        delimiter = ","
        for chunk in pd.read_csv(data_dir + input_fn,
                                     encoding=input_enc,
                                     delimiter=delimiter,  # decimal=decimal,
                                     on_bad_lines='skip', nrows=nrows, parse_dates=['FechaCesta'], usecols=p1, chunksize=chunksize):
            print(f"Loading chunk {i} ...")
            process_and_upload(chunk, engine)
            print(f"Loading chunk {i} DONE")
            i+=1
    else:
        bad_enc = False
        input_fn = f'Datos_{year}.csv'
        delimiter = ";"
        for chunk in pd.read_csv(data_dir + input_fn,
                                     encoding=input_enc,
                                     delimiter=delimiter,
                                     on_bad_lines='skip', nrows=nrows, parse_dates=['FechaCesta'], usecols=p1, chunksize=chunksize):
            print(f"Loading chunk {i} ...")
            process_and_upload(chunk, engine)
            print(f"Loading chunk {i} DONE")
            i += 1

def load_and_preprocess_data(year, nrows=None):
    # load data

    if year < 2018:
        bad_enc = True
        input_fn = f'{year} data.csv'
        delimiter = ","
        source_df_test = pd.read_csv(data_dir + input_fn,
                                     encoding=input_enc,
                                     delimiter=delimiter,  # decimal=decimal,
                                     on_bad_lines='skip', nrows=nrows, parse_dates=['FechaCesta'])

    else:
        bad_enc = False
        input_fn = f'Datos_{year}.csv'
        delimiter = ";"
        source_df_test = pd.read_csv(data_dir + input_fn,
                                     encoding=input_enc,
                                     delimiter=delimiter,
                                     on_bad_lines='skip', nrows=nrows, parse_dates=['FechaCesta'])

    source_df_test = fix_enc(source_df_test, bad_enc)

    # transform
    # if nrows:
    #    print(source_df_test.iloc[0][['idCompra','Promo_Envase','Promo_Folleto','Promo_TPR']])

    source_df_test['Promo_Envase'] = source_df_test['Promo_Envase'].map({'No': False, 'Si': True})
    source_df_test['Promo_Folleto'] = source_df_test['Promo_Folleto'].map({'No': False, 'Si': True})
    source_df_test['Promo_TPR'] = source_df_test['Promo_TPR'].map({'No': False, 'Si': True})


    source_df_test['promo'] = np.where(
        source_df_test['Promo_Folleto'] | source_df_test['Promo_Envase'] | source_df_test['Promo_TPR'], True, False)

    return source_df_test


# print(type_dict_new)

# %%
import json
from sqlalchemy_utils import drop_database, database_exists, create_database
from sqlalchemy import inspect


def get_connection():
    f = open("dbconnection.json")
    connection_data = json.load(f)
    DB_USER = connection_data["DB_USER"]

    DB_PASS = connection_data["DB_PASS"]
    DB_HOST = connection_data["DB_HOST"]
    DB_PORT = connection_data["DB_PORT"]
    DATABASE = connection_data["DATABASE"]
    CHARSET = "utf-8"
    connect_string = 'postgresql+psycopg2://{}:{}@{}:{}/{}'.format(DB_USER, DB_PASS, DB_HOST, DB_PORT, DATABASE)

    # for local test with trust auth
    if local_test_no_auth:
        DB_HOST = 'localhost'
        DB_PASS='postgres'
        DB_USER='postgres'
        connect_string = 'postgresql+psycopg2://{}:{}@{}:{}/{}'.format(DB_USER, DB_PASS, DB_HOST, DB_PORT, DATABASE)


    #connect_string = 'postgresql+psycopg2://{}:{}@{}:{}/{}?charset={}'.format(DB_USER, DB_PASS, DB_HOST, DB_PORT, DATABASE,
    #                                                                          CHARSET)
    '''
    if database_exists(connect_string):
        drop_database(connect_string)
    create_database(connect_string)
    '''
    if not database_exists(connect_string):
        create_database(connect_string)

    engine = create_engine(connect_string, connect_args={'client_encoding': CHARSET}, pool_pre_ping=True)


    return engine


year_table_name='years'
def table_exists(engine,name):
    ins = inspect(engine)
    ret =ins.dialect.has_table(engine.connect(),name)
    print('Table "{}" exists: {}'.format(name, ret))
    return ret
def get_years_already_loaded(engine):
    lst=[]
    if table_exists(engine,year_table_name):
        df = pd.read_sql_table(year_table_name, engine)
        lst = df['years'].values.tolist()
    return lst

def get_db_table(tn , engine):
    df = None
    if table_exists(engine,tn):
        df = pd.read_sql_table(year_table_name, engine)
    return df
def write_table(tn,df,engine):
    print(f"Uploading (WRITE or REPLACE) data to table {name} ...")
    df.to_sql(tn, engine, if_exists='replace', method='multi', index=False)
    print(f"Uploading (WRITE or REPLACE) data to table {name} is DONE")


def add_years(lst, engine):
    df = pd.DataFrame({'years':lst})
    df.to_sql(year_table_name, engine, if_exists='replace', method='multi', index=False)

def append_to_table(name, df, engine, mode='append'):
    from sqlalchemy import create_engine
    print(f"Uploading (APPEND) data to table {name}")
    df.to_sql(name, engine, if_exists='append', method='multi', index=False)
    print(f"Uploading (APPEND) data to table {name} is DONE")


# %%
def separate_and_save(df, dfs, engine):
    # separating the tables:
    for name, attributes in tables.items():

        data = df[attributes]
        if name == 'purchases':
            if not local_test_no_db:
                append_to_table(name, data, engine)
            del df
            # data = get_oultiers(data)
        else:
            data = data.drop_duplicates()
            dfs[name].append(data)

        # fn = result_dir+f'{year}_{name}.dta'
        # dfs[name].append(fn)

        # to_table(name,data)
        # all_data = pd.concat(df_list, ignore_index=True)

    return dfs


# %%
import fnmatch

nrows = None
if test:
    nrows = 1000

# first separate and save
years = []
for f in os.listdir(data_dir):
    if fnmatch.fnmatch(f, 'Datos_*.csv'):
        print(f)
        try:
            year = int(f.replace("Datos_", "").replace(".csv", ""))
            years.append(year)

        except Exception as e:
            print(f"Bad filename {f}: ´{e}")

    if fnmatch.fnmatch(f, '* data.csv'):
        print(f)
        try:
            year = int(f.replace(" data.csv", ""))
            years.append(year)
        except Exception as e:
            print(f"Bad filename {f}: ´{e}")


dfs = {name: [] for name in tables.keys()}

years_to_retry=[2014,2018,2019,2020,2021,2022]
retry_purchases=True

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if local_test_no_db:
        engine = None
    else:
        engine = get_connection()
    years_already_loaded = get_years_already_loaded(engine)
    if retry_purchases:
        for year in years_to_retry:
            try:
                load_only_purchase_per_year(year,engine)
                years_already_loaded.append(year)
            except Exception as e:
                print(f"Load failed {year}: ´{e}")
        add_years(years_already_loaded,engine)
        print("loading purchases DONE")
    else:
        for year in years:
            try:
                if year in years_already_loaded:
                    print(f'Data for {year} already in db.')
                    continue
                print(f'Loading data for {year} to db..')
                df = load_and_preprocess_data(year, nrows)
                # df=get_oultiers(df)
                dfs = separate_and_save(df, dfs, engine)

                for name, l in dfs.items():
                    if name == 'purchases':
                        continue
                    print(f"Concatanating {name} from {year}...")
                    dfs[name] = [pd.concat(l, ignore_index=True).drop_duplicates()]
                    print(f"Concatanating {name} from {year} is DONE")

                    #if not local_test_no_db:
                    #    write_table(name, data, engine)
                print(f'Processing data for {year} is DONE')
                years_already_loaded.append(year)
            except Exception as e:
                print(f"Load failed {year}: ´{e}")
        for name, l in dfs.items():
            if name == 'purchases':
                continue
            data_in_db= get_db_table(name,engine)
            data = None
            if l:
                if isinstance(data_in_db,pd.DataFrame):
                    l.insert(0, data_in_db)
                    data = pd.concat( l, ignore_index=True).drop_duplicates()
                else:
                    data = l[0]
            if not local_test_no_db and isinstance(data,pd.DataFrame)  :
                write_table(name, data, engine)
        add_years(years_already_loaded,engine)
        print('Finished loading')







# See PyCharm help at https://www.jetbrains.com/help/pycharm/
