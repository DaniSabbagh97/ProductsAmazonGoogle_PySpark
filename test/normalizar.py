from pyspark.sql.functions import trim, lower, col
from nltk.tokenize import word_tokenize
from pyspark.ml.feature import Tokenizer

def normalizacion_Fuentes(df_amazon, df_google):
    #Se busca cambiar el nombre a la columna diferente del CSV(ya dataframe) para poder unirlos
    df_amazon = df_amazon.withColumnRenamed("title", "name")
    #Se unen los dos dataframes de los CSV
    df_amazon_google = df_amazon.union(df_google)
    #Se normalizan los datos, o sea se pasan todos los datos a minúsculas y se les quita los espacios de los lados.
    #En este caso a 3 columnas porque se hicieron diferentes pruebas en el proyecto, con el que más encajó fue con descripción, es el que se usa
    df_amazon_google = df_amazon_google.withColumn("description", lower(trim(col("description"))))
    df_amazon_google = df_amazon_google.withColumn("manufacturer", lower(trim(col("manufacturer"))))
    df_amazon_google = df_amazon_google.withColumn("price", lower(trim(col("price"))))
    #Ahora se descartan duplicados y filas con datos nulos.
    df_amazon_google = df_amazon_google.dropna()
    df_amazon_google = df_amazon_google.dropDuplicates()
    #Comienza la fase de tokenización para poder tener listas de palabras en vez de frases.
    tokenizer = Tokenizer(inputCol="description", outputCol="description_tokenized")
    df_amazon_google = tokenizer.transform(df_amazon_google)
    #Una vez normalizado el dataframe con los dos csv, se envía para la siguiente fase.
    return df_amazon_google


#Se aplica el word_tokenize de NLTK
def tokenizar(columna):
   return word_tokenize(columna)


