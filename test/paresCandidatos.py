from pyspark.ml.feature import HashingTF, IDF, MinHashLSH
import pyspark.sql.functions as F

def generarParesCandidatos(dataframe, nombreColumna):
    #HasgingTF(Hashing Term Frecuency) convierte nuestra lista de palabras(descripcion) en un vector de características utilizando una función Hash.
    #La función Hash, asigna un valor númerico a cada palabra.
    #HashingTF cuenta el número de veces que aparece cada palabra, finalmente las agrupa en un vector de características.
    troceo = HashingTF(inputCol=nombreColumna, outputCol="tf")
    tfidf = IDF(inputCol="tf", outputCol="caracteristicas").fit(troceo.transform(dataframe)).transform(troceo.transform(dataframe))
    #MinHashLSH reduce los datos lo que permite realizar una búsqueda de similitud eficiente.
    locality_sensitive_hasing = MinHashLSH(inputCol="caracteristicas", outputCol="trozos", numHashTables=5)
    modelo = locality_sensitive_hasing.fit(tfidf)
    tfidf_google = tfidf.where(F.col("id").startswith("http"))
    tfidf_amazon = tfidf.where(~F.col("id").startswith("http"))
    #Se devuelven los datos para aplicar la función de similitud y obtener la puntuación por cada par de candidatos.
    return tfidf_google, tfidf_amazon, modelo