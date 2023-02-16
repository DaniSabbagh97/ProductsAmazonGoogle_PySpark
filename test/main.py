from pyspark.sql import SparkSession
from normalizar import normalizacion_Fuentes
from paresCandidatos import generarParesCandidatos
from puntuarParesCandidatos import puntuarConJaccard

def main():

    ''' #FIXME Maneras de iniciar la sesión de SPARK dependiendo de tu ordenador.

        # conf = SparkConf().set("spark.driver.memory", "20g")
        # spark = SparkSession.builder.config(conf=conf).getOrCreate()

        # spark.conf.set("spark.executor.memory", "16g")
        # spark.conf.set("spark.executor.cores", "8")

    '''
    #Inicio de la sesión de Spark, señalando que use todos los cores de mi CPU y el nombre del proyecto
    spark = SparkSession.builder.appName("Practica3_Daniel_Sabbagh").master("local[*]").getOrCreate()
    #warn
    spark.sparkContext.setLogLevel('warn')
    #Lectura de los csv
    df_amazon = spark.read.csv("../data/Amazon.csv", header=True, inferSchema=True)
    df_google = spark.read.csv("../data/GoogleProducts.csv", header=True, inferSchema=True)
    #Se aplica la fase de normalización
    df_normalizado = normalizacion_Fuentes(df_amazon, df_google)
    #Se aplica la fase de Pares de Candidatos
    df_google_final, df_amazon_final, modelo= generarParesCandidatos(df_normalizado, "description_tokenized")
    #Se aplica la fase de puntuar los pares de candidatos
    resultado_final = puntuarConJaccard(df_google_final, df_amazon_final, modelo)
    #Se muestra el resultado final
    resultado_final.show()
    #Se finaliza la sesión de Spark
    spark.stop()

if __name__ == '__main__':
    main()


