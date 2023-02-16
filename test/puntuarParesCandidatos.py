def puntuarConJaccard(df_google, df_amazon, modelo):
    #Jaccard es un coeficiente de similitud que se basa en la proporción de los elementos comunes entre los dos conjuntos de datos (df_amazon y df_google)
    #Se aplica el modelo obtenido en la fase anterior
    productos_similares = modelo.approxSimilarityJoin(df_amazon, df_google, 0.8, distCol="Distancia_Jaccard")
    #Solo se muestran productos con un parecido superor al 30%
    productos_similares = productos_similares.filter(productos_similares['Distancia_Jaccard'] > 0.3)
    #se devuelve el nuevo dataframe con las comparaciones y sus puntuaciones de similitud del 0 al 1.
    return productos_similares