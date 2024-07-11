#!/usr/bin/env python
# coding: utf-8

# ## Para regresión logistica

# In[3]:


import os
import json
import pandas as pd
directorio_actual_1 = os.getcwd()
# Ruta de la carpeta que deseas explorar
ruta_carpeta = directorio_actual_1

# Recorrer la estructura de directorios y archivos de manera recursiva
for directorio_actual, directorios, archivos in os.walk(ruta_carpeta):
    # Verificar si hay archivos con extensión .json en este directorio
    if any(archivo.endswith('.json') for archivo in archivos):
        # Mostrar el directorio actual
        print("\n\n\n\n")
        print("Directorio con archivos .json:", directorio_actual)
        separacion_de_directorio = directorio_actual.split("\\")
        print(f"Directorio_separado: {separacion_de_directorio}")
        # Recorrer los archivos en este directorio
        for archivo in archivos:
            # Verificar si el archivo tiene extensión .json
            if archivo.endswith('.json'):
                # Mostrar el nombre del archivo .json
                print("Nombre del archivo .json:", archivo)
                directorio_json = directorio_actual + "\\" + archivo
                with open(directorio_json, "r") as archivo:
                    datos = json.load(archivo)
                    datos_data = pd.DataFrame.from_dict(datos, orient = 'index')
                    print(datos_data)
                    directorio_de_guardado_pd = directorio_actual + "\\" + separacion_de_directorio[-1] + ".xlsx"
                    directorio_de_guardado_pd_1 = directorio_actual + "\\" + separacion_de_directorio[-1] + "_dataframe" + ".csv"
                    print(f"Dataframe_almacenado en: {directorio_de_guardado_pd}")
                    datos_data.to_excel(directorio_de_guardado_pd, index=True)  # Guardar el DataFrame sin incluir el índice
                    datos_data.to_csv(directorio_de_guardado_pd_1, index=True)  # Guardar el DataFrame sin incluir el índice


# In[79]:


# Para la carpeta neural network 
directorio_actual_1 = os.getcwd()
# Ruta de la carpeta que deseas explorar
ruta_carpeta = directorio_actual_1


nuevo_dataframe = pd
archivos_diccionario = []

for directorio_actual, directorios, archivos in os.walk(ruta_carpeta):
    #print(f"directorio_actual: {directorio_actual}")
    #print(f"directorios: {directorios}")
    for k in directorios:
        if k == "Neural_network_classification":
            ruta_carpeta_neural_network = ruta_carpeta + "\\" + k 
            #print(f"ruta_carpeta: {ruta_carpeta_neural_network}")
            for directorio_actual_neural, directorios_neural, archivos_neural in os.walk(ruta_carpeta_neural_network):
                #if directorios_neural != []:
                #    for m in directorios_neural:
                #        datos = m.split("_")
                #        epocas.append(datos[1])
                #        factor_de_aprendizaje.append(datos[2])
                #        test_size.append(datos[3])
                #    data = {'epocas':epocas,"factor_de_aprendizaje":factor_de_aprendizaje,"test_size":test_size}
                #    df = pd.DataFrame(data)
                    #print(df)
                #print(f"directorio_actual: {directorio_actual_neural}")
                #print(f"directorios: {directorios_neural}")
                #print(f"archivos: {archivos_neural}")
                j = 0
                for archivos in archivos_neural:
                    if archivos.endswith('.xlsx'):
                    # Mostrar el nombre del archivo .json
                        #print("**************************************************")
                        #print(f"directorio_actual: {directorio_actual_neural}")
                        #print(f"archivos: {archivos}")
                        separar = directorio_actual_neural.split("\\")
                        metricas = separar[-1]
                        datos = metricas.split("_")
                        epocas = (datos[1])
                        factor_de_aprendizaje = (datos[2])
                        test_size = (datos[3])

                        diccionario_terminos = {}
                        #print(epocas)
                        #print(factor_de_aprendizaje)
                        #print(test_size)
                        
                        diccionario_terminos['epocas'] = epocas
                        diccionario_terminos['factor_de_aprendizaje'] = factor_de_aprendizaje
                        diccionario_terminos['test_size'] = test_size
                        
                        #print(metricas)
                        #print("Nombre del archivo .xlsx:", archivos)
                        directorio_final = directorio_actual_neural + "\\" + archivos
                        #print(f"Directorio de archivo final: {directorio_final}")
                        model = pd.read_excel(directorio_final)
                        #print(model)

                        precision_row = model[model['Unnamed: 0'] == 'accuracy']
                        precision_value = precision_row.iloc[0, 1]
                        diccionario_terminos['accuracy'] = precision_value
                        
                        precision_row = model[model['Unnamed: 0'] == 'precision']
                        precision_value = precision_row.iloc[0, 1]
                        diccionario_terminos['precision'] = precision_value

                        precision_row = model[model['Unnamed: 0'] == 'recall']
                        precision_value = precision_row.iloc[0, 1]
                        diccionario_terminos['recall'] = precision_value

                        precision_row = model[model['Unnamed: 0'] == 'f1']
                        precision_value = precision_row.iloc[0, 1]
                        diccionario_terminos['f1'] = precision_value
                        archivos_diccionario.append(diccionario_terminos)
            
                        
                        #print(diccionario_terminos)
    #print(f"archivos: {archivos}")
    #print("*******************************************************")
    nuevo_data = pd.DataFrame(archivos_diccionario)
    print(nuevo_data)
nueva_ruta = ruta_carpeta_neural_network + "\\" + "resultados_neural_network.xlsx"
nuevo_data.to_excel(nueva_ruta, index=False)
print(f"DataFrame guardado exitosamente en {nueva_ruta}")


# In[85]:


# Para la carpeta LSTM
directorio_actual_1 = os.getcwd()
# Ruta de la carpeta que deseas explorar
ruta_carpeta = directorio_actual_1


nuevo_dataframe = pd
archivos_diccionario = []

for directorio_actual, directorios, archivos in os.walk(ruta_carpeta):
    #print(f"directorio_actual: {directorio_actual}")
    #print(f"directorios: {directorios}")
    for k in directorios:
        if k == "LSTM_Neural_Network":
            ruta_carpeta_neural_network = ruta_carpeta + "\\" + k 
            #print(f"ruta_carpeta: {ruta_carpeta_neural_network}")
            for directorio_actual_neural, directorios_neural, archivos_neural in os.walk(ruta_carpeta_neural_network):
                #if directorios_neural != []:
                #    for m in directorios_neural:
                #        datos = m.split("_")
                #        epocas.append(datos[1])
                #        factor_de_aprendizaje.append(datos[2])
                #        test_size.append(datos[3])
                #    data = {'epocas':epocas,"factor_de_aprendizaje":factor_de_aprendizaje,"test_size":test_size}
                #    df = pd.DataFrame(data)
                    #print(df)
                #print(f"directorio_actual: {directorio_actual_neural}")
                #print(f"directorios: {directorios_neural}")
                #print(f"archivos: {archivos_neural}")
                j = 0
                for archivos in archivos_neural:
                    if archivos.endswith('.xlsx'):
                    # Mostrar el nombre del archivo .json
                        #print("**************************************************")
                        #print(f"directorio_actual: {directorio_actual_neural}")
                        #print(f"archivos: {archivos}")
                        separar = directorio_actual_neural.split("\\")
                        metricas = separar[-1]
                        datos = metricas.split("_")
                        epocas = (datos[1])
                        factor_de_aprendizaje = (datos[2])
                        test_size = (datos[3])

                        diccionario_terminos = {}
                        #print(epocas)
                        #print(factor_de_aprendizaje)
                        #print(test_size)
                        
                        diccionario_terminos['epocas'] = epocas
                        diccionario_terminos['factor_de_aprendizaje'] = factor_de_aprendizaje
                        diccionario_terminos['test_size'] = test_size
                        
                        #print(metricas)
                        #print("Nombre del archivo .xlsx:", archivos)
                        directorio_final = directorio_actual_neural + "\\" + archivos
                        #print(f"Directorio de archivo final: {directorio_final}")
                        model = pd.read_excel(directorio_final)
                        #print(model)

                        precision_row = model[model['Unnamed: 0'] == 'accuracy']
                        precision_value = precision_row.iloc[0, 1]
                        diccionario_terminos['accuracy'] = precision_value
                        
                        precision_row = model[model['Unnamed: 0'] == 'precission']
                        precision_value = precision_row.iloc[0, 1]
                        diccionario_terminos['precision'] = precision_value

                        precision_row = model[model['Unnamed: 0'] == 'recall']
                        precision_value = precision_row.iloc[0, 1]
                        diccionario_terminos['recall'] = precision_value

                        precision_row = model[model['Unnamed: 0'] == 'f1']
                        precision_value = precision_row.iloc[0, 1]
                        diccionario_terminos['f1'] = precision_value
                        archivos_diccionario.append(diccionario_terminos)
            
                        
                        #print(diccionario_terminos)
    #print(f"archivos: {archivos}")
    #print("*******************************************************")
    nuevo_data = pd.DataFrame(archivos_diccionario)
    print(nuevo_data)
nueva_ruta = ruta_carpeta_neural_network + "\\" + "LSTM_neural_network.xlsx"
nuevo_data.to_excel(nueva_ruta, index=False)
print(f"DataFrame guardado exitosamente en {nueva_ruta}")

