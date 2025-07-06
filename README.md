# OULADPredict
Proyecto final de Ciencias de Datos I, Predicción de resultados de estudiantes de maestría.


### Configurar Entorno Python

Activar el entorno virtual (macOS/Linux):
```bash
python3 -m venv venv
```
```bash 
source venv/bin/activate
```

Windows:
```bash
python -m venv venv
```
```bash
venv\Scripts\activate
```


Instalar dependencias:
```bash
python -m pip install -r requirements.txt
```

### Descargar datasets

```bash
python data/downloadDatasets.py data/raw/
```
Los CSV originales de OULAD se deben guardar en  ```data/raw```.

## Para visualizar el Notebook 

Vamos a la carpeta EDA y ejecutamos los siguientes comandos:
```Nota: si abres una nueva terminal, no olvides activar tu entorno (venv).```

```bash
source venv/bin/activate
```
```bash
python -m pip install notebook jupyterlab ipywidgets
```
Agregar Kernel ENV
```bash
python -m ipykernel install --user --name=venv_oulad --display-name "Python (venv OULAD)"
```
Abrir notebook:
```bash
jupyter notebook EDA/EDA_NOTEBOOK.ipynb
```
Esto debe abrir en su navegador la siguiente ruta: http://localhost:8889/notebooks/EDA_NOTEBOOK.ipynb
En la barra de menu, debes ir donde dice [Kernel] y seleccionar tu  "Python (venv OULAD)".
