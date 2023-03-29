# Chatbot


Repositorio para proyecto de implementación de práctica de un chatbot en python usando la librería pytorch

# Instalación
## Crear un nuevo ambiente virtual (Conda o venv)

```
mkdir myproject
$ cd myproject
$ python3 -m venv venv
```

## Activate it
Mac / Linux:

```. venv/bin/activate```

Windows:

```venv\Scripts\activate```

## Instalar dependencias y PyTorch
Revisa la [documentación oficial de PyTorch](https://pytorch.org/) para su instalación.

Ademas necesitas nltk:

```pip install nltk```

Si recibís un error en la primera ejecutción, tenes que instalar tambien ```nltk.tokenize.punkt```: Corre esto una vez en tu terminal:

```
$ python
>>> import nltk
>>> nltk.download('punkt')
```
## Uso
### Ejecución

```python train.py```
Esto genera como salida el archivo data.pth. 

Finalmente corre
```python chat.py```

# Customize
El archivo intents.json contiene los datos de entrenamiento. Podes adaptarlo a tu caso de uso. 

Solo hay que definir una nueva etiqueta, patrones de entrada y posibles respuestas.

Al modificarlo tenés que re entrenar tu chat ejecutanto devuelta el entrenamiento

```
{
  "intents": [
    {
      "tag": "Saludos",
      "patterns": [
        "Hola",
        "Buenas",
        "¿Que tal?",
        "Buen día",
      ],
      "responses": [
        "Hola :-)",
        "Hola, gracias por contactarte",
        "Hola, ¿qué puedo hacer por vos?",
        "Hola, ¿Cómo te puedo ayudar?"
      ]
    },
    ...
  ]
}
```
