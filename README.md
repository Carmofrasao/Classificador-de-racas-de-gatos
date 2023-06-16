# Classificação de imagens com deep learning e TensorFlow

## Baixe e Instale dependencias 
    # Atualiza o comando pip
    pip install --upgrade pip

    # instala a partir do comando pip o tensorflow
    pip install tensorflow

    #Intale matplotlib
    sudo apt-get install python3-matplotlib

    ou

    pip3 install -U matplotlib

## Iniciar o re-treinamento do modelo

    python3 retrain.py

    Após a execução, o código irá salvar o modelo gerado no arquivo dogs-vs-cats.h5

## Consultar o modelo gerado

    python3 label_image.py

## Próximos passos

Podemos pegar o modelo que treinamos e criar uma aplicação para disponibilizar uma API que recebe a imagem de um gato e retorna sua raça fazendo consulta a este modelo. Ou podemos também criar uma aplicativo para celular, onde tiramos a foto de um gato e consultamos o modelo para saber a sua raça.

Estes são alguns exemplos de como utilizar isso no mundo real, e lembre-se de que podemos utilizar este modelo também para outros tipos de imagens e classificações.

(Precisamos mudar o algoritmo pra ter esses objetivos)

## Fonte
https://imasters.com.br/back-end/classificacao-de-imagens-com-deep-learning-e-tensorflow
https://www.tensorflow.org/tutorials/images/transfer_learning?hl=en
https://www.tensorflow.org/tutorials/keras/save_and_load?hl=pt-br#save_the_entire_model