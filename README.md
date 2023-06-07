# Classificação de imagens com deep learning e TensorFlow

## Instale o docker

    docker run hello-world

## Baixar a imagem do TensorFlow

    docker pull tensorflow/tensorflow

## Iniciar o container baseado na imagem do TensorFlow

    docker run -it --volume ${PWD}:/tf_files --workdir /tf_files --publish 6006:6006 --rm tensorflow/tensorflow bash

## Iniciar o treinamento

    python -m retrain \
        --bottleneck_dir=bottlenecks \
        --how_many_training_steps=500 \
        --model_dir=models/ \
        --summaries_dir=training_summaries/"${ARCHITECTURE}" \
        --output_graph=retrained_graph.pb \
        --output_labels=retrained_labels.txt \
        --architecture="${ARCHITECTURE}" \
        --image_dir=cats

## Consultar o modelo gerado

    python label_image.py gato.jpg

## Próximos passos

Podemos pegar o modelo que treinamos e criar uma aplicação para disponibilizar uma API que recebe a imagem de um gato e retorna sua raça fazendo consulta a este modelo. Ou podemos também criar uma aplicativo para celular, onde tiramos a foto de um gato e consultamos o modelo para saber a sua raça.

Estes são alguns exemplos de como utilizar isso no mundo real, e lembre-se de que podemos utilizar este modelo também para outros tipos de imagens e classificações.

## Fonte
https://imasters.com.br/back-end/classificacao-de-imagens-com-deep-learning-e-tensorflow

## ERRO

O código esta apresntando o seguinte erro na hora de fazer o treinamento:

    TypeError: <tf.Tensor 'pool_3/_reshape:0' shape=(1, 2048) dtype=float32> is out of scope and cannot be used here. Use return values, explicit Python locals or TensorFlow collections to access it.
    Please see https://www.tensorflow.org/guide/function#all_outputs_of_a_tffunction_must_be_return_values for more information.

    <tf.Tensor 'pool_3/_reshape:0' shape=(1, 2048) dtype=float32> was defined here:
    <unknown>

    The tensor <tf.Tensor 'pool_3/_reshape:0' shape=(1, 2048) dtype=float32> cannot be accessed from here, because it was defined in <tensorflow.python.framework.ops.Graph object at 0x7fe137c3e9a0>, which is out of scope.