# IAtari
### Playing Atari with Deep Reinforcement Learning. I did the project in 2017 for the course Intelligent Systems.

## Introducción
El objetivo es crear un sistema inteligente para que aprenda a jugar de forma autónoma a juegos de la consola Atari 2600. Para ello nos apoyaremos en el paper publicado por Deep Mind en 2013 “Playing Atari with Deep Reinforcement Learning”, el cual usa aprendizaje automático por refuerzo usando la técnica de Q-learning y redes de neuronas para conseguir que el agente aprenda a jugar.

## Herramientas
Como lenguaje de programación usaremos Python y las librerías más importantes que usaremos serán:

* OpenAI Gym: Es una libreria la cual nos ofrece una serie de entornos para entrenar a nuestro sistema inteligente, entre ellos un entorno que emula la consola Atari 2600 y varios de sus juegos.
* Tensorflow: Es una librería open-source sobre machine learning y la más utilizada a la hora de trabajar con redes de neuronas.  
* Keras: Es un API de alto nivel que se apoya en Tensorflow, la cual es más fácil de manejar y entender a la hora de modelar nuestra red de neuronas


## Diseño
El modelo que queremos implementar esta basado en Machine Learning y dentro de esta rama se trata de un algoritmo de aprendizaje por refuerzo llamado Deep Q-Learning.

Nuestro modelo estará formado por:
* Agente: La inteligencia artificial que interactúa con el entorno.
* Entorno: Será un videojuego de la Atari 2600
* Estado: todas las posibles situaciones que se pueden dar en un entorno, al tratarse    de un videojuego el numero de estados sera infinito.
* Acción: dado un estado el agente realiza un acción dentro de su conjunto de acciones que repercutirá en el entorno y generará una recompensa.
* Recompensa: cada vez que se ejecuta una acción el sistema obtiene una recompensa puede ser tanto positiva como negativa.
* Política: conjunto de reglas que el agente aprende de forma autónoma para determinar dado un estado que acción tomar para conseguir la mayor recompensa.

![image](https://user-images.githubusercontent.com/47385326/149672823-193bcbbb-eb17-408e-8bf9-841d9c982c6b.png)

## Q-Learning
El objetivo del algoritmo de Q-Learning es obtener la máxima recompensa final, dado un estado elegir la acción la cual haga que la recompensa final sea la máxima posible. Para ello el algoritmo de Q-Learning se basa en construir una tabla Q[s,a], compuesta por estados(state) y acciones(action). Cada entrada de la tabla relaciona qué recompensa se va a conseguir dado un estado s realizando la acción a. Esta entrada de la tabla se le asigna un valor Q con la función Q(s, a). La cual es:

![image](https://user-images.githubusercontent.com/47385326/149673021-4b796fc3-42fe-48a3-8886-54bfa79a2b37.png)

Como se ve en la función para asignar este valor Q no solo se mira la recompensa directa rt sino que también se tiene en cuenta la recompensa máxima futura maxQ que acarrea tomar la acción a en el estado st+1 . Las otras dos variables son:
* Alpha, α es la media de aprendizaje, toma valores entre 0 y 1, sirve para saber cuánta importancia se le da al nuevo valor aprendido.
* Gamma, γ es un valor entre 0 y 1, sirve para cuantificar la importancia que se da a las futuras recompensas

## Deep Q-Learning
Para problemas como el nuestro con infinitos estados y acciones que se pueden tomar, la implementación del algoritmo de Q-Learning mediante las tablas previamente explicadas es inviable. Por eso Deep Mind desarrollo esta variante del algoritmo Q-Learning llamado Deep Q-Learning.
Este modelo se basa en el uso de redes neuronales convolucionales las cuales toman como input una imagen RGB como array de píxeles con tamaño (210, 160, 3) y cuyo output será el valor  Q de cada una de las acciones.

![image](https://user-images.githubusercontent.com/47385326/149672953-a596c3df-ecbb-4945-ad0d-2ae399be293f.png)

La función que hace la red de neuronas es la de aproximar la función Q, además dota al algoritmo de poder generalizar situaciones basadas en experiencias previas a estados que todavía no ha explorado.



