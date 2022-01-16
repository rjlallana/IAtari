import gym
import numpy as np
import matplotlib.pyplot as plt

import time
import random
import pickle
from collections import deque

import tensorflow as tf
import keras

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, Input, Layer
from keras.optimizers import Adam

# from rl.util import huber_loss

import keras.backend as K

TOTAL_STEPS = 1750000
TRAIN_START = 10000
TRAIN_FRECUENCY = 4
TARGET_MODEL_UPDATE= 10000
LOGGER_FRECUENCY = 1000

'''OPCIONES'''
RENDER = False
LOAD = True
SAVE = True

def max_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

def huber_loss(y_true, y_pred, clip_value):
    # Huber loss, https://en.wikipedia.org/wiki/Huber_loss, referencia: https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)

class Deep_Q_Network():
    """
    Hyperparamentros de la red neuronal
    """
    def __init__(self, num_actions, start_training=50000, target_update=10000):
        self.num_actions = num_actions
        self.input_shape = (105, 80, 4)  # (ancho, largo, canales)
        self.gamma = 0.99 # sirve para cuantificar la importancia que se da a las futuras recompensas
        self.epsilon = 1.0 # ratio de exploracion, al principio se toman decisiones aleatorias (exploracion), luego se va reduciendo gradualmente hasta que se queda con una decision determinista
        self.epsilon_min = 0.1 # 1% de probabilidad de tomar una decision aleatoria una vez se haya reducido el ratio de exploracion al minimo
        self.exploration = 500000 # A la mitad del entrenamiento se reduce el ratio de exploracion al minimo
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.exploration  # ratio de decrecimiento de epsilon
        self.batch_size = 32 
        self.start_training = start_training
        self.target_update = target_update
        self.main_model   = None
        self.target_model = None

        self.delta_clip = 1.0 

    def build_network(self):
        '''
        Arquitectura basada en el paper: https://deepmind.com/research/publications/2019/playing-atari-deep-reinforcement-learning
        '''
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=(105, 80, 4)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.num_actions))
        model.add(Activation('linear'))
        model.compile(optimizer='sgd', loss='mse')
        return model

    def compile(self, optimizer, metrics=[]):
        metrics += [max_q] 

        self.target_model.compile(optimizer='sgd', loss='mse')
        self.main_model.compile(optimizer='sgd', loss='mse')

        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)

        y_pred = self.main_model.output
        y_true = Input(name='y_true', shape=(self.num_actions,))
        mask = Input(name='mask', shape=(self.num_actions,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask])
        ins = [self.main_model.input] if type(self.main_model.input) is not list else self.main_model.input
        trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  
            lambda y_true, y_pred: K.zeros_like(y_pred),
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

    def update_target_model(self):
        self.target_model.set_weights(self.main_model.get_weights())

    def select_action(self, state):
        # Si el ratio de exploracion todavia no ha llegado al minimo, decrece
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        # Si el ratio de exploracion todavia no ha llegado al minimo, decrece
        if self.epsilon < random.random():
            return random.randrange(self.num_actions)
        else:
            # Pasar estado actual a escala de grises
            state = np.float32(state / 255.0)
            # Predecir accion Q a partir de la estado actual
            q_value = self.main_model.predict_on_batch(state)
            # Tomar la accion con valor Q mas alto
            action = np.argmax(q_value)
            return action

    def train(self, memory):
        '''
        Un minibatch de experiencias seleccionadas aleatoriamente, para entrenar la red neuronal
        '''
        minibatch = random.sample(memory, self.batch_size)
        state_batch = []
        targets = np.zeros((self.batch_size, self.num_actions)) # (32, 4)
        dummy_targets = np.zeros((self.batch_size,))            # (32,)
        masks = np.zeros((self.batch_size, self.num_actions))   # (32, 4)
        i = 0
        for state, action, reward, new_frame, done in minibatch:
            next_state = np.append(new_frame, state[:,:,:,:3], axis=3)
            next_state = np.float32(next_state/255.)

            state = np.float32(state/255.0)
            state_batch.append(state)

            next_q_value = self.target_model.predict_on_batch(next_state)

            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * np.amax(next_q_value)

            dummy_targets[i] = targets[i][action]
            masks[i][action] = 1.
            i += 1


        max_q = np.amax(dummy_targets)

        state_batch = np.array(state_batch).astype('float32')
        # state_batch = state_batch.squeeze()
        state_batch = np.reshape(state_batch, (32, 105, 80, 4))

        # print(state_batch.shape)
        targets = np.array(targets).astype('float32')
        # print(targets.shape)
        masks = np.array(masks).astype('float32')
        # print(masks.shape)

        metrics = self.trainable_model.train_on_batch(
            [state_batch, targets, masks], [dummy_targets, targets])
        return max_q, metrics

def preprocess_input(frame):
    '''
    Preprocesa un frame para ser usado en la red neuronal,
        1. Convierte a escala de grises
        2. Divide el tamaño del frame por la mitad 
    '''
    frame = np.mean(frame, axis=2).astype(np.uint8)
    frame = frame[::2,::2]

    return frame # -> Imagen (105,80)

def plot_training(y, string, time_step):
    plt.plot(y)
    plt.xlabel('episode')
    plt.ylabel(string)
    plt.savefig('/content/gdrive/My Drive/training/graficas/{}-{}'.format(string, time_step))

if __name__ == '__main__':
    env_name = 'BreakoutDeterministic-v4'
    env = gym.make(env_name)
    actions = env.action_space.n
    dqn = Deep_Q_Network(num_actions=actions)
    dqn.main_model   = dqn.build_network()  # La red de neuronas principal es sobre la que se toman las decisiones
    dqn.target_model = dqn.build_network()  # La red de neuronas objetivo es sobre la que se entrena
    dqn.compile(Adam(lr=.00025), metrics=['mae']) # Compilamos ambas redes
    '''Hyperparametros'''
    time_step = 0
    episode_step = 0 # episodio, es 
    episode = 0
    score = 0.
    start_time = time.time()
    exit = False
    memory = deque(maxlen=150000) # ~ Ocupa mas o menos 11GB de RAM
    score_per_episode = []
    max_q = []
    metrics_list = []
    '''Load'''
    if LOAD:
        pickle_off = open('/content/gdrive/My Drive/training/data.txt', 'rb')
        data = pickle.load(pickle_off)
        # print(np.shape(data))
        time_step, episode, epsilon, score_per_episode,\
        max_q, metrics_list = data
        dqn.main_model.load_weights('/content/gdrive/My Drive/training/modelos/weights-BreakoutDeterministic-v4-600000.h5'.format(time_steps))
        dqn.target_model.load_weights('/content/gdrive/My Drive/training/modelos/weights-BreakoutDeterministic-v4-600000.h5'.format(time_steps))
        pickle_off.close()
    time_step = 6
    while time_step < TOTAL_STEPS:
        frame = env.reset()
        frame = preprocess_input(frame)
        # State corresponde a las 4 ultimas imagenes del entorno
        state = np.stack((frame, frame, frame, frame))
        # State -> (105,80,4), 
        state = np.reshape(state, (1, 105, 80, 4))
        print('Iteration:{}\tEpisode:{}\tmax_q:{}\tscore:{}\tepsilon:{}\ttime:{}'
            .format(time_step, episode, np.mean(max_q), score, dqn.epsilon, time.time()-start_time))
        episode_step = 0
        score = 0.
        exit = False
        while not exit:
            if RENDER:
                env.render()
                time.time(0.060)
            # Decidir la mejor accion a partir de la estado actual
            action = dqn.select_action(state)
            # Tomar la accion
            new_frame, reward, done, info = env.step(action)
            reward = np.clip(reward, -1., 1.)

            new_frame = preprocess_input(new_frame)
            new_frame = np.reshape(new_frame, (1, 105, 80, 1))
            # memory, es una lista de experiencias, antes de poder entrenar al modelo se necesita al menos TRAIN_START experiencias
            memory.append((state, action, reward, new_frame, done))
            # Entrenar la red cada 4 frames
            if time_step > TRAIN_START and time_step % TRAIN_FRECUENCY == 0:
                max_q_value, metrics = dqn.train(memory)
                max_q.append(max_q_value)
                metrics_list.append(metrics)
            # Actualizar el estado
            if time_step % TARGET_MODEL_UPDATE == 0:
                dqn.update_target_model()
            # Guardar el modelo cada 150000 iteraciones
            if time_step % 150000  == 0 and SAVE: 
                dqn.main_model.save_weights('/content/gdrive/My Drive/training/modelos/{}-{}.h5'.format(env_name, time_step))
                dqn.main_model.save('/content/gdrive/My Drive/training/modelos/weights-{}-{}.h5'.format(env_name, time_step))
                pickle_on = open('/content/gdrive/My Drive/training/data.txt', 'wb')
                pickle.dump([time_step, episode, dqn.epsilon, score_per_episode,
                            max_q, metrics_list], pickle_on)
                pickle_on.close()

                plot_training(score_per_episode,'score_per_episode', time_step)
                plot_training(max_q,'max_q', time_step)

            state = np.append(new_frame, state[:,:,:,:3], axis=3)
            score += reward

            time_step    += 1
            episode_step += 1

            if done:
                score_per_episode.append(score)
                episode += 1
                exit = True
                break
