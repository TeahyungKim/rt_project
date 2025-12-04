import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from utils import LEARNING_RATE, GAMMA, CLIP_RATIO

# Hyperparameters
learning_rate = LEARNING_RATE
gamma         = GAMMA
lmbda         = 0.95
eps_clip      = CLIP_RATIO
K_epoch       = 3

class PPO(tf.keras.Model):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = tf.keras.layers.Dense(256, activation='relu')
        self.fc_mu = tf.keras.layers.Dense(1, activation='sigmoid')
        self.fc_sigma = tf.keras.layers.Dense(1, activation='softplus')
        self.fc_v  = tf.keras.layers.Dense(1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def pi(self, x):
        x = self.fc1(x)
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma
    
    def v(self, x):
        x = self.fc1(x)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s = tf.convert_to_tensor(np.array(s_lst), dtype=tf.float32)
        a = tf.convert_to_tensor(np.array(a_lst), dtype=tf.float32)
        r = tf.convert_to_tensor(np.array(r_lst), dtype=tf.float32)
        s_prime = tf.convert_to_tensor(np.array(s_prime_lst), dtype=tf.float32)
        done_mask = tf.convert_to_tensor(np.array(done_lst), dtype=tf.float32)
        prob_a = tf.convert_to_tensor(np.array(prob_a_lst), dtype=tf.float32)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            with tf.GradientTape() as tape:
                v_prime = self.v(s_prime)
                td_target = r + gamma * v_prime * done_mask
                v_s = self.v(s)
                delta = td_target - v_s
                delta = delta.numpy()

                advantage_lst = []
                advantage = 0.0
                for delta_t in delta[::-1]:
                    advantage = gamma * lmbda * advantage + delta_t[0]
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantage = tf.convert_to_tensor(advantage_lst, dtype=tf.float32)

                mu, sigma = self.pi(s)
                dist = tfp.distributions.Normal(loc=mu, scale=sigma + 1e-5)
                
                log_pi_a = dist.log_prob(a)
                ratio = tf.exp(log_pi_a - prob_a)  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1-eps_clip, 1+eps_clip) * advantage
                
                loss = -tf.reduce_min(tf.stack([surr1, surr2], axis=0), axis=0) + tf.keras.losses.Huber()(v_s, td_target)
                loss = tf.reduce_mean(loss)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
