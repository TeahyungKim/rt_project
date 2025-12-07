import tensorflow as tf
import numpy as np
from parameters import LEARNING_RATE, GAMMA, CLIP_RATIO, ENTROPY_COEF, LMBDA, K_EPOCH

class PPO(tf.keras.Model):
    def __init__(self, state_dim=9, action_dim=1):
        super(PPO, self).__init__()
        self.data = []
        
        # Common layers or separate? Reference uses shared fc1? 
        # Reference: fc1 -> fc_pi, fc1 -> fc_v. Shared body.
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        
        # Actor (Policy)
        # Continuous action space: Output Mean (mu) and Std (sigma)
        self.fc_mu = tf.keras.layers.Dense(action_dim, activation='sigmoid') # Output [0, 1]
        self.fc_sigma = tf.keras.layers.Dense(action_dim, activation='softplus') # Output > 0
        
        # Critic (Value)
        self.fc_v = tf.keras.layers.Dense(1)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    def call(self, x):
        # Required for tf.keras.Model, but we use pi and v explicitly
        x = self.fc1(x)
        return self.fc_mu(x), self.fc_v(x)

    def pi(self, x):
        """
        Returns mean and standard deviation for the policy distribution.
        """
        x = self.fc1(x)
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma
    
    def v(self, x):
        """
        Returns the value estimation.
        """
        x = self.fc1(x)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, log_prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, log_prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            log_prob_a_lst.append([log_prob_a])
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
            
        s = tf.convert_to_tensor(np.array(s_lst), dtype=tf.float32)
        a = tf.convert_to_tensor(np.array(a_lst), dtype=tf.float32)
        r = tf.convert_to_tensor(np.array(r_lst), dtype=tf.float32)
        s_prime = tf.convert_to_tensor(np.array(s_prime_lst), dtype=tf.float32)
        done_mask = tf.convert_to_tensor(np.array(done_lst), dtype=tf.float32)
        log_prob_a = tf.convert_to_tensor(np.array(log_prob_a_lst), dtype=tf.float32)
        
        self.data = []
        return s, a, r, s_prime, done_mask, log_prob_a


    def train_net(self):
        s, a, r, s_prime, done_mask, old_log_prob_a = self.make_batch()

        for i in range(K_EPOCH):
            with tf.GradientTape() as tape:
                v_prime = self.v(s_prime)
                td_target = r + GAMMA * v_prime * done_mask
                v_s = self.v(s)
                delta = td_target - v_s
                delta = delta.numpy()

                advantage_lst = []
                advantage = 0.0
                for delta_t in delta[::-1]:
                    advantage = GAMMA * LMBDA * advantage + delta_t[0]
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantage = tf.convert_to_tensor(advantage_lst, dtype=tf.float32)

                mu, sigma = self.pi(s)
                sigma = sigma + 1e-5
                
                # Manual Log Probability Calculation
                # log_prob = -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5 * log(2*pi)
                log_pi_a = -0.5 * tf.square((a - mu) / sigma) - tf.math.log(sigma) - 0.5 * np.log(2 * np.pi)
                
                # Ratio = exp(new_log_prob - old_log_prob)
                ratio = tf.exp(log_pi_a - old_log_prob_a)

                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * advantage
                
                # PPO Loss + Value Loss + Entropy Bonus (Optional)
                loss_actor = -tf.reduce_mean(tf.minimum(surr1, surr2))
                loss_critic = tf.reduce_mean(tf.keras.losses.Huber()(v_s, td_target))
                
                # Manual Entropy Calculation
                # entropy = 0.5 * (1 + log(2*pi)) + log(sigma)
                entropy = tf.reduce_mean(0.5 * (1.0 + np.log(2 * np.pi)) + tf.math.log(sigma))
                
                loss = loss_actor + 0.5 * loss_critic - ENTROPY_COEF * entropy

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))