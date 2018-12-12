import keras.backend as K
from keras import optimizers


class Actor(object):

    def __init__(self, model, critic_model, lr=1e-4):
        self.model = model
        self.opt = optimizers.Adam(lr)
        self.model.compile(self.opt, 'mse')

        state_input = self.model.inputs[0]
        critic_out = critic_model([state_input, self.model(state_input)])

#        updates = self.opt.get_updates(
#            params=self.model.trainable_weights, constraints=self.model.constraints,
#            loss=-K.mean(critic_out))
        updates = self.opt.get_updates(self.model.trainable_weights, [],-K.mean(critic_out))        
        updates += self.model.updates
        # learning_phase added in case of batchnorm layers.
        self.train_step = K.function(
            inputs=[state_input, K.learning_phase()], outputs=[], updates=updates)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)
        
    def save_model_weights(self, suffix):
        # Helper function to save your model / weights. 
        self.model.save_weights(suffix)    

    def step(self, states):
        learning_phase = 1  # Signalize training phase. Handled implicitly for prediction.
        self.train_step([states, learning_phase])

    def __call__(self, state):
        return self.model.predict_on_batch([state])
