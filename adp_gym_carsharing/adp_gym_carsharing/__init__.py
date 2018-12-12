import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='AdpCarsharing-v0',
    entry_point='adp_gym_carsharing.envs:AdpCarsharingEnv',
)
