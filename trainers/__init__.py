from trainers.pgtrainer import PGTrainer
from trainers.ppotrainer import PPOTrainer

TRAINERS = {
    "pg": PGTrainer,
    "ppo": PPOTrainer,
}
