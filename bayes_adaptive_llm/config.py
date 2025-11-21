"""
Bayes-Adaptive LLM configuration skeleton.
This mirrors the structure of `baselines/TRIP/config.py` so we can plug in
Bayesian preference search and DPO/MCTS training without refactoring other code.
"""

from config.config import ModelConfig
from config.constants import rec_special_tokens_dict, neg_special_tokens_dict, es_special_tokens_dict, \
    pg_special_tokens_dict


class BayesAdaptiveConfig(ModelConfig):
    """
    Base config for the Bayes-Adaptive LLM pipeline.
    Defaults closely follow TRIP so trainers/pipelines can be reused.
    """

    # backbone
    tokenizer = "roberta-large"
    plm = "roberta-large"
    lm_size = 1024
    combined_action = True
    dropout = 0.1

    # training flow toggles
    run_sft = True
    run_preference_search = False  # generate preference pairs via MCTS loop
    run_dpo = False
    run_offline_eval = False
    run_online_eval = False

    # optimisation
    batch_size = 4
    gradient_accumulation = 1
    learning_rate = 5e-5
    warmup_ratio = 0.05
    weight_decay = 0.01
    num_train_epochs = 3
    logging_steps = 10
    save_total_limit = 2
    max_grad_norm = 1.0

    # inference / prompting
    temperature = 0.7
    max_gen_tokens = 64
    max_sequence_length = 512

    # DPO-related knobs
    dpo_beta = 0.1
    max_length = None
    max_prompt_length = None
    fp16 = False
    bf16 = False
    gradient_checkpointing = False
    reference_model = None
    output_dir = None

    # MCTS loop for preference pair generation
    mcts_iterations = 30
    mcts_num_generate = 4
    mcts_num_evaluate = 4
    mcts_max_depth = 4
    top_k_preferences = 1  # pick the highest-scoring samples to form pairs

    def __init__(self, params):
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class BayesAdaptiveConfigForRecommendation(BayesAdaptiveConfig):
    combined_action = False
    special_tokens_dict = rec_special_tokens_dict
    # recommendation: predict goal only
    n_goals = 14
    n_topics = 1


class BayesAdaptiveConfigForNegotiation(BayesAdaptiveConfig):
    combined_action = True
    special_tokens_dict = neg_special_tokens_dict
    n_goals = 14
    n_topics = 5


class BayesAdaptiveConfigForEmotionalSupport(BayesAdaptiveConfig):
    combined_action = False
    special_tokens_dict = es_special_tokens_dict
    n_goals = 14
    n_topics = 1


class BayesAdaptiveConfigForPersuation(BayesAdaptiveConfig):
    combined_action = False
    special_tokens_dict = pg_special_tokens_dict
    n_goals = 14
    n_topics = 1
