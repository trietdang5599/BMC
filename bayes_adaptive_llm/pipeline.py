"""
Pipeline skeleton for Bayes-Adaptive LLM.
Modeled after `baselines/TRIP/pipeline.py` so we can swap pipelines with minimal changes
and later plug in MCTS-based preference generation plus DPO training.
"""

import os
from typing import Any, Dict, Optional, Sequence, Tuple

from loguru import logger

from base.pipeline import Pipeline
from logger.wandb_logger import WanDBLogger


class BayesAdaptiveLLMPipeline(Pipeline):

    def load_pretrained_model(self, is_rl: bool = False, is_last: bool = False):
        """
        Load the latest supervised/DPO checkpoint.
        RL support is not wired yet but the signature mirrors TRIP/PPDPP.
        """
        if is_rl:
            saved_model_path = os.path.join(self.model_config.saved_dir, "rl_model.pth")
        else:
            saved_model_path = os.path.join(self.model_config.saved_dir, "model.pth")

        if not os.path.exists(saved_model_path):
            raise FileNotFoundError("No pretrained model found at {}".format(saved_model_path))

        self.model = self.trainer.load_model(saved_model_path)

    def execute(self):
        """
        High-level pipeline:
        1) optional SFT
        2) optional preference generation via MCTS
        3) optional DPO training on generated pairs
        4) optional offline/online evaluation
        """
        offline_eval_results, online_eval_results, preference_pairs = None, None, None

        if getattr(self.model_config, "run_sft", False):
            logger.info("Running supervised fine-tuning ...")
            self.run_sft()

        if getattr(self.model_config, "run_preference_search", False):
            logger.info("Generating preference pairs with MCTS loop ...")
            preference_pairs = self.generate_preference_pairs_with_mcts()

        if getattr(self.model_config, "run_dpo", False):
            logger.info("Training with DPO on preference pairs ...")
            # assuming dataset already carries preference data or was just generated
            self.load_pretrained_model(is_rl=False)
            self.trainer.train_dpo(self.dataset, self.device)

        if getattr(self.model_config, "run_offline_eval", False):
            logger.info("Offline evaluation ...")
            self.load_pretrained_model(is_rl=False)
            offline_eval_results = self.run_offline_test()

        if getattr(self.model_config, "run_online_eval", False):
            logger.info("Online evaluation ...")
            self.load_pretrained_model(is_rl=False)
            online_eval_results = self.run_online_test()

        return offline_eval_results, online_eval_results, preference_pairs

    def run_offline_test(self):
        """
        Evaluate the model on a static test set.
        """
        self.trainer.offline_evaluator.reset()
        self.load_pretrained_model(is_rl=False)
        results = self.trainer.test(self.dataset)

        for lg in self.trainer.loggers:
            if not isinstance(lg, WanDBLogger):
                lg.record(results, "Test Set")
        return results

    def run_sft(self):
        """
        Supervised fine-tuning entrypoint.
        """
        return self.trainer.train_sft(self.dataset, self.device)

    def inference(self, instance: Dict[str, Any], action_mapping=None):
        """
        Predict next action for a single instance.
        """
        return self.trainer.predict(instance, action_mapping=action_mapping)

    def generate_preference_pairs_with_mcts(self) -> Sequence[Tuple[Any, Any]]:
        """
        Placeholder for the MCTS preference loop.
        Should return a collection of (chosen, rejected) pairs constructed by
        sampling rollouts and keeping the highest-scoring samples.
        """
        logger.warning("MCTS preference generation not implemented; returning empty list.")
        return []

    def run_online_test(self):
        """
        Stub for online evaluation with simulators.
        """
        raise NotImplementedError("Online evaluation is not implemented for Bayes-Adaptive LLM.")
