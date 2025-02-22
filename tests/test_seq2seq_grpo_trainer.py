# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import tempfile
import unittest

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    WhisperForConditionalGeneration,
)
from transformers.testing_utils import require_peft, require_torch_accelerator

from trl import GRPOConfig, GRPOTrainer, GRPOSeq2SeqTrainer


class GRPOTrainerTester(unittest.TestCase):
    def test_training_reward_func_standard(self):
        # Test if trainer can handle reward function with standard format
        dataset = load_dataset("neulus/whisper-internal-test", split="train")

        import random 
        def reward_func(completions, **kwargs):
            """Random reward function that gives generally higher scores to longer completions."""
            return [random.random() for completion in completions]

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=32,  # reduce the completion length to reduce memory usage
                report_to="none",
            )
            model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-tiny"
            )
            trainer = GRPOSeq2SeqTrainer(
                model=model,
                reward_funcs=reward_func,
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {
                n: param.clone() for n, param in trainer.model.named_parameters()
            }

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(
                    torch.equal(param, new_param), f"Parameter {n} has not changed."
                )
    # def test_training_reward_func_conversational(self):
    #     # Test if trainer can handle reward function with conversational format
    #     dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

    #     def reward_func(completions, **kwargs):
    #         """Reward function that gives higher scores to longer completion content."""
    #         completion_contents = [completion[0]["content"] for completion in completions]
    #         return [float(len(content)) for content in completion_contents]

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         training_args = GRPOConfig(
    #             output_dir=tmp_dir,
    #             learning_rate=0.1,  # increase the learning rate to speed up the test
    #             per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
    #             num_generations=3,  # reduce the number of generations to reduce memory usage
    #             max_completion_length=32,  # reduce the completion length to reduce memory usage
    #             report_to="none",
    #         )
    #         trainer = GRPOTrainer(
    #             model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    #             reward_funcs=reward_func,
    #             args=training_args,
    #             train_dataset=dataset,
    #         )

    #         previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

    #         trainer.train()

    #         self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    #         # Check the params have changed
    #         for n, param in previous_trainable_params.items():
    #             new_param = trainer.model.get_parameter(n)
    #             self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    # def test_training_multiple_reward_funcs(self):
    #     # Test that GRPOTrainer can be instantiated with multiple reward functions
    #     dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

    #     def reward_func1(completions, **kwargs):
    #         """Reward function that rewards longer completions."""
    #         return [float(len(completion)) for completion in completions]

    #     def reward_func2(completions, **kwargs):
    #         """Reward function that rewards completions with more unique letters."""
    #         return [float(len(set(completion))) for completion in completions]

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         training_args = GRPOConfig(
    #             output_dir=tmp_dir,
    #             learning_rate=0.1,  # increase the learning rate to speed up the test
    #             per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
    #             num_generations=3,  # reduce the number of generations to reduce memory usage
    #             max_completion_length=32,  # reduce the completion length to reduce memory usage
    #             report_to="none",
    #         )
    #         trainer = GRPOTrainer(
    #             model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    #             reward_funcs=[reward_func1, reward_func2],
    #             args=training_args,
    #             train_dataset=dataset,
    #         )

    #         previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

    #         trainer.train()

    #         self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    #         # Check the params have changed
    #         for n, param in previous_trainable_params.items():
    #             new_param = trainer.model.get_parameter(n)
    #             self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    # def test_training_multiple_reward_funcs_with_weights(self):
    #     """Test that GRPOTrainer can handle multiple reward functions with weights."""
    #     dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

    #     def reward_func1(completions, **kwargs):
    #         """Reward function that rewards longer completions."""
    #         return [float(len(completion)) for completion in completions]

    #     def reward_func2(completions, **kwargs):
    #         """Reward function that rewards completions with more unique letters."""
    #         return [float(len(set(completion))) for completion in completions]

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         training_args = GRPOConfig(
    #             output_dir=tmp_dir,
    #             learning_rate=0.1,  # increase the learning rate to speed up the test
    #             per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
    #             num_generations=3,  # reduce the number of generations to reduce memory usage
    #             max_completion_length=32,  # reduce the completion length to reduce memory usage
    #             report_to="none",
    #             reward_weights=[0.7, 0.3],  # weight of reward_func1 and reward_func2 respectively
    #         )
    #         trainer = GRPOTrainer(
    #             model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    #             reward_funcs=[reward_func1, reward_func2],
    #             args=training_args,
    #             train_dataset=dataset,
    #         )

    #         previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

    #         trainer.train()

    #         # Check that training logs contain both reward metrics
    #         self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])
    #         self.assertIn("rewards/reward_func1", trainer.state.log_history[-1])
    #         self.assertIn("rewards/reward_func2", trainer.state.log_history[-1])

    #         # Check that the params have changed
    #         for n, param in previous_trainable_params.items():
    #             new_param = trainer.model.get_parameter(n)
    #             self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    # def test_training_multiple_mixed_reward_funcs(self):
    #     # Test if the trainer can handle a mix of reward functions and reward models
    #     dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

    #     def reward_func(completions, **kwargs):
    #         """Reward function that rewards longer completions."""
    #         return [float(len(completion)) for completion in completions]

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         training_args = GRPOConfig(
    #             output_dir=tmp_dir,
    #             learning_rate=0.1,  # increase the learning rate to speed up the test
    #             per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
    #             num_generations=3,  # reduce the number of generations to reduce memory usage
    #             max_completion_length=32,  # reduce the completion length to reduce memory usage
    #             report_to="none",
    #         )
    #         trainer = GRPOTrainer(
    #             model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    #             reward_funcs=[reward_func, "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"],
    #             args=training_args,
    #             train_dataset=dataset,
    #         )

    #         previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

    #         trainer.train()

    #         self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    #         # Check the params have changed
    #         for n, param in previous_trainable_params.items():
    #             new_param = trainer.model.get_parameter(n)
    #             self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    # def test_training_reward_func_additional_column(self):
    #     # Test if trainer can handle reward function that rely on additional columns in the dataset
    #     dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

    #     # Add a column to the dataset (dummy example, the column could be anything)
    #     some_values = list(range(len(dataset)))
    #     dataset = dataset.add_column("some_values", some_values)

    #     def reward_func(completions, some_values, **kwargs):
    #         """Reward function that rewards completions with lengths closer to the values in some_values."""
    #         return [float(abs(len(completion) - value)) for completion, value in zip(completions, some_values)]

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         training_args = GRPOConfig(
    #             output_dir=tmp_dir,
    #             learning_rate=0.1,  # increase the learning rate to speed up the test
    #             per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
    #             num_generations=3,  # reduce the number of generations to reduce memory usage
    #             max_completion_length=32,  # reduce the completion length to reduce memory usage
    #             report_to="none",
    #         )
    #         trainer = GRPOTrainer(
    #             model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    #             reward_funcs=reward_func,
    #             args=training_args,
    #             train_dataset=dataset,
    #         )

    #         previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

    #         trainer.train()

    #         self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    #         # Check the params have changed
    #         for n, param in previous_trainable_params.items():
    #             new_param = trainer.model.get_parameter(n)
    #             self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    # @unittest.skipIf(not is_vllm_available(), "vLLM is not available")
    # @require_torch_accelerator
    # def test_training_vllm(self):
    #     """Test that training works with vLLM for generation."""
    #     dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         training_args = GRPOConfig(
    #             output_dir=tmp_dir,
    #             learning_rate=0.1,  # increase the learning rate to speed up the test
    #             per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
    #             num_generations=3,  # reduce the number of generations to reduce memory usage
    #             max_completion_length=32,  # reduce the completion length to reduce memory usage
    #             report_to="none",
    #             use_vllm=True,
    #             vllm_device="cuda:0",  # will raise a warning, but allows this test to work with only one GPU
    #         )
    #         trainer = GRPOTrainer(
    #             model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
    #             reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
    #             args=training_args,
    #             train_dataset=dataset,
    #         )

    #         previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

    #         trainer.train()

    #         self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    #         # Check that the params have changed
    #         for n, param in previous_trainable_params.items():
    #             new_param = trainer.model.get_parameter(n)
    #             self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    # @unittest.skipIf(sys.platform.startswith("win"), "Skipping on Windows")  # compiling seems to be broken on Windows
    # def test_training_torch_compile(self):
    #     dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         training_args = GRPOConfig(
    #             output_dir=tmp_dir,
    #             learning_rate=0.1,  # increase the learning rate to speed up the test
    #             per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
    #             num_generations=3,  # reduce the number of generations to reduce memory usage
    #             max_completion_length=32,  # reduce the completion length to reduce memory usage
    #             torch_compile=True,
    #             report_to="none",
    #         )
    #         trainer = GRPOTrainer(
    #             model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    #             reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
    #             args=training_args,
    #             train_dataset=dataset,
    #         )

    #         previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

    #         trainer.train()

    #         self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    #         # Check that the params have changed
    #         for n, param in previous_trainable_params.items():
    #             new_param = trainer.model.get_parameter(n)
    #             self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    # def test_training_with_sync_ref_model(self):
    #     dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         training_args = GRPOConfig(
    #             output_dir=tmp_dir,
    #             learning_rate=0.1,  # increase the learning rate to speed up the test
    #             per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
    #             num_generations=3,  # reduce the number of generations to reduce memory usage
    #             max_completion_length=32,  # reduce the completion length to reduce memory usage
    #             sync_ref_model=True,
    #             ref_model_sync_steps=2,  # reduce sync steps to ensure a sync happens
    #             report_to="none",
    #         )
    #         trainer = GRPOTrainer(
    #             model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    #             reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
    #             args=training_args,
    #             train_dataset=dataset,
    #         )

    #         previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

    #         trainer.train()

    #         self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    #         # Check that the params have changed
    #         for n, param in previous_trainable_params.items():
    #             new_param = trainer.model.get_parameter(n)
    #             self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    # @unittest.skipIf(not is_vllm_available(), "vLLM is not available")
    # @require_torch_accelerator
    # @require_peft
    # def test_training_vllm_and_peft(self):
    #     """Test that training works with vLLM for generation."""
    #     dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         training_args = GRPOConfig(
    #             output_dir=tmp_dir,
    #             learning_rate=0.1,  # increase the learning rate to speed up the test
    #             per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
    #             num_generations=3,  # reduce the number of generations to reduce memory usage
    #             max_completion_length=32,  # reduce the completion length to reduce memory usage
    #             use_vllm=True,
    #             report_to="none",
    #         )
    #         trainer = GRPOTrainer(
    #             model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
    #             reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
    #             args=training_args,
    #             train_dataset=dataset,
    #         )

    #         previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

    #         trainer.train()

    #         self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    #         # Check that the params have changed
    #         for n, param in previous_trainable_params.items():
    #             new_param = trainer.model.get_parameter(n)
    #             self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")
