import unittest
from transformers import AutoTokenizer, AutoModelForCausalLM
from multidecode.mdecode import MultiDecodeLLM
import torch

class TestSetupOnePromptNRuns(unittest.TestCase):
    def setUp(self):
        # Initialize the tokenizer and model for testing
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", padding_side="left")
        self.llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        self.multi_decode_llm = MultiDecodeLLM(self.llm, self.tokenizer)

    def test_setup_one_prompt_n_runs(self):
        prompt = "Once upon a time"
        expected_mask_shape = ...  # Define the expected shape of the mask

        # Call the method
        mask = self.test_instance.setup_one_prompt_n_runs(prompt)

        # Assert the shape of the mask
        self.assertEqual(mask.shape, expected_mask_shape)

        # Optionally test verbose output
        with self.assertLogs(level='INFO') as log:
            self.test_instance.setup_one_prompt_n_runs(prompt, verbose=True)
            self.assertIn("Expected log message", log.output)

if __name__ == '__main__':
    unittest.main()
