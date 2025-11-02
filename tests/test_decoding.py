import unittest
from transformers import AutoTokenizer, AutoModelForCausalLM
from multidecode.mdecode import MultiDecodeLLM
import torch


class TestMultiDecodeLLM(unittest.TestCase):

    def setUp(self):
        # Initialize the tokenizer and model for testing
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", padding_side="left")
        self.llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        self.multi_decode_llm = MultiDecodeLLM(self.llm, self.tokenizer)

    def test_generate_single_branch(self):
        prompt = "Once upon a time"
        generated_texts = self.multi_decode_llm.generate(prompt, n_branch=1, tokens_to_add=5)
        self.assertEqual(len(generated_texts[0]), 1)
        self.assertTrue(isinstance(generated_texts[0][0], str))

    def test_generate_multiple_branches(self):
        prompt = "In a galaxy far, far away"
        generated_texts = self.multi_decode_llm.generate(prompt, n_branch=3, tokens_to_add=5)
        self.assertEqual(len(generated_texts[0]), 3)
        for text in generated_texts[0]:
            self.assertTrue(isinstance(text, str))

    def test_generate_with_steer_function(self):
        prompt = "The future of AI is"
        def custom_steer(branchs, logits, output):
            return torch.argmax(logits, dim=-1)  # Simple steer function for testing

        generated_texts = self.multi_decode_llm.generate(prompt, n_branch=2, tokens_to_add=5, steer=custom_steer)
        self.assertEqual(len(generated_texts[0]), 2)
        for text in generated_texts[0]:
            self.assertTrue(isinstance(text, str))

    # def test_generate_empty_prompt(self):
    #     prompt = ""
    #     with self.assertRaises(ValueError):
    #         self.multi_decode_llm.generate(prompt, n_branch=1, tokens_to_add=5)

if __name__ == '__main__':
    unittest.main()
