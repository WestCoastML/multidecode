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
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids'].to(self.llm.device)
        mask = self.multi_decode_llm.setup_one_prompt_n_runs(prompt)
        output = self.multi_decode_llm.generate(input_ids=input_ids, mask=mask, n_branch=1, gen_len=5)
        self.assertEqual(output['branch_ids'].shape[1], 1)
        self.assertTrue(isinstance(output['branch_ids'][0][0], torch.Tensor))

    def test_generate_multiple_branches(self):
        prompt = "In a galaxy far, far away"
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids'].to(self.llm.device)
        mask = self.multi_decode_llm.setup_one_prompt_n_runs(prompt)
        output = self.multi_decode_llm.generate(input_ids=input_ids, mask=mask, n_branch=3, gen_len=5)
        self.assertEqual(output['branch_ids'].shape[1], 3)
        for text in output['branch_ids'][0]:
            self.assertTrue(isinstance(text, torch.Tensor))

    def test_generate_with_steer_function(self):
        prompt = "The future of AI is"
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids'].to(self.llm.device)
        mask = self.multi_decode_llm.setup_one_prompt_n_runs(prompt)
        
        def custom_steer(branchs, logits, output):
            return torch.argmax(logits, dim=-1)  # Simple steer function for testing

        output = self.multi_decode_llm.generate(input_ids=input_ids, mask=mask, n_branch=2, gen_len=5, steer=custom_steer)
        self.assertEqual(output['branch_ids'].shape[1], 2)
        for text in output['branch_ids'][0]:
            self.assertTrue(isinstance(text, torch.Tensor))

    # def test_generate_empty_prompt(self):
    #     prompt = ""
    #     with self.assertRaises(ValueError):
    #         self.multi_decode_llm.generate(prompt, n_branch=1, tokens_to_add=5)

if __name__ == '__main__':
    unittest.main()
