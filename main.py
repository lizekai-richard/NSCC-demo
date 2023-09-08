from transformers import LlamaTokenizer, LlamaForCausalLM
from argparse import ArgumentParser


def load_model(model_path):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    return tokenizer, model





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/path/to/model")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)

    input = "A step by step recipe to make bolognese pasta: "
    tokenized_input = tokenizer(input, return_tensors='pt').input_ids
    output_ids = model.generate(tokenized_input)
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(output)