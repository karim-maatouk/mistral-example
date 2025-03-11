from peft import PeftConfig, PeftModel

def load_finetuned_model(adapter_config_path, adapter_model_path, model_id='mistralai/Mistral-7B-v0.1', max_tokens=512):
    """Lazily loads the fine-tuned model and tokenizer using adapter configuration."""
    global model, tokenizer

    if model is None or tokenizer is None:
        logger.info('Loading tokenizer...')
        tokenizer = get_tokenizer(model_id, max_tokens, add_eos_token=False)

        logger.info('Loading base model...')
        base_model = get_model(model_id=model_id, quantization_config=get_quantization_config())

        logger.info('Loading adapter config...')
        config = PeftConfig.from_pretrained(adapter_config_path) 

        logger.info('Loading fine-tuned model...')
        model = PeftModel.from_pretrained(base_model, adapter_model_path, config=config).eval() 
