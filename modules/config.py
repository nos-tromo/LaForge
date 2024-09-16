# LLM models
llm_model_config = {
    "gemma-2-9b-it-Q8": {
        "model_id": "lmstudio-community/gemma-2-9b-it-GGUF",
        "model_file": "gemma-2-9b-it-Q8_0.gguf",
    },
    "gemma-2-9b-it-Q4": {
        "model_id": "lmstudio-community/gemma-2-9b-it-GGUF",
        "model_file": "gemma-2-9b-it-Q4_K_M.gguf",
    },
    "gemma-2-2b-it-Q8": {
        "model_id": "bartowski/gemma-2-2b-it-abliterated-GGUF",
        "model_file": "gemma-2-2b-it-abliterated-Q8_0.gguf"
    }
}

# LLM prompts
system_prompt = """You are a helpful assistant that follows instructions precisely and responds only with the generated result. You do not provide an interpretation, comments, or other acknowledgements except it is explicitly prompted. You do not use confirmation messages like "Sure, here it comes: ", "Got it", "The translation is..." or similar. You exclude markdown from your response. All your answers and outputs are given in {language}.

Question: {question}

Answer: """

summarization_template = "Create a summary of the following text: '{text}'"
translation_template = "Translate the following text: '{text}'"
topic_label_sum_template = """
I have a topic that is described by the following keywords: '{keywords}'
The topic contains the following documents: \n'{docs}'
Based on the above information, create a short and descriptive title for the topic.
"""
topic_docs_sum_template = """
I have a topic that is described by the following title: '{title}'
The topic is described by the following keywords: '{keywords}'
The topic contains the following documents: \n'{docs}'
Based on the above information, create a summary of the topic.
"""
