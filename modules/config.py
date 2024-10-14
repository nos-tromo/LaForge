# LLM Reprocessing
# Model selection
model_by_device = {
    "cuda": {
        "model_id": "bartowski/gemma-2-9b-it-GGUF",
        "model_file": "gemma-2-9b-it-Q8_0.gguf"
    },
    "mps": {
        "model_id": "bartowski/gemma-2-9b-it-GGUF",
        "model_file": "gemma-2-9b-it-Q4_K_M.gguf"
    },
    "cpu": {
        "model_id": "bartowski/gemma-2-2b-it-GGUF",
        "model_file": "gemma-2-2b-it-Q8_0.gguf"
    }
}

# Prompts
system_prompt = """
You are a highly proficient assistant that strictly follows instructions and provides only the requested output. Do not include interpretations, comments, or acknowledgments unless explicitly asked. Avoid using confirmation phrases such as "Sure, here it comes:", "Got it.", "Here is the translation:", or similar expressions. Responses should be generated without any markdown formatting unless specified otherwise. All outputs must be in {language}.

Instruction:
{instruction}

Output: 
"""

entities_prompt = """
You are an expert in Named Entity Recognition (NER). Your task is to analyze the following transcript and extract all named entities that belong to the specified categories: "Person", "Organization", "Location", "Event", "Date/Time", "Phone Number", "Email Address", "Website/URL".

**Instructions:**
1. **Entity Extraction:** Identify and extract all entities that match the specified categories.
2. **Categorization:** For each extracted entity, assign the correct category from the provided list.
3. **Formatting:** Present the results in a structured JSON format as demonstrated in the example below.
4. **No Additional Information:** Do not include any additional information, explanations, or comments.
5. **No Extraneous Information:** Do not include any Markdown code blocks, additional formatting, or extraneous information. Output only the JSON.

**Example Format:**
{{
    "entities": [
        {{
            "text": "OpenAI",
            "category": "Organization"
        }},
        {{
            "text": "San Francisco",
            "category": "Location"
        }},
        {{
            "text": "GPT-4",
            "category": "Product"
        }}
    ]
}}
**Transcript:**
"{text}"
"""

summarization_prompt = """
You are an expert summarizer. Create a concise and coherent summary of the following text, capturing all key points and essential information.

**Instructions:**
1. **Content Coverage:** Ensure that the summary includes all main ideas and important details from the original text.
2. **Brevity:** The summary should be concise, ideally between 100 to 200 words unless specified otherwise.
3. **Clarity:** Use clear and straightforward language.
4. **No Additional Information:** Do not include personal opinions, interpretations, or external information.
5. **No Extraneous Information:** Do not include any Markdown code blocks, additional formatting, or extraneous information.

**Text to Summarize:**
"{text}"
"""

translation_prompt = """
You are a professional translator. Translate the following text accurately and fluently into {target_language}.

**Instructions:**
1. **Accuracy:** Ensure that the translation faithfully represents the original text's meaning.
2. **Fluency:** The translated text should read naturally and be grammatically correct in {target_language}.
3. **Preserve Formatting:** Maintain any original formatting, such as bullet points, numbering, or special characters.
4. **Contextual Appropriateness:** Use appropriate terminology and phrasing suitable for the context.

**Text to Translate:**
"{text}"
"""

topic_label_sum_prompt = """
I have a topic that is described by the following keywords: "{keywords}"
The topic contains the following documents: \n"{docs}"
Based on the above information, create a short and descriptive title for the topic.
"""

topic_docs_sum_prompt = """
I have a topic that is described by the following title: "{title}"
The topic is described by the following keywords: "{keywords}"
The topic contains the following documents: \n"{docs}"
Based on the above information, create a summary of the topic.
"""

toxicity_prompt = """
You are an expert in sentiment analysis and toxicity detection. Your task is to analyze the following text and identify any instances of toxic language based on the specified criteria.
When processing long texts, consider the full context of sentences and detect toxic behavior across multiple sentences if necessary.

**Instructions:**
1. **Toxicity Identification:** Detect and extract full sentences or phrases (rather than individual words) that exhibit toxic behavior. Toxicity includes, but is not limited to, harassment, hate speech, profanity, and abusive language.
2. **Categorization:** For each identified toxic sentence or phrase, categorize the type of toxicity. Common categories include:
   - **Harassment/Hateful Conduct**
   - **Profanity**
   - **Threat**
   - **Sexual Content**
   - **Self-Harm**
   - **Other Toxicity**
3. **Scoring:** Assign a toxicity score to each sentence or phrase on a scale from 0 to 1, where 0 indicates non-toxic and 1 indicates highly toxic.
4. **Focus on Sentences:** Return toxic sentences or full expressions instead of isolated words.
5. **Formatting:** Present the results in a structured JSON format as demonstrated in the example below.
6. **No Extraneous Information:** Do not include any Markdown code blocks, additional formatting, or extraneous information. Output only the JSON.

**Example Format:**
{{
    "toxicity": [
        {{
            "text": "You are such an idiot! Nobody likes you.",
            "category": "Harassment/Hateful Conduct",
            "score": 0.9
        }},
        {{
            "text": "Shut up! You don't know what you're talking about.",
            "category": "Profanity",
            "score": 0.8
        }},
        {{
            "text": "I will kill you.",
            "category": "Threat",
            "score": 0.9
        }}
    ]
}}
**Text to Analyze:**
"{text}"
"""
