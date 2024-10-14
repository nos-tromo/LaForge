import json
import logging
from pathlib import Path

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import pycountry
import torch

from modules.config import (
    model_by_device, system_prompt, entities_prompt, summarization_prompt, topic_label_sum_prompt, topic_docs_sum_prompt,
    toxicity_prompt, translation_prompt
)


class LLMAnalysis:
    """
    A class to optimize language models for various tasks such as translation and summarization.

    Inherits from:
        FileProcessor

    Attributes:
        language (str): The language code for the target language.
        language_name (str): The name of the target language.
        n_ctx (int): Context size of the language model.
        max_response_length (int): The maximum number of tokens that is allowed for the response.
        callback_manager (CallbackManager): Manages callbacks for streaming output.
        model_id (str): Path to the language model.
        llm (LlamaCpp): Instance of the language model.
        system_prompt (PromptTemplate): The system prompt to be populated with the instruction.
        system_prompt_length (int): The token length of the system prompt.
        available_length (int): The number of tokens available in the context window after subtracting the system prompt and the max response length.

    Methods:
        _convert_to_language_name(): Converts the language code to its full language name.
        _set_callback_manager(): Initializes the callback manager for streaming outputs.
        _load_model(): Loads the language model with specific configurations.
        _create_system_prompt(): Sets up the system prompt for the language model.
        _calculate_available_length(): Calculates the available tokens for the language model.
        _truncate_input_text(): Truncates the input text to the maximum token length available.
        _model_inference(instruction: str): Performs inference using the language model.
        _process_long_text(): Perform entity recognition, summary, or toxicity analysis on long text using a sliding window approach.
        _replace_keywords_with_titles(df: pd.DataFrame, column: str, titles: list): Replaces keywords in a DataFrame with titles.
        _write_file_output(data: pd.DataFrame | str, task: str): Writes the output data to a file.
        data_pipeline(task: str, data: pd.DataFrame | list | str, input_column: str, output_column: str = None): Processes data through the language model for specified tasks.
    """

    def __init__(
            self,
            language: str,
            n_ctx: int = 8192,
            max_response_length: int = 512,
            chunk_overlap: int = 512
    ) -> None:
        """
        Initializes the LLMInference class.

        :param language: The target language for processing.
        :param n_ctx: Context size of the language model.
        :param max_response_length: The maximum number of tokens that is allowed for the response.
        :param chunk_overlap: Size of overlapping chunks for the sliding-window mechanism.
        """

        self.logger = logging.getLogger(self.__class__.__name__)

        self.language = language
        self.language_name = self._convert_to_language_name()
        self.n_ctx = n_ctx
        self.max_response_length = max_response_length
        self.chunk_overlap = chunk_overlap

        self._set_callback_manager()
        self._load_model()
        self._create_system_prompt()
        self._calculate_available_length()

    def _convert_to_language_name(self) -> str | None:
        """
        Converts the language code to its full language name.

        :return: The full language name.
        """
        try:
            return pycountry.languages.get(alpha_2=self.language.lower()).name
        except Exception as e:
            self.logger.error(f"Error converting language: {e}.", exc_info=True)
            return None

    def _set_callback_manager(self) -> None:
        """
        Initializes the callback manager to handle streaming output.
        """
        try:
            self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        except Exception as e:
            self.logger.error(f"Error setting callback manager: {e}", exc_info=True)
            raise

    def _load_model(self) -> None:
        """
        Loads the large language model with specific parameters and configurations.
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

            model = model_by_device.get(device)
            self.model_id = model.get("model_id")
            model_file = model.get("model_file")

            model_path = str(Path.cwd() / "gguf" / model_file)

            self.llm = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=-1,
                n_ctx=self.n_ctx,
                n_batch=256,
                max_tokens=self.max_response_length,
                f16_kv=True,
                callback_manager=self.callback_manager,
                verbose=True,
                seed=1234
            )
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def _create_system_prompt(self) -> None:
        """
        Sets up the system prompt for the large language model.
        """
        try:
            self.system_prompt = PromptTemplate.from_template(system_prompt)
        except Exception as e:
            self.logger.error(f"Error creating system prompt: {e}", exc_info=True)
            raise

    def _calculate_available_length(self) -> None:
        """
        Calculates the number of tokens available for input text after accounting for the prompt and response length.
        """
        try:
            prompt = str(self.system_prompt)
            self.system_prompt_length = self.llm.get_num_tokens(prompt)
            self.available_length = self.n_ctx - self.system_prompt_length - self.max_response_length
        except Exception as e:
            self.logger.error(f"Error calculating available tokens: {e}", exc_info=True)
            raise

    def _truncate_input_text(self, text: str) -> str:
        """
        Truncates the input text to the maximum token length available.

        :param text: The input text to truncate.
        :return: The truncated text.
        """
        try:
            tokens = self.llm.get_num_tokens(text)
            if tokens > self.available_length:
                self.logger.info(f"Context window exceeded - Truncating to {self.available_length} tokens.")
                tokens = self.llm.get_num_tokens(text[:self.available_length])
                text = text[:tokens]
        except Exception as e:
            self.logger.error(f"Error truncating input text: {e}", exc_info=True)

        return text

    def _model_inference(self, instruction: str) -> str:
        """
        Performs inference using the large language model.

        :param instruction: The input instruction or text for the model to process.
        :return: The model's response as a string.
        """
        try:
            instruction = self._truncate_input_text(instruction)
            llm_chain = self.system_prompt | self.llm
            return llm_chain.invoke({"language": self.language_name, "instruction": instruction})
        except Exception as e:
            self.logger.error(f"Error during model inference: {e}", exc_info=True)
            raise

    def _process_long_text(self, text: str, task: str) -> str | pd.DataFrame | None:
        """
        Processes long text using a sliding-window mechanism to handle texts that exceed the context window size.

        :param text: The long text to process.
        :param task: The task to execute ("entities", "summary", or "toxicity").
        :return: The combined result from the text processing, either as type string or DataFrame.
        """
        try:
            prompt = summarization_prompt if task == "summary" else entities_prompt if task == "entities" else toxicity_prompt
            populated_prompt = prompt.format(text=text)
            prompt_length = self.llm.get_num_tokens(populated_prompt)
            results = []
            final_result = None

            if self.n_ctx > self.system_prompt_length + prompt_length + self.max_response_length and task == "summary":
                self.logger.info("System prompt, prompt, and response fit into context window.")
                final_result = self._model_inference(populated_prompt)
            else:
                self.logger.info("Context window exceeded - proceeding with sliding window mechanism.")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.available_length,
                    chunk_overlap=self.chunk_overlap
                )

                chunks = splitter.split_text(text)
                chunk_number = 0

                for chunk in chunks:
                    try:
                        result = self._model_inference(prompt.format(text=chunk))
                        if not result:
                            self.logger.warning(
                                "Received an empty or invalid response from the model.", exc_info=True
                            )
                            continue
                        if task == "entities" or task == "toxicity":
                            clean_result = result.replace("```json", "").replace("```", "").strip()
                            parsed_result = json.loads(clean_result)
                            result = parsed_result.get(task, [])
                    except json.JSONDecodeError as json_error:
                        self.logger.error(f"JSON decoding failed: {json_error}", exc_info=True)
                        continue
                    except Exception as e:
                        self.logger.error(f"Error during model inference or processing: {e}", exc_info=True)
                        raise

                    if task == "summary":
                        results.append(result)
                    else:
                        results.extend(result)
                    chunk_number += 1

                self.logger.info(f"Processed {chunk_number} chunks.")

                if task == "summary":
                    final_result = "\n".join(results)
                    combined_summary_tokens = self.llm.get_num_tokens(final_result)
                    if combined_summary_tokens > self.available_length:
                        self.logger.info("Combined summary exceeds context window, summarizing again.")
                        final_result = self._model_inference(prompt.format(text=final_result))

            if task == "entities" or task == "toxicity":
                if results:
                    final_result = pd.DataFrame(results).drop_duplicates(subset=["text"])  # "text" must be identical to the JSON key in the prompt

            return final_result

        except Exception as e:
            self.logger.error(f"Error during sliding window processing: {e}", exc_info=True)
            raise

    @staticmethod
    def _replace_keywords_with_titles(
            data: pd.DataFrame,
            title_column: str,
            summary_column: str,
            titles: list,
            summaries: list
    ) -> pd.DataFrame:
        """
        Replaces keywords in a DataFrame with titles.

        :param data: The DataFrame containing the data.
        :param title_column: The column in the DataFrame to process.
        :param titles: The list of titles to replace the keywords.
        :param summaries: The list of summaries to replace the keywords.
        :return: The updated DataFrame.
        """
        try:
            df = data.copy()
            df[title_column] = titles
            df[summary_column] = summaries
            return df
        except Exception as e:
            logging.getLogger().error(f"Error replacing keywords with titles: {e}", exc_info=True)
            raise

    def data_pipeline(
            self,
            task: str,
            data: pd.DataFrame | list | str,
            text_column: str = None,
            keywords_column: str = None,
            documents_column: str = None,
            results_column: str = None
    ) -> str | pd.DataFrame | None:
        """
        Processes data through the language model for specified tasks.

        :param task: The task to perform (e.g., translation, summarization).
        :param data: The input data to process.
        :param text_column: The column in the DataFrame to read input from.
        :param keywords_column: The column in the DataFrame with the topics' keywords.
        :param documents_column: The column in the DataFrame with the topics' example documents.
        :param results_column: The column in the DataFrame to write output to (optional).
        :return: The processed data.
        """
        self.logger.info(f"Proceeding with task: '{task}' with model '{self.model_id}' - Initialized.")
        result = None
        try:
            match task:
                case "entities" | "summary" | "toxicity":
                    if isinstance(data, pd.DataFrame):
                        text = "".join(data[text_column])
                    elif isinstance(data, list):
                        text = "".join(data)
                    else:
                        text = data
                    result = self._process_long_text(
                        text=text,
                        task=task
                    )
                case "translation":
                    if isinstance(data, pd.DataFrame):
                        data[results_column] = data[text_column].apply(
                            lambda x: self._model_inference(
                                translation_prompt.format(
                                    target_language=self.language_name,
                                    text=x
                                )
                            )
                        )
                        result = data.copy()
                    elif isinstance(data, str):
                        result = self._model_inference(
                            translation_prompt.format(
                                target_language=self.language_name,
                                text=data
                            )
                        )
                    else:
                        self.logger.error(f"Error - Invalid data format.", exc_info=True)
                case "topic summary":
                    data = data[[keywords_column, documents_column]]
                    row_list = data.iloc[:, :].values.tolist()
                    titles = []
                    summaries = []
                    for item in row_list:
                        # create labels for each topic
                        label_inference = self._model_inference(
                            topic_label_sum_prompt.format(
                                keywords=item[0],
                                docs=item[1]
                            )
                        )
                        # create summaries for each topic
                        doc_inference = self._model_inference(
                            topic_docs_sum_prompt.format(
                                title=label_inference,
                                keywords=item[0],
                                docs=item[1],
                            )
                        )
                        titles.append(label_inference)
                        summaries.append(doc_inference)
                    result = self._replace_keywords_with_titles(
                        data,
                        keywords_column,
                        documents_column,
                        titles,
                        summaries
                    )
                case _:
                    self.logger.error(f"Error - No valid task selected.", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error in data pipeline for task '{task}': {e}", exc_info=True)
            raise
        finally:
            self.logger.info(
                f"Proceeding with task: '{task}' with model '{self.model_id}' - Finished."
            )
        return result
