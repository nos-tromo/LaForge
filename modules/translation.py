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
    llm_model_config, system_prompt, topic_label_sum_template, topic_docs_sum_template, summarization_template,
    translation_template
)


class LLMAnalysis:
    """
    A class to optimize language models for various tasks such as translation and summarization.

    Attributes:
        language (str): The language code for the target language.
        n_ctx (int): Context size of the language model.
        callback_manager (CallbackManager): Manages callbacks for streaming output.
        model_id (str): Path to the language model.
        llm (LlamaCpp): Instance of the language model.
        system_prompt (PromptTemplate): The system prompt to be populated with the question.
        available_tokens (int): The number of tokens available in the context window after subtracting the prompt.

    Methods:
        _set_callback_manager(): Initializes the callback manager for streaming outputs.
        _load_model(): Loads the language model with specific configurations.
        _create_system_prompt(): Sets up the system prompt for the language model.
        _calculate_available_tokens(): Calculates the available tokens for the language model.
        _convert_to_language_name(): Converts the language code to its full language name.
        _truncate_input_text(): Truncates the input text to the maximum token length available.
        _model_inference(question: str): Performs inference using the language model.
        _sliding_window_summarization(): Summarizes long text using a sliding window approach.
        _replace_keywords_with_titles(df: pd.DataFrame, column: str, titles: list): Replaces keywords in a DataFrame with titles.
        _write_file_output(data: pd.DataFrame | str, task: str): Writes the output data to a file.
        data_pipeline(task: str, data: pd.DataFrame | list | str, input_column: str, output_column: str = None): Processes data through the language model for specified tasks.
    """

    def __init__(
            self,
            language: str,
            n_ctx: int = 8192,
            chunk_overlap: int = 512
    ) -> None:
        """
        Initializes the LLMInference class.

        :param language: The language code for processing.
        :param n_ctx: Context size of the language model.
        :param chunk_overlap: Size of overlapping chunks for the sliding-window mechanism.
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.language = language
        self.n_ctx = n_ctx
        self.chunk_overlap = chunk_overlap

        self._set_callback_manager()
        self._load_model()
        self._create_system_prompt()
        self._calculate_available_tokens()

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
        Loads the language model with specific parameters and configurations.
        """
        try:
            if torch.cuda.is_available():
                cuda_model = list(llm_model_config.items())[0]
                model_name, config = cuda_model
                self.model_id, model_file = config.values()
            elif torch.backends.mps.is_available():
                mps_model = list(llm_model_config.items())[1]
                model_name, config = mps_model
                self.model_id, model_file = config.values()
            else:
                cpu_model = list(llm_model_config.items())[-1]
                model_name, config = cpu_model
                self.model_id, model_file = config.values()

            model_path = str(Path.home() / '.cache' / 'lm-studio' / 'models' / self.model_id / model_file)

            self.llm = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=-1,
                n_ctx=self.n_ctx,
                n_batch=256,
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
        Sets up the system prompt for the language model.
        """
        try:
            template = system_prompt
            self.system_prompt = PromptTemplate.from_template(template)
        except Exception as e:
            self.logger.error(f"Error creating system prompt: {e}", exc_info=True)
            raise

    def _calculate_available_tokens(self) -> None:
        """
        Calculates the number of tokens available for input text after accounting for the prompt and response length.
        """
        try:
            prompt = str(self.system_prompt)
            prompt_length = self.llm.get_num_tokens(prompt)
            self.available_tokens = self.n_ctx - prompt_length
        except Exception as e:
            self.logger.error(f"Error calculating available tokens: {e}", exc_info=True)
            raise

    def _convert_to_language_name(self) -> str | None:
        """
        Converts the language code to its full language name.

        :return: The full language name.
        """
        try:
            return pycountry.languages.get(alpha_2=self.language.lower()).name
        except Exception as e:
            self.logger.error(f'Error converting language: {e}.', exc_info=True)
            return None

    def _truncate_input_text(self, text: str) -> str:
        """
        Truncates the input text to the maximum token length available.

        :param text: The input text to truncate.
        :return: The truncated text.
        """
        try:
            tokens = self.llm.get_num_tokens(text)
            if tokens > self.available_tokens:
                self.logger.info(f"Context window exceeded - Truncating to {self.available_tokens} tokens.")
                tokens = self.llm.get_num_tokens(text[:self.available_tokens])
                text = text[:tokens]
        except Exception as e:
            self.logger.error(f"Error truncating input text: {e}", exc_info=True)

        return text

    def _model_inference(self, question: str) -> str:
        """
        Performs inference using the language model.

        :param question: The input question or text for the model to process.
        :return: The model's response as a string.
        """
        try:
            question = self._truncate_input_text(question)
            llm_chain = self.system_prompt | self.llm
            return llm_chain.invoke({"language": self.language, "question": question})
        except Exception as e:
            self.logger.error(f"Error during model inference: {e}", exc_info=True)
            raise

    def _sliding_window_summarization(self, text: str) -> str:
        """
        Summarizes long text using a sliding-window mechanism to handle texts that exceed the context window size.

        :param text: The long text to summarize.
        :return: The combined summary of the text.
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.available_tokens,
                chunk_overlap=self.chunk_overlap
            )

            chunks = splitter.split_text(text)
            summaries = []
            chunk_number = 0

            for chunk in chunks:
                summary = self._model_inference(summarization_template.format(text=chunk))
                summaries.append(summary)
                chunk_number += 1

            self.logger.info(f"Created summaries for {chunk_number} chunks.")

            combined_summary = "\n".join(summaries)
            final_summary = combined_summary
            combined_summary_tokens = self.llm.get_num_tokens(combined_summary)
            if combined_summary_tokens > self.available_tokens:
                self.logger.info("Combined summary exceeds context window, summarizing again.")
                final_summary = self._model_inference(summarization_template.format(text=combined_summary))
            return final_summary

        except Exception as e:
            self.logger.error(f"Error during sliding window summarization: {e}", exc_info=True)
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

    # def _write_file_output(self, data: str | list | pd.DataFrame, task: str) -> None:
    #     """
    #     Writes the output data to a file.
    #
    #     :param data: The data to write to the file.
    #     :param task: The task associated with the data.
    #     """
    #     try:
    #         pass
    #     except Exception as e:
    #         self.logger.error(f"Error writing file output: {e}", exc_info=True)
    #         raise

    def data_pipeline(
            self,
            task: str,
            data: pd.DataFrame | list | str,
            input_column: str = None,
            output_column: str = None
    ) -> str | pd.DataFrame | None:
        """
        Processes data through the language model for specified tasks.

        :param task: The task to perform (e.g., translation, summarization).
        :param data: The input data to process.
        :param input_column: The column in the DataFrame to read input from.
        :param output_column: The column in the DataFrame to write output to (optional).
        :return: The processed data.
        """
        self.logger.info(f'Creating {task} with model "{self.model_id}" - Initialized.')
        result = None
        try:
            match task:
                case "summary":
                    if isinstance(data, pd.DataFrame):
                        text = "".join(data[input_column])
                    elif isinstance(data, list):
                        text = "".join(data)
                    else:
                        text = data
                    result = self._sliding_window_summarization(text)

                case "translation":
                    if isinstance(data, pd.DataFrame):
                        data[output_column] = data[input_column].apply(
                            lambda x: self._model_inference(translation_template.format(text=x))
                        )
                        result = data.copy()
                    elif isinstance(data, str):
                        result = self._model_inference(translation_template.format(text=data))
                    else:
                        self.logger.error(f"Error - Invalid data format.", exc_info=True)
                case "topic summary":
                    data = data[[input_column, output_column]]  # only progress with keyword and document columns
                    row_list = data.iloc[:, :].values.tolist()  # write the keyword and document columns to a list
                    titles = []
                    summaries = []
                    for item in row_list:
                        # create labels for each topic
                        label_inference = self._model_inference(
                            topic_label_sum_template.format(
                                keywords=item[0],
                                docs=item[1]
                            )
                        )
                        # create summaries for each topic
                        doc_inference = self._model_inference(
                            topic_docs_sum_template.format(
                                title=label_inference,
                                keywords=item[0],
                                docs=item[1],
                            )
                        )
                        titles.append(label_inference)
                        summaries.append(doc_inference)
                    result = self._replace_keywords_with_titles(data, input_column, output_column, titles, summaries)
                case _:
                    self.logger.error(f"Error - No valid task selected.", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error in data pipeline for task '{task}': {e}", exc_info=True)
            raise
        finally:
            self.logger.info(
                f'Processing with model "{self.model_id}" - Finished.'
            )
            # if result is not None:
            #     self._write_file_output(result, task)
        return result
