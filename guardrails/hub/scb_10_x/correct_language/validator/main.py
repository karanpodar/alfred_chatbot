from typing import Callable, Dict, Optional, Union

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

try:
    from fast_langdetect import detect
except ImportError:
    detect = None

try:
    from iso_language_codes import language_name
except ImportError:
    language_name = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# Local imports
from .assets import MODEL_CODES


@register_validator(name="scb-10x/correct_language", data_type="string")
class CorrectLanguage(Validator):
    """
    **Key Properties**

    | Property                      | Description                   |
    | ----------------------------- | ----------------------------- |
    | Name for `format` attribute   | `is-correct-language`         |
    | Supported data types          | `string`                      |
    | Programmatic fix              | translated text if possible   |

    **Description**
    Validates that an LLM-generated text is in the expected language. If the text
    is not in the expected language, the validator will attempt to translate it
    to the expected language.

    Uses the `fast-langdetect` library to detect the language of the input text,
    and the `iso-language-codes` library to get the language names from the ISO codes.
    Meta AI's `facebook/nllb-200-distilled-600M` translation model (available on Huggingface)
    is used to translate the text from the detected language to the expected language.

    **Arguments**
        expected_language_iso (str): The ISO 639-1 code of the expected language. Defaults to "en".
            Please find the ISO 639-1 codes: https://www.loc.gov/standards/iso639-2/php/code_list.php
        threshold (float): The minimum confidence score required to accept the detected language.
        on_fail (str or callable): The action to take when the validation fails. Defaults to None.

    **Example usage**
    ```python
    from validator import IsCorrectLanguage
    validator = IsCorrectLanguage(
        expected_language_iso="de",
        threshold=0.7,
        on_fail="fix"
    )
    ```
    """

    def __init__(
        self,
        expected_language_iso: str = "en",
        threshold: float = 0.7,
        on_fail: Optional[Union[Callable, str]] = None,
        **kwargs,
    ):
        super().__init__(
            expected_language_iso=expected_language_iso,
            threshold=threshold,
            on_fail=on_fail,
            **kwargs,
        )

        self._expected_language_iso = expected_language_iso.strip().lower()
        self._threshold = threshold
        self._translation_model = "facebook/nllb-200-distilled-600M"

        if detect is None:
            raise RuntimeError(
                "The fast-langdetect library is required for this validator. "
                "Please install it using `pip install fast-langdetect` and try again."
            )

        if language_name is None:
            raise RuntimeError(
                "The iso-language-codes library is required for this validator. "
                "Please install it using `pip install iso-language-codes` and try again."
            )

        if pipeline is None:
            raise RuntimeError(
                "The HuggingFace transformers library is required for this validator. "
                "Please install it using `pip install transformers` and try again."
            )

        # Set up the translation pipeline
        try:
            self._translation_pipe = pipeline(
                "translation", model=self._translation_model
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to set up the translation pipeline. Please use a "
                "valid translation model from HuggingFace, and try again."
            ) from e

    def get_translated_text(
        self, text: str, src_lang_iso: str, tgt_lang_iso: str
    ) -> Optional[str]:
        """
        Translate the text from the source language to the target language
        using the translation pipeline

        Args:
            text (str): The input text to be translated
            src_lang_iso (str): The ISO code of the source language
            tgt_lang_iso (str): The ISO code of the target language

        Returns:
            res (str): The translated text
        """

        # Get the language names from the ISO codes
        try:
            src_lang_name = language_name(src_lang_iso)  # type: ignore
            tgt_lang_name = language_name(tgt_lang_iso)  # type: ignore
        except KeyError:
            return None

        # Get the language codes from the language names
        src_lang_code = MODEL_CODES.get(src_lang_name)
        tgt_lang_code = MODEL_CODES.get(tgt_lang_name)

        # If any of the language codes are not found, return the original text
        if src_lang_code is None or tgt_lang_code is None:
            return None

        # Translate the text from the source language to the target language
        translation = self._translation_pipe(
            text, src_lang=src_lang_code, tgt_lang=tgt_lang_code
        )

        if not translation:
            return None

        # Return the translated text
        res = translation[0].get("translation_text")  # type: ignore
        return res

    def validate(self, value: str, metadata: Dict = {}) -> ValidationResult:
        if not isinstance(value, str):
            raise TypeError(f"Expected a string, got {type(value).__name__}")

        # Detect the language of the input text
        prediction = detect(value.splitlines()[0])  # type: ignore
        pred_language_iso, pred_confidence = (
            prediction.get("lang"),
            prediction.get("score"),
        )

        # If detection was not successful, return a PassResult
        if pred_language_iso is None or pred_confidence is None:
            return PassResult()

        # Only consider results with a confidence score above the threshold
        # If the detected language does not match the expected language
        if (
            float(pred_confidence) > self._threshold
            and pred_language_iso != self._expected_language_iso
        ):
            # Return a FailResult with fix_value = value translated to the expected language
            error_message = (
                f"Expected {self._expected_language_iso}, got {pred_language_iso}"
            )
            fix_value = self.get_translated_text(
                    text=value,
                    src_lang_iso=str(pred_language_iso),
                    tgt_lang_iso=self._expected_language_iso,
                )

            return FailResult(
                error_message=error_message,
                fix_value=fix_value,
            )

        # Return a PassResult in all other cases:
        # 1. If the confidence score is above the threshold,
        #   and the predicted language matches the expected language

        # 2. If the confidence score is below the threshold
        # (doesn't matter if the predicted language matches the expected language or not)
        # This is a conservative approach, as we don't want to make a decision based on low confidence
        return PassResult()
