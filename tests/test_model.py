from collections import UserDict

from app.config import Settings
from app.model import TransformersTranslator
from app.schemas import TranslateRequest


class FakeInputIds:
    shape = (1, 3)


class FakeGeneratedRow:
    def __getitem__(self, item):
        assert isinstance(item, slice)
        return [101, 102]


class FakeModelInputs(UserDict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, *args, **kwargs):
        return FakeModelInputs({"input_ids": FakeInputIds(), "attention_mask": object()})

    def decode(self, token_ids, skip_special_tokens):
        assert token_ids == [101, 102]
        assert skip_special_tokens is True
        return "translated text"


class FakeModel:
    device = "cuda:0"

    def generate(self, *args, **kwargs):
        assert args == ()
        assert "input_ids" in kwargs
        assert "attention_mask" in kwargs
        assert kwargs["max_new_tokens"] == 1024
        return [FakeGeneratedRow()]


class TestableTransformersTranslator(TransformersTranslator):
    def _load(self):
        self._tokenizer = FakeTokenizer()
        self._model = FakeModel()


def test_transformers_translator_handles_batch_encoding_inputs():
    translator = TestableTransformersTranslator(Settings())

    result = translator.translate(
        TranslateRequest(
            source_lang="en",
            target_lang="zh",
            text="Hello.",
        )
    )

    assert result == "translated text"
