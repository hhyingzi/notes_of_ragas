from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset

from ragas.embeddings.base import (
    HuggingfaceEmbeddings,
    OpenAIEmbeddings,
    embedding_factory,
)
from ragas.exceptions import OpenAIKeyNotFound
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import CallbackManager

    from ragas.embeddings.base import RagasEmbeddings


@dataclass
class AnswerSimilarity(MetricWithLLM):
    """
    Scores the semantic similarity of ground truth with generated answer.
    cross encoder score is used to quantify semantic similarity.
    SAS paper: https://arxiv.org/pdf/2108.06130.pdf

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size.
    model_name:
        The model to be used for calculating semantic similarity
        Defaults open-ai-embeddings
        select cross-encoder model for best results
        https://huggingface.co/spaces/mteb/leaderboard
    threshold:
        The threshold if given used to map output to binary
        Default 0.5
    """

    name: str = "answer_similarity"
    evaluation_mode: EvaluationMode = EvaluationMode.ga
    batch_size: int = 15
    embeddings: RagasEmbeddings = field(default_factory=embedding_factory)
    is_cross_encoder: bool = False
    threshold: float = 0.5

    def __post_init__(self: t.Self):
        # only for cross encoder
        if isinstance(self.embeddings, HuggingfaceEmbeddings):
            self.is_cross_encoder = True if self.embeddings.is_cross_encoder else False
            self.embeddings.encode_kwargs = {
                "batch_size": self.batch_size,
            }

    def init_model(self):
        super().init_model()

        if isinstance(self.embeddings, OpenAIEmbeddings):
            if self.embeddings.openai_api_key == "no-key":
                raise OpenAIKeyNotFound

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        ground_truths, answers = dataset["ground_truths"], dataset["answer"]
        ground_truths = [item[0] for item in ground_truths]  # 多个 ground_truths ，只取第一个来评估

        if self.is_cross_encoder: # 如果用了 cross_encoder 则直接调用 predict() 进行预测
            assert isinstance(self.embeddings, HuggingfaceEmbeddings)
            inputs = [list(item) for item in list(zip(ground_truths, answers))]
            scores = np.array(self.embeddings.predict(inputs))
        else:  # 如果没有使用 cross_encoder 模型，则手动预测
            embeddings_1 = np.array(self.embeddings.embed_documents(ground_truths))  # ground_truths 的 embeddings
            embeddings_2 = np.array(self.embeddings.embed_documents(answers))  # answers 的 embeddings
            similarity = embeddings_1 @ embeddings_2.T  # python3.5+ ，运算符 a @ b 等价于 matmul(a, b.T)，矩阵相乘（不是元素直接相乘，即 * ）
            scores = np.diagonal(similarity)  # numpy.diagonal() 返回矩阵对角线上的值

        # 如果设置了 threshold 阈值，则将分数转换为布尔值（相关，不相关）
        assert isinstance(scores, np.ndarray), "Expects ndarray"
        if self.threshold:
            scores = scores >= self.threshold  # type: ignore

        return scores.tolist()


answer_similarity = AnswerSimilarity()
