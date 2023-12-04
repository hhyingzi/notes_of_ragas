from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset

from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import CallbackManager


@dataclass
class AnswerCorrectness(MetricWithLLM):

    """
    Measures answer correctness compared to ground truth as a combination of
    semantic similarity and factuality

    Attributes
    ----------
    name: string
        The name of the metrics
    batch_size: int
        batch size for evaluation
    weights:
        a list of two weights corresponding to semantic similarity and factuality
        Defaults [0.5, 0.5]
    answer_similarity:
        The AnswerSimilarity object
    faithfulness
        The faithfulness object
    """

    name: str = "answer_correctness"
    evaluation_mode: EvaluationMode = EvaluationMode.qga
    batch_size: int = 15
    weights: list[float] = field(default_factory=lambda: [0.5, 0.5])
    answer_similarity: AnswerSimilarity | None = None
    faithfulness: Faithfulness | None = None

    def __post_init__(self: t.Self):
        # 初始化"回答语义（answer 和 ground_truth）相似度计算工具"
        if self.answer_similarity is None:
            self.answer_similarity = AnswerSimilarity(
                llm=self.llm, batch_size=self.batch_size
            )
        # 初始化“忠实度(answer 和 context)计算工具”
        if self.faithfulness is None:
            self.faithfulness = Faithfulness(llm=self.llm, batch_size=self.batch_size)

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        # 移除数据集中的 "context" 列
        if "contexts" in dataset.column_names:
            ds_faithfulness = dataset.remove_columns(["contexts"])
        else:
            ds_faithfulness = dataset

        # 重构数据集
        ds_faithfulness = ds_faithfulness.rename_columns({"ground_truths": "contexts"})  # 将 ground_truths 重命名为 contexts 列
        faith_scores = self.faithfulness._score_batch(ds_faithfulness)  # 计算忠实度分数（question, answer 和 ground_truths）
        similarity_scores = self.answer_similarity._score_batch(dataset)  # 计算语义相似性分数（answer 和 ground_truths）

        scores_stacked = np.vstack([faith_scores, similarity_scores])  # 将两个分数纵向叠加为矩阵
        scores = np.average(  # 纵向求平均值，即为最终分数
            scores_stacked,
            axis=0,
            weights=self.weights,
        )

        return scores.tolist()


answer_correctness = AnswerCorrectness()
