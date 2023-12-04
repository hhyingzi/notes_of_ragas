from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import trace_as_chain_group
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.embeddings.base import embedding_factory
from ragas.exceptions import OpenAIKeyNotFound
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import CallbackManager

    from ragas.embeddings.base import RagasEmbeddings


QUESTION_GEN = HumanMessagePromptTemplate.from_template(
    """
Generate question for the given answer.
Answer:\nThe PSLV-C56 mission is scheduled to be launched on Sunday, 30 July 2023 at 06:30 IST / 01:00 UTC. It will be launched from the Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, India 
Question: When is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?

Answer:{answer}
Question:
"""  # noqa: E501
)


@dataclass
class AnswerRelevancy(MetricWithLLM):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Attributes
    ----------
    name: string
        The name of the metrics
    batch_size: int
        batch size for evaluation
    strictness: int
        Here indicates the number questions generated per answer.
        Ideal range between 3 to 5.
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy"
    evaluation_mode: EvaluationMode = EvaluationMode.qa
    batch_size: int = 15
    strictness: int = 3
    embeddings: RagasEmbeddings = field(default_factory=embedding_factory)

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
        questions, answers = dataset["question"], dataset["answer"]
        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            prompts = []
            for ans in answers:
                human_prompt = QUESTION_GEN.format(answer=ans)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            results = self.llm.generate(
                prompts,
                n=self.strictness,
                callbacks=batch_group,
            )
            results = [[i.text for i in r] for r in results.generations]  # self.strictness=3，所以结果中含有为answer生成的3个问题。

            scores = []
            for question, gen_questions in zip(questions, results):
                cosine_sim = self.calculate_similarity(question, gen_questions) # 计算余弦相似度值 [0.5, 0.8, ...]
                scores.append(cosine_sim.mean())  # 对每个 question 和其所有 gen_questions 的余弦相似度取平均值，得到最终值 score

        return scores

    # 计算 question 和 generated_questions 的相似度，最终返回的是一个 余弦相似度列表[0.5, 0.8, ...]
    def calculate_similarity(
        self: t.Self, question: str, generated_questions: list[str]
    ):
        assert self.embeddings is not None
        # langchain.embeddings 在本项目中的实现在：src/ragas/embeddings/base.py
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)  # Langchain 的Embedding相关函数，返回一个 List[float]
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)  # Langchain 的Embedding相关函数，返回一个List[List[float]]
        )
        # 求余弦相似度
        # 计算分母||A|| * ||B||，其中norm(A)是计算范式 ||A||
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            # reshape(-1) 将二维数组展平成一维数组，如 [12, 42, 1, 4, 3, 23, ...]
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )


answer_relevancy = AnswerRelevancy()
