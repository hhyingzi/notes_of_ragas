from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM

# prompt 大意：给定一个问题和一个上下文，验证给定上下文中的信息是否有用于回答问题。返回一个是/否的答案。
CONTEXT_PRECISION = HumanMessagePromptTemplate.from_template(
    """\
Given a question and a context, verify if the information in the given context is useful in answering the question. Return a Yes/No answer.
question:{question}
context:\n{context}
answer:
"""
)

@dataclass
class ContextPrecision(MetricWithLLM):
    """
    Average Precision is a metric that evaluates whether all of the
    relevant items selected by the model are ranked higher or not.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_precision"
    evaluation_mode: EvaluationMode = EvaluationMode.qc
    batch_size: int = 15

    # 用于计算批处理数据得分的函数
    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list:
        prompts = []  # 存储提示的列表
        questions, contexts = dataset["question"], dataset["contexts"]  # 从数据集中提取问题和上下文
        # 使用上下文管理器跟踪带有回调函数的批处理操作
        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            # 遍历每个问题和上下文列表对
            for qstn, ctx in zip(questions, contexts):
                # 对问题和上下文生成 prompt
                human_prompts = [
                    ChatPromptTemplate.from_messages(
                        [CONTEXT_PRECISION.format(question=qstn, context=c)]  # 配对为[question, c]
                    )
                    for c in ctx  # 对 context[] 中的每个子元素 c
                ]
                # 将生成的prompt添加到列表中
                prompts.extend(human_prompts)
            # 初始化存储响应的列表
            responses: list[list[str]] = []
            # 对每个 prompt 生成一个响应，响应用来判断“上下文中的信息是否有助于回答问题”，是则响应中含有 "yes" 字符串
            results = self.llm.generate(
                prompts,
                n=1,
                callbacks=batch_group,
            )
            # 从 results 的响应中提取文本，获取最终的响应（针对每个问题）
            responses = [[i.text for i in r] for r in results.generations]
            context_lens = [len(ctx) for ctx in contexts]  # 将 contexts 列表中每个元素 ctx 的长度存储到列表中
            context_lens.insert(0, 0)  # 开头插入元素 0
            context_lens = np.cumsum(context_lens)  # 计算 context 总长度
            # 将 response 进行切分
            grouped_responses = [
                responses[start:end]
                for start, end in zip(context_lens[:-1], context_lens[1:])
            ]
            scores = []

            # 进行打分
            for response in grouped_responses:
                # response 中的每个item，有 yes 字段就转换成1，没有就是 0。把response这样组织成列表。
                response = [int("Yes" in resp) for resp in response]  # [1, 0, 1, 1, 0, 1, ...]
                denominator = sum(response) + 1e-10  # 分母就是列表中 1 的数量。不计算 0，因为用 count() 才会计算0。
                numerator = sum(
                    [
                        (sum(response[: i + 1]) / (i + 1)) * response[i]  # response[] 中每个元素 resp 的位置评分 (位置 i 前面所有 1 的和)/ i
                        for i in range(len(response))
                    ]
                )
                scores.append(numerator / denominator)

        return scores


context_precision = ContextPrecision()
