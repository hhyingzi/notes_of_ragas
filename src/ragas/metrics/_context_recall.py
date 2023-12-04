from __future__ import annotations

import typing as t
from dataclasses import dataclass

from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM


# prompt大意：给定一个上下文和一个答案，分析答案中的每个句子，并判断该句子是否可以归因于给定的上下文。
# context：阿尔伯特·爱因斯坦（1879年3月14日至1955年4月18日）是一位德国出生的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的科学家之一。他最著名的成就是发展了相对论理论，他还对量子力学做出了重要贡献，因此成为20世纪前几十年科学对自然的理解进行了革命性改变的中心人物。他的质能等价公式E = mc²是相对论理论的结果，被称为“世界上最著名的方程式”。他因其对理论物理学的贡献，特别是对光电效应定律的发现，而获得了1921年的诺贝尔物理学奖，这是量子理论发展中的关键一步。他的工作也以其对科学哲学的影响而闻名。根据英国期刊《物理世界》1999年对130位全球顶级物理学家的调查，爱因斯坦被评为史上最伟大的物理学家。他的智力成就和独创性使爱因斯坦成为天才的代名词。
# answer:阿尔伯特·爱因斯坦于1879年3月14日出生，是一位德国出生的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的科学家之一。他因对理论物理学的贡献而获得了1921年的诺贝尔物理学奖。他在1905年发表了4篇论文。爱因斯坦于1895年搬到瑞士。
# 分类1：阿尔伯特·爱因斯坦于1879年3月14日出生，是一位德国出生的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的科学家之一。在上下文中明确提到了爱因斯坦的出生日期。因此，可以归因于上下文。[归因]
# 分类2：他因对理论物理学的贡献而获得了1921年的诺贝尔物理学奖。给定的上下文中确实存在这句话。因此，可以归因于上下文。[归因]
# 分类3：在1905年发表了4篇论文。在给定的上下文中没有提到他写了多少篇论文。因此，不能归因于上下文。[未归因]
# 分类4：爱因斯坦于1895年搬到瑞士。在给定的上下文中没有支持这个说法的证据。因此，不能归因于上下文。[未归因]
CONTEXT_RECALL_RA = HumanMessagePromptTemplate.from_template(
    """
Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not.
Think in steps and reason before coming to conclusion. 

context: Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist,widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.
answer: Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895 
classification
1. Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. The date of birth of Einstein is mentioned clearly in the context. So [Attributed]
2. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. The exact sentence is present in the given context. So [Attributed]
3. He published 4 papers in 1905. There is no mention about papers he wrote in given the context. So [Not Attributed]
4. Einstein moved to Switzerland in 1895. There is not supporting evidence for this in the given the context. So [Not Attributed]

context:{context}
answer:{ground_truth}
classification:
"""  # noqa: E501
)


@dataclass
class ContextRecall(MetricWithLLM):

    """
    Estimates context recall by estimating TP and FN using annotated answer and
    retrieved context.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_recall"
    evaluation_mode: EvaluationMode = EvaluationMode.gc
    batch_size: int = 15

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list:
        verdict_token = "[Attributed]"
        prompts = []
        ground_truths, contexts = dataset["ground_truths"], dataset["contexts"]

        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            # 将每个 ground_truths[] 和 context[] 组成元组 (gt, ctx)
            for gt, ctx in zip(ground_truths, contexts):
                # 如果是列表，就将列表中的元素用 \n 拼接起来。如果是单个元素，就不处理。
                gt = "\n".join(gt) if isinstance(gt, list) else gt
                ctx = "\n".join(ctx) if isinstance(ctx, list) else ctx
                human_prompt = CONTEXT_RECALL_RA.format(context=ctx, ground_truth=gt)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            # 用 LLM 对 context, ground_truth 进行分析，收集回答到 responses 中。
            responses: list[list[str]] = []
            results = self.llm.generate(
                prompts,
                n=1,
                callbacks=batch_group,
            )
            responses = [[i.text for i in r] for r in results.generations]

            # 对 response 进行分数计算。
            scores = []
            for response in responses:
                sentences = response[0].split("\n")  # 每个 response 根据换行符分割成句子。
                denom = len(sentences)  # 分母为从 ground_truth 中提取的句子数量总数
                # 分子为句子中，能够从 context 找到线索的句子数量之和。
                numerator = sum(
                    bool(sentence.find(verdict_token) != -1) for sentence in sentences  # 检查 response 中每行句子是否有确认标记，记作布尔值。
                )
                scores.append(numerator / denom)

        return scores


context_recall = ContextRecall()
