from __future__ import annotations

import typing as t
from dataclasses import dataclass

from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from datasets import Dataset

#################
# NLI Score
#################
# Prompt大意：给定一个问题和答案，从给定的答案中创建一个或多个语句。

# 问题：阿尔伯特·爱因斯坦是谁，他以什么而闻名？
# 答案：他是一位德国出生的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的物理学家之一。他以发展相对论理论而最为著名，他还对量子力学理论的发展做出了重要贡献。
# 语句：
# 阿尔伯特·爱因斯坦出生在德国。
# 阿尔伯特·爱因斯坦以他的相对论理论而最为著名。

# 问题：氯化镉在这种化学物质中微溶，它还被称为什么？
# 答案：酒精
# 语句：氯化镉在酒精中微溶。

# 问题：Shahul和Jithin是同一个国籍吗？
# 答案：他们来自不同的国家。
# 语句：Shahul和Jithin来自不同的国家。
LONG_FORM_ANSWER_PROMPT = HumanMessagePromptTemplate.from_template(
    """\
Given a question and answer, create one or more statements from each sentence in the given answer.
question: Who was  Albert Einstein and what is he best known for?
answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
statements:\nAlbert Einstein was born in Germany.\nAlbert Einstein was best known for his theory of relativity.
question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
answer: alcohol
statements:\nCadmium Chloride is slightly soluble in alcohol.
question: Were Shahul and Jithin of the same nationality?
answer: They were from different countries.
statements:\nShahul and Jithin were from different countries.
question:{question}
answer: {answer}
statements:\n"""  # noqa: E501
)

# Prompt 大意：自然语言推理
# 考虑给定的上下文和以下陈述，然后确定它们是否得到上下文中的信息支持。在得出结论之前，请为每个陈述提供简要解释（是/否）。在最后按照给定格式给出每个陈述的最终结论。请不要偏离指定的格式。
# 上下文：约翰是XYZ大学的学生。他正在攻读计算机科学学位。他在本学期注册了几门课程，包括数据结构、算法和数据库管理。约翰是一个勤奋的学生，他花了大量时间学习和完成作业。他经常在图书馆待到很晚来做项目。
#
# 语句：
# 约翰主修生物学。
# 约翰正在修读人工智能课程。
# 约翰是一个专注的学生。
# 约翰有一份兼职工作。
# 约翰对计算机编程感兴趣。

# 回答：
# 约翰主修生物学。
# 解释：上下文明确提到约翰的专业是计算机科学。没有任何信息表明他主修生物学。结论：不是。

# 约翰正在修读人工智能课程。
# 解释：上下文提到了约翰目前注册的课程，但没有提到人工智能。因此，不能推断约翰正在修读人工智能课程。结论：不是。

# 约翰是一个专注的学生。
# 解释：提示中提到他花了大量时间学习和完成作业。此外，还提到他经常在图书馆待到很晚来做项目，这暗示了他的专注程度。结论：是。

# 约翰有一份兼职工作。
# 解释：上下文中没有提到约翰有兼职工作的信息。因此，不能推断约翰有兼职工作。结论：不是。

# 约翰对计算机编程感兴趣。
# 解释：上下文提到约翰正在攻读计算机科学学位，这暗示了他对计算机编程的兴趣。结论：是。
NLI_STATEMENTS_MESSAGE = HumanMessagePromptTemplate.from_template(
    """
Prompt: Natural language inference
Consider the given context and following statements, then determine whether they are supported by the information present in the context.Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.

Context:\nJohn is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
statements:\n1. John is majoring in Biology.\n2. John is taking a course on Artificial Intelligence.\n3. John is a dedicated student.\n4. John has a part-time job.\n5. John is interested in computer programming.\n
Answer:
1. John is majoring in Biology.
Explanation: John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.  Verdict: No.
2. John is taking a course on Artificial Intelligence.
Explanation: The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI. Verdict: No.
3. John is a dedicated student.
Explanation: The prompt states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication. Verdict: Yes.
4. John has a part-time job.
Explanation: There is no information given in the context about John having a part-time job. Therefore, it cannot be deduced that John has a part-time job.  Verdict: No.
5. John is interested in computer programming.
Explanation: The context states that John is pursuing a degree in Computer Science, which implies an interest in computer programming. Verdict: Yes.
Final verdict for each statement in order: No. No. Yes. No. Yes.
context:\n{context}
statements:\n{statements}
Answer:
"""  # noqa: E501
)


@dataclass
class Faithfulness(MetricWithLLM):
    name: str = "faithfulness"
    evaluation_mode: EvaluationMode = EvaluationMode.qac
    batch_size: int = 15

    def _score_batch(
        self: t.Self,
        ds: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        """
        returns the NLI score for each (q, c, a) pair
        """

        question, answer, contexts = ds["question"], ds["answer"], ds["contexts"]
        prompts = []

        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            for q, a in zip(question, answer):
                human_prompt = LONG_FORM_ANSWER_PROMPT.format(question=q, answer=a)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            result = self.llm.generate(prompts, callbacks=batch_group)
            list_statements: list[list[str]] = []
            for output in result.generations:
                # use only the first generation for each prompt
                statements = output[0].text.split("\n")
                list_statements.append(statements)

            prompts = []
            for context, statements in zip(contexts, list_statements):
                statements_str: str = "\n".join(
                    [f"{i+1}.{st}" for i, st in enumerate(statements)]
                )
                contexts_str: str = "\n".join(context)
                human_prompt = NLI_STATEMENTS_MESSAGE.format(
                    context=contexts_str, statements=statements_str
                )
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            result = self.llm.generate(prompts, callbacks=batch_group)
            outputs = result.generations

            scores = []
            final_answer = "Final verdict for each statement in order:"
            final_answer = final_answer.lower()
            for i, output in enumerate(outputs):
                output = output[0].text.lower().strip()
                if output.find(final_answer) != -1:
                    output = output[output.find(final_answer) + len(final_answer) :]
                    score = sum(
                        0 if "yes" in answer else 1
                        for answer in output.strip().split(".")
                        if answer != ""
                    )
                    score = score / len(list_statements[i])
                else:
                    score = max(0, output.count("verdict: no")) / len(
                        list_statements[i]
                    )

                scores.append(1 - score)

        return scores


faithfulness = Faithfulness()
