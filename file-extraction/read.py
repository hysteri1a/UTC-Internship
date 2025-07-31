import os
import re
from pathlib import Path
from typing import Optional
from typing import AsyncGenerator, Dict, Any, List, Tuple, Union
import json
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dataclasses import dataclass, field
from docx import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from function import normalize,lcs_similarity,safe_eval, json_to_readable
from auxilary import WordDocumentProcessor, ChapterReaderTool


@dataclass
class ReviewRule:
    """评审规则类"""
    name: str
    criteria: str
    score_range: str
    details: str = ""
    reference_paragraphs: List[int] = field(default_factory=list)

    # 新增层级结构支持
    section: str = ""  # 大项名称
    subsection: str = ""  # 子项名称
    section_score_range: str = ""  # 大项分值范围
    subsection_score_range: str = ""  # 子项分值范围

class ReviewSystem:
    """评审系统主类"""

    def __init__(self, chapters_dir: str = "./chapters"):
        self.chapters_dir = Path(chapters_dir)
        self.chapter_reader = ChapterReaderTool(chapters_dir)
        self.review_rules = []
        self.rules_learned = False
        self.principle_chapter_name: Optional[str] = None  # 存储包含评审原则的章节名
        self.scoring_tables = []  # 新增：存储所有评分表格信息
        # 初始化大语言模型
        self.llm = ChatOpenAI(
            base_url="http://192.168.1.224:8000/v1",
            api_key="EMPTY",
            model="qwen3-32b",
            temperature=0.1,
            streaming = True,
        )

        # 创建代理工具
        self.tools = [self.chapter_reader]
        self.agent = self._create_agent()

    def _create_agent(self):
        """创建带有章节阅读工具的代理"""
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate

        # 用原始字符串和 Markdown 代码块，避免手动转义引号和大括号导致的控制字符问题
        system_prompt = r"""
        你是一个专业的文档评审专家。你的任务是分析文档内容，提取所有评审得分相关的表格内容。  
        文档中可能包含多个评分表格，比如技术评分细则、商务评分细则、报价评分细则等。  
        你可以使用工具来查阅文档的任何章节。  
        请仔细分析内容，提取出所有的评分表格和规则，并识别每个表格的层级结构。

        下面请按照 **严格的 JSON** 格式返回所有的评审表格（不要包含多余注释），注意要包含文档中的每一个评分表格：

        ```json
        {{
          "scoring_tables": [
            {{
              "table_name": "表格名称，例如：技术评分细则",
              "table_description": "表格的文字描述",
              "total_score_range": "总分的取值范围，比如 0～100",
              "weight": "权重（如果有）",
              "sections": [
                {{
                  "section_name": "大项名称",
                  "section_score_range": "大项分值范围",
                  "subsections": [
                    {{
                      "subsection_name": "子项名称",
                      "subsection_score_range": "子项分值范围",
                      "items": [
                        {{
                          "item_name": "具体评分项名称",
                          "description": "详细评分标准描述",
                          "score_range": "子项分值范围",
                          "scoring_method": "评分方法说明"
                        }}
                      ]
                    }}
                  ]
                }}
              ]
            }}
          ],
          "overall_scoring_method": "整体评分方法说明",
          "total_possible_score": "所有表格的总分"
        }}
        """
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # 创建代理
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def load_chapters(self, doc_path: str) -> List[str]:
        """加载并分割文档章节"""
        processor = WordDocumentProcessor(str(self.chapters_dir))
        chapter_names = processor.extract_chapters(doc_path)
        return chapter_names

    async def find_review_principles_stream(self, chapter_names: List[str], threshold: float = 0.01) -> AsyncGenerator[
        Dict[str, Any], None]:
        """使用大语言模型查找包含评审原则的章节"""
        yield {"status": "start", "data": "开始分析各章节以查找评审原则..."}

        from langchain.prompts import (
            ChatPromptTemplate,
            SystemMessagePromptTemplate,
            HumanMessagePromptTemplate,
        )

        # 构建章节分析提示
        analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """你是一个专业的文档分析助手。
        你的任务是分析文档章节，找出包含评审原则、评分标准或评分规则的章节。

        评审原则/评分标准的特征：
        1. 包含“评分标准”、“评分规则”、“评审原则”等关键词。
        2. 包含具体的分数、分值、得分计算方法。
        3. 描述评价标准和评价维度。
        4. 包含评分细则或评分流程。

        请分析以下章节列表及内容，返回最符合条件的章节名称；若无符合，则返回“未找到”。"""
            ),
            HumanMessagePromptTemplate.from_template(
                "章节列表和内容：\n{chapters_info}"
            ),
        ])

        # 准备章节信息
        chapters_info = []
        for chapter_name in chapter_names:
            content = self.chapter_reader._run(chapter_name)
            preview = content
            chapters_info.append(f"章节名: {chapter_name}\n内容预览: {preview}\n" + "=" * 50)
        chapters_text = "".join(chapters_info)
        print(len(chapters_info))
        # 格式化 prompt，得到 ChatPromptValue
        prompt_value = analysis_prompt.format_prompt(chapters_info=chapters_text)

        # 转换为消息格式
        messages = []
        for message in prompt_value.messages:
            # LangChain 消息类型映射
            if message.__class__.__name__ == "SystemMessage":
                role = "system"
            elif message.__class__.__name__ == "HumanMessage":
                role = "user"
            elif message.__class__.__name__ == "AIMessage":
                role = "assistant"
            else:
                role = "user"  # 默认为用户消息

            messages.append({
                "role": role,
                "content": message.content
            })
        buffer = ""
        try:
            async for chunk in self.llm.astream(messages):
                token = chunk.content
                buffer += token
        except Exception as e:
            yield {"status": "error", "data": f"流处理错误: {str(e)}"}
            return
        result_chapter = buffer.strip()
        res_norm = normalize(result_chapter)
        yield {"status": "model_done", "data": f"模型推荐章节: {result_chapter}"}
        if result_chapter == "未找到":
            yield {"status": "complete", "data": "未找到相关章节"}
            return

        # 对每个章节名计算相似度，选出最高的
        best_name, best_score = None, 0.0
        for name in chapter_names:
            name_norm = normalize(name)
            score = lcs_similarity(res_norm, name_norm)
            if score > best_score:
                best_score, best_name = score, name

        # 检查是否通过阈值
        if best_score >= threshold:
            full_content = self.chapter_reader._run(best_name)  # 或者你的读取函数
            yield {"status": "complete", "found": True, "chapter": best_name, "content": full_content,
                   "all_chapters": chapter_names, "data": ""}

    async def learn_review_rules_stream(
            self,
            content: str,
            all_chapters: List[str],
            agent=None
    ) -> AsyncGenerator[Union[Dict[str, str], Tuple[str, Dict[str, str]]], None]:
        """使用代理学习评审规则，支持按三部分细则分别提取并返回权重和原文"""
        score_mapping = {
            "技术评分细则": "technical",
            "商务评分细则": "business",
            "价格评分细则": "price"
        }
        sections = ["技术评分细则", "商务评分细则", "价格评分细则"]
        combined_weights: Dict[str, str] = {}
        raw_sections: Dict[str, str] = {}
        import asyncio
        yield {"status": "start", "data": f"开始学习评审规则，共{len(sections)}部分...请等待"}

        def create_analysis_prompt(section):
            system_message = f"""
                你是一个专业的文档分析助手。你的任务是分析给定的评审原则文档，提取出其中所有的关于\"{section}\"评分表格。
                重要提醒：
                1. 仔细查看整个文档，不要遗漏任何一个包含评分标准的表格。
                2. 每个相关表格都要完整提取，包括表格标题、分值范围、评分项目等。
                3. 必须且仅按照严格的JSON格式输出，所有key必须使用英文。所有的value内容保持中文。
                输出格式要求：
                - 最外层必须包含 scoring_tables 数组。
                - 每个表格包含 table_name, table_description, total_score_range, weight, sections。
                - 每个section包含 section_name, section_score_range, subsections。
                - 每个subsection包含 subsection_name, subsection_score_range, items。
                - 每个item包含 item_name, description, score_range, scoring_method。
                4. 注意json结构，最外层只能有scoring_tables，当scoring_tables对应内容的列表闭合后，立即结束该字典。
                5. 确保字符串json结构可被json.loads解析, 解析完后会转字符串呈现给用户，要求内容是user-friendly的。
                """

            human_message = "文档内容：\n{content}\n\n所有章节列表：{all_chapters}\n"

            return ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_message),
                HumanMessagePromptTemplate.from_template(human_message)
            ])

        async def process_section(section: str) -> Tuple[str, Dict[str, str], str, bool]:
            """处理单个section，返回(section_name, weights, part_data, is_skipped)"""
            # 构建 Prompt
            analysis_prompt = create_analysis_prompt(section)
            prompt_value = analysis_prompt.format_prompt(
                content=content,
                all_chapters=', '.join(all_chapters)
            )
            messages = []
            for msg in prompt_value.messages:
                if hasattr(msg, 'type'):
                    role = 'system' if msg.type == 'system' else 'user'
                else:
                    role = 'system' if 'System' in msg.__class__.__name__ else 'user'
                messages.append({"role": role, "content": msg.content})

            # 使用异步调用LLM（如果支持ainvoke）
            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke(messages)
            else:
                # 如果不支持ainvoke，使用线程池异步执行
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.llm.invoke, messages
                )

            buffer = response.content

            # 检查是否跳过
            if "未找到评分表格" in buffer:
                return section, {}, "", True

            # 解析JSON
            json_text = buffer
            json_text = re.sub(r"^.*?```json\s*", "", json_text, flags=re.S)
            json_text = re.sub(r"\s*```.*$", "", json_text, flags=re.S)
            json_text = json_text.replace('\r', ' ').replace('\n', ' ')

            open_count = json_text.count('{')
            close_count = json_text.count('}')
            if open_count > close_count:
                json_text += '}' * (open_count - close_count)
            last = json_text.rfind('}')
            if last != -1:
                json_text = json_text[: last + 1]

            part_data = json.loads(json_text)

            # 提取权重函数
            def extract_weights(data: Dict) -> Dict[str, str]:
                weights: Dict[str, str] = {}
                for table in data.get("scoring_tables", []):
                    name = table.get("table_name", "未知表格")
                    if "weight" in table:
                        weights[name] = table["weight"]
                    for sec in table.get("sections", []):
                        sec_name = sec.get("section_name", "未知章节")
                        if "weight" in sec:
                            weights[f"{name} - {sec_name}"] = sec["weight"]
                        for sub in sec.get("subsections", []):
                            sub_name = sub.get("subsection_name", "未知子章节")
                            if "weight" in sub:
                                weights[f"{name} - {sec_name} - {sub_name}"] = sub["weight"]
                            for item in sub.get("items", []):
                                item_name = item.get("item_name", "未知项目")
                                if "weight" in item:
                                    weights[f"{name} - {sec_name} - {sub_name} - {item_name}"] = item["weight"]
                return weights

            sec_weights = extract_weights(part_data)
            self._process_rules_data(part_data)

            return section, sec_weights, part_data, False

        # 预启动所有异步任务
        section_tasks = []
        for section in sections:
            task = asyncio.create_task(process_section(section))
            section_tasks.append((section, task))

        # 按顺序处理每个section的结果
        for i, (section, task) in enumerate(section_tasks):
            yield {"status": "section_start", "section": section}

            # 等待当前section的LLM处理完成
            section_name, sec_weights, part_data, is_skipped = await task

            if is_skipped:
                yield {"status": "section_skip", "message": section, "data": "未找到评分表格"}
                continue

            # 更新权重
            combined_weights.update(sec_weights)

            # 当前section的内容
            section_content = []
            async for chunk in json_to_readable(part_data):
                section_content.append(chunk)

            # 在流式输出完成后，保存当前section的内容
            raw_sections[score_mapping[section]] = "".join(section_content)
            yield {"status": "process", "data": section + "部分学习完毕", "message": ""}

        print("\n✅ 所有评分表格学习完成！可以开始传入文档进行评分。")
        yield {"status": "end", "message": [combined_weights, raw_sections], "data": ""}
    def _display_table_summary(self):
        """显示表格汇总信息"""
        print("\n📊 评分表格汇总:")

        for table_info in self.scoring_tables:
            print(f"\n【{table_info['name']}】")
            if table_info['total_score_range']:
                print(f"  分值范围: {table_info['total_score_range']}")
            if table_info['weight']:
                print(f"  权重: {table_info['weight']}")

            # 按大项分组显示规则
            current_section = ""
            for rule in table_info['rules']:
                if hasattr(rule, 'section') and rule.section != current_section:
                    current_section = rule.section
                    print(f"  └─ {current_section} ({rule.section_score_range})")

                item_name = rule.name.split(' - ')[-1]  # 只显示具体项名称
                print(f"     ├─ {item_name} ({rule.score_range})")

    def _process_rules_data(self, rules_data: dict) -> bool:
        """处理多表格数据并更新系统状态"""

        # 处理所有评分表格
        tables = rules_data.get("scoring_tables", [])
        if not tables:
            print("❌ 未找到任何评分表格")
            print(rules_data)
            return False

        print(f"✅ 发现 {len(tables)} 个评分表格")

        total_rules_count = 0

        for table_idx, table in enumerate(tables):
            table_name = table.get("table_name", f"表格{table_idx+1}")
            table_description = table.get("table_description", "")
            total_score_range = table.get("total_score_range", "")
            weight = table.get("weight", "")

            print(f"\n【{table_name}】")
            if table_description:
                print(f"  描述: {table_description}")
            if total_score_range:
                print(f"  分值范围: {total_score_range}")
            if weight:
                print(f"  权重: {weight}")

            # 存储表格信息
            table_info = {
                "name": table_name,
                "description": table_description,
                "total_score_range": total_score_range,
                "weight": weight,
                "rules": []
            }

            # 处理表格中的评分规则
            table_rules_count = 0
            for section in table.get("sections", []):
                section_name = section.get("section_name", "")
                section_score_range = section.get("section_score_range", "")

                for subsection in section.get("subsections", []):
                    subsection_name = subsection.get("subsection_name", "")
                    subsection_score_range = subsection.get("subsection_score_range", "")

                    for item in subsection.get("items", []):
                        # 构建完整的规则名称，包含表格名称
                        full_name = f"{table_name} - {section_name} - {subsection_name} - {item.get('item_name', '')}"

                        rule = ReviewRule(
                            name=full_name,
                            criteria=item.get("description", ""),
                            score_range=item.get("score_range", ""),
                            details=item.get("scoring_method", "")
                        )

                        # 添加层级和表格信息
                        rule.table_name = table_name
                        rule.section = section_name
                        rule.subsection = subsection_name
                        rule.section_score_range = section_score_range
                        rule.subsection_score_range = subsection_score_range
                        rule.table_weight = weight

                        self.review_rules.append(rule)
                        table_info["rules"].append(rule)
                        table_rules_count += 1
                        total_rules_count += 1

            self.scoring_tables.append(table_info)
            print(f"  包含 {table_rules_count} 条评分规则")

        # 保存整体评分信息
        self.overall_scoring_method = rules_data.get("overall_scoring_method", "")
        self.total_possible_score = rules_data.get("total_possible_score", "")

        if self.review_rules:
            print(f"\n🎉 总共识别到 {total_rules_count} 条评分规则，分布在 {len(tables)} 个表格中")

            if self.overall_scoring_method:
                print(f"整体评分方法: {self.overall_scoring_method}")
            if self.total_possible_score:
                print(f"总分: {self.total_possible_score}")

            # 按表格分组显示详细信息
            self._display_table_summary()

            self.rules_learned = True
            return True
        else:
            print("❌ 未能识别出任何评分规则")
            return False

    def _parse_review_rules(self, content: str) -> List[ReviewRule]:
        """解析评审规则"""
        rules = []

        # 按段落分割
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]

        current_rule = None

        for paragraph in paragraphs:
            # 检测规则标题
            if self._is_rule_heading(paragraph):
                if current_rule:
                    rules.append(current_rule)

                current_rule = ReviewRule(
                    name=paragraph,
                    criteria="",
                    score_range=self._extract_score_range(paragraph),
                    details=""
                )
            elif current_rule:
                # 累积规则详情
                if current_rule.criteria:
                    current_rule.criteria += " " + paragraph
                else:
                    current_rule.criteria = paragraph

        # 添加最后一个规则
        if current_rule:
            rules.append(current_rule)

        return rules

    def _is_rule_heading(self, text: str) -> bool:
        """判断是否为规则标题"""
        patterns = [
            r'^\d+[\.、]',  # 数字编号
            r'^[（(]\d+[）)]',  # 括号编号
            r'^[一二三四五六七八九十]+[、.]',  # 中文数字
            r'^[A-Z][a-z]*:',  # 英文标题
        ]

        for pattern in patterns:
            if re.search(pattern, text):
                return True

        # 检查是否包含评分关键词
        score_keywords = ["分", "评分", "得分", "分值", "分数"]
        return any(keyword in text for keyword in score_keywords)

    def _extract_score_range(self, text: str) -> str:
        """提取分值范围"""
        patterns = [
            r'(\d+)[-~至到](\d+)分',
            r'满分(\d+)分',
            r'(\d+)分',
            r'(\d+)%'
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)

        return ""

    async def score_document_stream(self, data_folder: str, chapter_name: str, section_name: str, weights:Dict[str, float] = None)-> AsyncGenerator[Union[Dict[str, Any], str], None]:
        """
        为指定部分的文档进行评分

        Args:
            data_folder: 文档文件夹路径
            chapter_name: 章节名称
            section_name: 评分部分名称（如：技术评分细则、商务评分细则、价格评分细则）
            weights: 该部分的权重配置

        Returns:
            tuple: (评分文本, 得分, 详细信息列表)
        """
        if not getattr(self, 'rules_learned', False):
            yield {"status": "error", "data": "评审规则尚未学习，请先学习评审原则"}
            return

        yield {"status": "start", "data": f"开始评分章节：{chapter_name} 部分：{section_name}"}

        doc_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.docx')]
        if not doc_files:
            yield {"status": "error", "data": f"在文件夹 {data_folder} 中未找到任何Word文档"}
            return

        score_mapping = {
            "technical": "技术评分细则",
            "business": "商务评分细则",
            "price": "价格评分细则"
        }
        section_rules = [rule for rule in self.review_rules if rule.name.startswith(score_mapping[section_name])]
        if not section_rules:
            yield {"status": "error", "data": f"未找到 {section_name} 相关的评审规则"}
            return

        # 构建规则信息
        rules_info: List[str] = []
        for rule in section_rules:
            text = f"规则：{rule.name}\n标准：{rule.criteria}\n分值：{rule.score_range}\n"
            if rule.details:
                text += f"要点：{rule.details}\n"
            if rule.reference_paragraphs:
                paras = self.processor.get_paragraphs(chapter_name)
                ref_texts = [paras[i] for i in rule.reference_paragraphs if i in paras]
                text += "参考段落：" + " | ".join(ref_texts) + "\n"
            rules_info.append(text)

        # 处理权重归一化
        if weights is not None and isinstance(weights, dict):
            try:
                # 将字符串值转换为数字并计算总和
                numeric_weights = {key: float(value) for key, value in weights.items()}
                total_weight = sum(numeric_weights.values())

                if total_weight == 100:
                    # 将所有值除以100进行归一化
                    weights = {key: value / 100 for key, value in numeric_weights.items()}
                    print(
                        f"权重总和为100，已归一化: 原始权重总和={total_weight} -> 归一化后权重总和={sum(weights.values())}")
                else:
                    print(f"权重总和为{total_weight}，无需归一化")
                    weights = numeric_weights
            except (ValueError, TypeError) as e:
                print(f"权重转换失败: {e}")
        # 准备所有文档内容
        contents: List[str] = []
        for fname in doc_files:
            path = os.path.join(data_folder, fname)
            try:
                doc = Document(path)
                content = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                contents.append(f"文件名: {fname}\n内容:\n{content}")
            except Exception:
                continue
        # 转换为字符串格式用于prompt
        first_key = next(iter(weights))
        weights_str = str(weights[first_key])
        # 构建prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "请根据以下评审规则及参考段落给出这一部分的权重与评分：\n{rules}"),
            ("human", "你会被给予规则中某一部分的信息。请完成以下步骤：\n"
                      "1. 找出待评分文档的内容属于那一类规则\n"
                      "2. 依据规则为该部分评分\n"
                      "3. 将计算过程封装在Python可执行表达式中\n"
                      "4. 表达式只能使用 `+ - * / abs( )` 等内置运算，不要出现其他任何中文或复杂标记。给出表达式时，也要给出对应规则的权重，权重为:\n" + weights_str + "\n"
                      "5. 先推导给出每个分数的原因，然后使用特殊标记来包裹计算的表达式：<calc>{{表达式}}</calc>\n\n"
                     "例如：你判断得分为5,5,0,10,20。请你先给出这样打分的原因，然后以: <calc> (5+5+0+10+20) * 权重 </calc> 结尾\n\n"
                     "其中，小括号包裹表达式是某一部分的评分细则累积的表达式，系数是对应部分的权重。注意,每个部分乘上权重后才是最终结果。缺少的信息请不要假设，一律按照没有进行考虑。请按照我给你的格式输出。待评分文档内容：\n\n{content}"
                      "6. 每个部分都应按规则的顺序打分，即使是0分也要打上, 只在最后结果采用<calc>格式输出。不要重复输出内容，不要重复打分。")
        ])
        merged_content = "\n" + "=" * 40 + "\n".join(contents)

        params = {"rules": "".join(rules_info), "content": merged_content}

        # 使用 format_prompt 而不是 format
        prompt_value = prompt.format_prompt(**params)

        # 转换为 OpenAI 格式的消息
        messages = []
        for msg in prompt_value.messages:
            if hasattr(msg, 'type'):
                # 新版本 LangChain
                role = "system" if msg.type == "system" else "user" if msg.type == "human" else "assistant"
            else:
                # 通过类名判断
                role = "system" if "System" in msg.__class__.__name__ else "user"

            messages.append({"role": role, "content": msg.content})

        buffer = ""
        try:
            async for chunk in self.llm.astream(messages):
                token = chunk.content
                buffer += token
                yield {"status": "stream", "message": section_name, "data": token}
        except Exception as e:
            yield {"status": "error", "data": f"流处理错误: {str(e)}"}
            return
        print(buffer)
        res_text = buffer.strip()
        calc_match = re.search(r'<calc>(.*?)</calc>', res_text, re.DOTALL)

        if calc_match:
            calc_expr = calc_match.group(1).strip()

            # 计算得分
            section_score = safe_eval(calc_expr)

            # 清理响应文本格式
            if res_text.startswith("```"):
                res_text = re.sub(r"^```(json)?", "", res_text).rstrip("`")

            document_results = {
                "status": "section_done",
                "message":{"section": section_name,"response": res_text.strip(),"calc_expression": calc_expr},
                "data": section_score,
            }

            yield  document_results
            return

        else:
            yield {
                "status": "error",
                "message":{"section": section_name,"response": res_text},
                "data": "未找到计算表达式"
            }
            return

    # 如果需要流式返回版本，可以使用这个版本
    async def process_multiple_image_analysis_with_llm_stream(self, image_analysis_results, all_details, weights_str):
        """调用大模型处理多个文件的图片分析结果 - 流式版本"""
        # 构建prompt模板
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "请根据以下评审规则综合分析投标文件的图片识别结果和评分详情。"),
            ("human", "你会被给予多个文件的图片分析结果和评分详情。请完成以下步骤：\n"
                      "1. 综合分析所有文件的图片识别结果\n"
                      "2. 对比图片信息与原始评分详情\n"
                      "3. 根据评分规则和新添加的额外文件图片信息，更新评分\n"
                      "4. 提供综合评估和专业建议,输出格式请对用户友好\n\n"
                      "5. 将计算过程封装在Python可执行表达式中\n"
                      "6. 表达式只能使用 `+ - * / abs( )` 等内置运算，不要出现其他任何中文或复杂标记。给出表达式时，也要给出对应规则的权重，权重为:\n" + str(weights_str) + "\n"
                      "7. 先推导给出每个分数的原因，然后使用特殊标记来包裹计算的表达式：<calc>{{表达式}}</calc>\n\n"
                     "例如：你判断得分为5,5,0,10,20。请你先给出这样打分的原因，然后以: <calc> (5+5+0+10+20) * 权重 </calc> 结尾\n\n"
                     "其中，小括号包裹表达式是某一部分的评分细则累积的表达式，系数是对应部分的权重。注意,每个部分乘上权重后才是最终结果。缺少的信息请不要假设，一律按照没有进行考虑。请按照我给你的格式输出。"
                      "8. 每个部分都应按规则的顺序打分，即使是0分也要打上, 只在最后结果采用<calc>格式输出。不要重复输出内容，不要重复打分。"
                      "## 额外文件图片信息：\n{image_info}\n\n"
                      "## 未得到额外文件图片信息时的评分：\n{original_details}\n\n"
                      "请根据评分规则和新添加的额外文件图片信息，更新你未得到额外文件图片信息时的评分。")
        ])
        # 准备参数
        params = {
            # "rules": self.review_rules,
            "image_info": json.dumps(image_analysis_results, ensure_ascii=False, indent=2),
            "original_details": json.dumps(all_details, ensure_ascii=False, indent=2)
        }
        # 使用 format_prompt 而不是 format
        prompt_value = prompt.format_prompt(**params)
        # 转换为 OpenAI 格式的消息
        messages = []
        for msg in prompt_value.messages:
            if hasattr(msg, 'type'):
                # 新版本 LangChain
                role = "system" if msg.type == "system" else "user" if msg.type == "human" else "assistant"
            else:
                # 通过类名判断
                role = "system" if "System" in msg.__class__.__name__ else "user"

            messages.append({"role": role, "content": msg.content})
        print(messages['content'])

        # 流式处理
        buffer = ""
        try:
            async for chunk in self.llm.astream(messages):
                token = chunk.content
                buffer += token
                yield {"status": "stream", "message": "图片分析综合处理", "data": token}
        except Exception as e:
            yield {"status": "error", "data": f"流处理错误: {str(e)}"}
            return

        print(buffer)
        res_text = buffer.strip()
        calc_match = re.search(r'<calc>(.*?)</calc>', res_text, re.DOTALL)

        if calc_match:
            calc_expr = calc_match.group(1).strip()

            # 计算得分
            score = safe_eval(calc_expr)

            # 清理响应文本格式
            if res_text.startswith("```"):
                res_text = re.sub(r"^```(json)?", "", res_text).rstrip("`")

            document_results = {
                "status": "ImageScore_Done",
                "message": {"type":"文字图片内容","response": res_text.strip(), "calc_expression": calc_expr},
                "data": score,
            }

            yield document_results
            return

        else:
            yield {
                "status": "error",
                "data": {"response": res_text},
                "message": "未找到计算表达式"
            }
            return

# def main():
#     """主函数"""
#     print("=== Word文档评审系统 ===")
#
#     # 初始化系统
#     review_system = ReviewSystem()
#
#     # 第一步：输入包含评审原则的文档
#     while True:
#         doc_path = input("\n请输入包含评审原则的Word文档路径: ").strip()
#         if not doc_path:
#             continue
#
#         if not os.path.exists(doc_path):
#             print("文件不存在，请重新输入")
#             continue
#
#         try:
#             # 分割章节
#             print("\n正在分割文档章节...")
#             chapter_names = review_system.load_chapters(doc_path)
#             print(f"共分割出 {len(chapter_names)} 个章节: {', '.join(chapter_names)}")
#
#             # 查找评审原则
#             print("\n正在查找评审原则...")
#             result = review_system.find_review_principles(chapter_names)
#
#             if result:
#                 chapter_name, content, all_chapters = result
#                 print(f"找到评审原则章节: {chapter_name}")
#
#                 # 学习评审规则（传入所有章节名称）
#                 print("\n正在学习评审规则...")
#                 judge, weights = review_system.learn_review_rules(content, chapter_name, agent=review_system.agent)
#                 if judge:
#                     break
#                 else:
#                     print("评审规则学习失败，请检查文档内容")
#             else:
#                 print("未找到评审原则，请确认文档包含评分规则相关内容")
#
#         except Exception as e:
#             print(f"处理文档时出错: {str(e)}")
#             import traceback
#             traceback.print_exc()
#
#     # 第二步：评分新文档
#     while True: # weights
#         print("\n" + "=" * 50)
#         doc_path = input("请输入需要评分的Word文档的文件夹路径 (输入'quit'退出): ").strip()
#
#         if doc_path.lower() == 'quit':
#             break
#
#         if not os.path.exists(doc_path):
#             print("文件不存在，请重新输入")
#             continue
#
#         # 进行评分
#         result, score, error = review_system.score_document(doc_path, chapter_name, weights)
#
#         if "error" in result:
#             print(f"评分失败: {result['error']}")
#             continue
#
#         print(result)
#         print()
#         print("最终得分: ", score)
#
# if __name__ == "__main__":
#     main()