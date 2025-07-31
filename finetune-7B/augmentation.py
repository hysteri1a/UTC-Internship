import json
import random
import re
import torch
from nlpaug.augmenter.word import ContextualWordEmbsAug, SynonymAug
from textattack.augmentation import EasyDataAugmenter
from googletrans import Translator
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class AdvancedDialogAugmenter:
    def __init__(self, similarity_threshold=0.86):
        self.ctx_aug = ContextualWordEmbsAug(
            model_path='bert-base-chinese',
            action="substitute",
            device='cuda' if torch.cuda.is_available() else 'cpu',
            aug_p=0.3
        )

        self.translator = Translator()
        self.eda_aug = EasyDataAugmenter(pct_words_to_swap=0.3, transformations_per_example=3)
        self.keyword_aug = SynonymAug(aug_src='wordnet', lang='cmn')

        self.model = SentenceTransformer('thenlper/gte-large-zh', cache_folder=r'./similarity')
        self.similarity_threshold = similarity_threshold

    def _preserve_special_tokens(self, text):
        special_tokens = re.findall(r'\[图片\]|http[s]?://\S+|(\|\|SPECIAL\|\|)', text)
        placeholder = '||SPECIAL||'
        masked_text = re.sub(r'(\[图片\]|http[s]?://\S+)', placeholder, text)
        return masked_text, special_tokens

    def _restore_special_tokens(self, text, special_tokens):
        for token in special_tokens:
            text = text.replace('||SPECIAL||', token, 1)
        return text

    def back_translate(self, text):
        try:
            special_tokens = re.findall(r'\|\|SPECIAL\|\|', text)
            text_without_placeholders = text.replace('||SPECIAL||', '')

            # 简化回译次数并添加超时处理
            translated = self.translator.translate(text_without_placeholders, src='zh-cn', dest='en').text
            translated = self.translator.translate(translated, src='en', dest='zh-cn').text

            # 恢复特殊标记
            for token in special_tokens:
                translated += ' ' + token

            return translated.strip()
        except AttributeError:
            # 当返回None时直接返回原文本
            return text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def is_adjective(self, word):
        pos = nltk.pos_tag([word])
        return pos[0][1] in {'JJ', 'JJR', 'JJS'}

    def augment_with_similarity_check(self, original_text, words, special_tokens):
        original_embedding = self.model.encode(original_text)
        augmented_texts = set()

        attempts = 0
        max_attempts = 50
        print(original_text)
        while len(augmented_texts) < 1 and attempts < max_attempts:
            new_words = []
            for word in words:
                if not self.is_adjective(word):
                    # 动态选择策略（排除单次词时的EDA）
                    if len(words) == 1:
                        strategies = [
                            lambda w: self.ctx_aug.augment(w)[0],
                            lambda w: self.back_translate(w),
                            lambda w: self.keyword_aug.augment(w)[0]
                        ]
                    else:
                        strategies = [
                            lambda w: self.ctx_aug.augment(w)[0],
                            lambda w: self.back_translate(w),
                            lambda w: self.keyword_aug.augment(w)[0],
                            lambda w: ' '.join(self.eda_aug.augment(w)[0] if len(w.split()) > 1 else w)
                        ]

                    strategy = random.choice(strategies)
                    new_word = strategy(word)
                    new_words.append(new_word)
                else:
                    new_words.append(word)

            augmented_text = ' '.join(new_words)
            augmented_text = self._restore_special_tokens(augmented_text, special_tokens)
            augmented_embedding = self.model.encode(augmented_text)
            similarity = cos_sim(original_embedding, augmented_embedding).item()

            if similarity >= self.similarity_threshold:
                augmented_texts.add(augmented_text)

            attempts += 1

        return list(augmented_texts)

    def augment_input(self, text, num_aug=3):
        if text == "客户已接入，会话开始":
            return [text]

        masked_text, special_tokens = self._preserve_special_tokens(text)
        words = masked_text.split()
        augmented_texts = self.augment_with_similarity_check(text, words, special_tokens)

        return list(augmented_texts)[:num_aug]

    def augment_conversation(self, dialog, num_variants=5):
        augmented_dialogs = []

        for _ in range(num_variants):
            new_dialog = {'conversation': []}
            for turn in dialog['conversation']:
                new_turn = {}

                if 'system' in turn:
                    new_turn['system'] = random.choice([
                        "作为专业客服，我的职责是解决客户问题",
                        "您好，这里是客户服务中心",
                        "欢迎咨询，请描述您的需求"
                    ])

                if 'input' in turn:
                    inputs = self.augment_input(turn['input'])
                    new_turn['input'] = random.choice(inputs + [turn['input']])

                if 'output' in turn:
                    new_turn['output'] = turn['output']

                new_dialog['conversation'].append(new_turn)

            augmented_dialogs.append(new_dialog)

        return augmented_dialogs


if __name__ == "__main__":
    augmenter = AdvancedDialogAugmenter()

    with open('standard-data.json', encoding="UTF-8") as f:
        data = json.load(f)

    augmented_data = []
    for sample in data:
        augmented = augmenter.augment_conversation(sample, num_variants=3)
        augmented_data.extend(augmented)

    with open('augmented_data.json', 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
