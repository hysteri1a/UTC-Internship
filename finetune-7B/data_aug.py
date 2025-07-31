import json
import random
import re
import torch
from nlpaug.augmenter.word import ContextualWordEmbsAug, SynonymAug
from textattack.augmentation import EasyDataAugmenter
from googletrans import Translator
import synonyms
import jieba
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class AdvancedDialogAugmenter:
    def __init__(self):
        self.ctx_aug = ContextualWordEmbsAug(
            model_path='bert-base-chinese',
            action="substitute",
            device='cuda' if torch.cuda.is_available() else 'cpu',
            aug_p=0.3
        )

        self.translator = Translator()

        self.eda_aug = EasyDataAugmenter(
            pct_words_to_swap=0.3,
            transformations_per_example=3
        )

        self.keyword_aug = SynonymAug(aug_src='wordnet', lang='cmn')

    def _preserve_special_tokens(self, text):
        special_tokens = re.findall(r'\[图片\]|http[s]?://\S+', text)
        placeholder = '||SPECIAL||'
        masked_text = re.sub(r'(\[图片\]|http[s]?://\S+)', placeholder, text)
        return masked_text, special_tokens

    def _restore_special_tokens(self, masked_text, special_tokens):
        for token in special_tokens:
            masked_text = str(masked_text).replace('||SPECIAL||', token, 1)
        return masked_text

    def back_translate(self, text):
        try:
            translated = self.translator.translate(text, src='zh-cn', dest='en').text
            translated = self.translator.translate(translated, src='en', dest='zh-cn').text
            translated = self.translator.translate(translated, src='zh-cn', dest='en').text
            translated = self.translator.translate(translated, src='en', dest='zh-cn').text
            return translated
        except Exception as e:
            print(f"回译失败: {e}")
            return text

    def is_adjective(self, word):
        """检查词是否为形容词"""
        pos = nltk.pos_tag([word])
        return pos[0][1] in {'JJ', 'JJR', 'JJS'}  # 形容词的词性标记

    def augment_input(self, text, num_aug=3):
        """仅增强输入文本"""
        if text == "客户已接入，会话开始":
            return [text]  # 不增强特定输入

        masked_text, special_tokens = self._preserve_special_tokens(text)
        words = masked_text.split()
        augmented_texts = set()

        for _ in range(num_aug):
            new_words = []
            for word in words:
                # 仅对非形容词进行增强
                if not self.is_adjective(word):
                    # 选择增强策略
                    strategy = random.choice([
                        lambda w: self.ctx_aug.augment(w)[0],
                        lambda w: self.back_translate(w),
                        lambda w: self.keyword_aug.augment(w)[0],
                        lambda w: ' '.join(self.eda_aug.augment(w)[0])
                    ])
                    new_word = strategy(word)
                    new_words.append(new_word)
                else:
                    new_words.append(word)  # 保持形容词不变

            augmented_text = ' '.join(new_words)
            augmented_text = self._restore_special_tokens(augmented_text, special_tokens)
            augmented_texts.add(augmented_text)

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

    with open('output.json', encoding="UTF-8") as f:
        data = json.load(f)

    augmented_data = []
    for sample in data:
        augmented = augmenter.augment_conversation(sample, num_variants=3)
        augmented_data.extend(augmented)

    with open('augmented_data.json', 'w') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
