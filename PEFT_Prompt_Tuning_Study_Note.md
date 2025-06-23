
# 📘 使用 PEFT 进行提示微调 —— 学习笔记

## ✅ 1. 项目目标

> 利用 Hugging Face 和 PEFT（参数高效微调）框架，在 `bloomz-560m` 基础上进行提示微调（Prompt Tuning），提升模型在提示式任务上的理解与生成能力。

---

## 📦 2. 使用模型与数据集

- 模型基座：`bigscience/bloomz-560m`（支持多语言/多任务）
- 微调方法：PEFT 中的 Prompt Tuning
- 数据集：
  - [`fka/awesome-chatgpt-prompts`](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts)（任务式指令）
  - [`Abirate/english_quotes`](https://huggingface.co/datasets/Abirate/english_quotes)（自然句子）

---

## 🔧 3. 核心训练配置与代码分析

### 💡 模型初始化 + Tokenizer 加载

```python
model_name = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
foundational_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
```

### 🧠 软提示参数设置

```python
NUM_VIRTUAL_TOKENS = 4
NUM_EPOCHS = 6
```

- `NUM_VIRTUAL_TOKENS`：用于提示引导的虚拟 token 数量
- `NUM_EPOCHS`：训练轮次

### 🚀 推理函数设计

```python
def get_outputs(model, inputs, max_new_tokens=100):
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.5,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id
    )
    return outputs
```

### ✅ 推理样例

```python
input_prompt = tokenizer("I want you to act as a motivational coach.", return_tensors="pt")
outputs = foundational_model.generate(...)  # 推理未微调模型效果
```

---

## 🧪 4. 训练流程（简要）

1. 使用 `PromptTuningConfig` 配置虚拟 token 微调方式
2. 使用 `get_peft_model()` 创建模型
3. 使用 Hugging Face `Trainer` 训练

---

## 📍 5. 错误排查与修复

### ❌ os.mkdir 报错

```python
# 错误写法：
os.mkdir(path, existing=True)

# 正确写法：
os.makedirs(path, exist_ok=True)
```

### ❌ generate 用法注意

- `generate()` 是推理方法，非训练用
- 使用 `tokenizer.decode()` 获取文本
- 控制重复性可调 `repetition_penalty`, `top_p`, `temperature` 等

---

## 🧠 6. 面试问题卡片

| 面试问题 | 回答建议 |
|----------|----------|
| 什么是 Prompt Tuning？ | 只训练少量虚拟 token 的参数高效微调方法 |
| num_virtual_tokens 是什么？ | 代表可学习的提示 token 数量 |
| DataCollatorForLanguageModeling 用途？ | 自动处理 padding 和动态掩码 |
| model.generate() 和 forward() 区别？ | 一个用于推理，一个用于训练 |
| 为什么选 bloomz 而非 GPT？ | bloomz 支持多语言指令，泛化性好 |
| 如何避免生成内容重复？ | 设置 `repetition_penalty` 和 top_p、temperature |

---

## 📝 7. 简历描述推荐

> 使用 Hugging Face Transformers 和 PEFT 框架，在 BLOOMZ 模型上实现 Prompt Tuning 微调。独立完成数据处理、模型配置、训练与推理流程，显著提升模型对任务式提示的理解能力，并设计多轮 prompt 实验分析微调效果。

---

## 🚀 后续建议

- 整理为 Notion / GitHub 文档
- 尝试添加 Gradio 推理 Demo
- 总结训练日志与对比结果
- 准备下一个项目：LoRA 微调 / ChatGLM 指令训练
