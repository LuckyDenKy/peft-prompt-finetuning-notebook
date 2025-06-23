import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 模型路径（可以是微调后的本地路径）
model_name = "path/to/your/peft/model"  # 替换为你的本地模型路径
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
foundational_model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m",trust_remote_code=True)
loaded_model_prompt = PeftModel.from_pretrained(
    foundational_model,
    "peft_outputs_prompt",
    device_map="auto",
    is_trainable=False,
)

# 推理函数
def get_outputs(model,inputs,max_new_tokens=100):
    outputs = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        # temperature = 2.0,
        # top_P = 0.95,
        # do_sample=True,
        repetition_penalty=1.5,  # Avoid repetition
        early_stopping=True,  # The model can stop before reach the max_length
        eos_token_id=tokenizer.eos_token_id
    )
    return outputs
    
def generate_text(prompt, max_new_tokens=100):
    # 对输入 prompt 编码
    inputs = tokenizer(prompt, return_tensors="pt")
    # 使用已有的推理函数
    outputs = get_outputs(loaded_model_prompt, inputs, max_new_tokens=max_new_tokens)
    # 解码并返回结果
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Gradio 界面
iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=4, placeholder="输入你的提示..."),
    outputs="text",
    title="🧠 Prompt Tuning Demo",
    description="试试你的 prompt 会得到怎样的模型输出？（模型：PEFT + BLOOMZ）"
)

iface.launch()
