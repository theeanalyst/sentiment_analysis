


import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from huggingface_hub import notebook_login


notebook_login()

model_name = "Shiko07/tuned_test_trainer-bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return {0: "Negative", 1: "Neutral", 2: "Positive"}[predicted_class]

custom_css = """
    .gradio {
        background-color: #0074D9;  /* Change background color to blue */
    }
"""

# predict_sentiment function
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, label="Enter your text:"),
    outputs="text",
    title="Marrakech Sentiment Analysis App",
    description="An app for sentiment analysis for Tweet posts on covid 19 vaccine.",
    css=custom_css,
    examples = [ ["Vaccine misinformation is harmful."],
                 ["I'm hopeful about the vaccine."],
                 ["Second dose excitement."],
                 ["I'm worried about vaccine side effects."],
                 ["Vaccine distribution updates are available."],
                 ["Vaccine distribution is too slow."],
                 ["I'm gathering information about the vaccine."]
    ]
)
interface.launch()
