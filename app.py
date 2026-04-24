# 电影评论情感分类系统
# 模型：英文情感分类模型
# 功能：输入英文评论 → 输出 positive / negative

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 加载模型
def load_model():
    model = pipeline(
        task=Tasks.text_classification,
        model="damo/nlp_bert_sentiment-analysis_english-base"
    )
    return model

# 情感预测
def predict_sentiment(text, model):
    result = model(text)
    label = result["labels"][0].lower()
    score = round(result["scores"][0], 4)
    return label, score

# 主程序
if __name__ == "__main__":
    model = load_model()
    reviews = [
        "This movie is fantastic!",
        "The plot is boring",
        "I really like this film",
        "The movie is not bad.",
        "The acting was great, but the story was too long."
    ]

    print("=== 电影评论情感分类结果 ===")
    for review in reviews:
        label, score = predict_sentiment(review, model)
        print(f"{review} -> {label} (置信度: {score})")