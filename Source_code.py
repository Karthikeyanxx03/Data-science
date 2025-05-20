from google.colab import files
uploaded = files.upload()

import pandas as pd

df = pd.read_csv('Fake News Detection Dataset.csv')


# View structure
print(df.head())
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
df.info()
df.describe(include='all')



print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())




# Example: Drop rows with missing values
df.dropna(inplace=True)



import seaborn as sns
import matplotlib.pyplot as plt

# Class distribution (assuming 'label' column: 1 = fake, 0 = real)
sns.countplot(data=df, x='label')
plt.title('Distribution of Fake and Real News')
plt.show()


X = df['text']  # Assuming 'text' is the news content
y = df['label']  # 0 = real, 1 = fake


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_vec, y_train)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

sample_news = ["This is a sample news article text..."]
sample_vec = vectorizer.transform(sample_news)
pred = model.predict(sample_vec)

print("Prediction (1 = Fake, 0 = Real):", pred[0])


import gradio as gr

def predict_fake_news(news_text):
    vec = vectorizer.transform([news_text])
    prediction = model.predict(vec)[0]
    return "Fake News" if prediction == 1 else "Real News"

gr.Interface(
    fn=predict_fake_news,
    inputs=gr.Textbox(lines=10, placeholder="Paste news article here..."),
    outputs="text",
    title="📰 Fake News Detector",
    description="Paste a news article and get prediction: Real or Fake"
).launch()


!pip install gradio



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)


import gradio as gr

def predict_news(news_text):
    vec = vectorizer.transform([news_text])
    pred = model.predict(vec)[0]
    return "Fake News" if pred == 1 else "Real News"


gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=10, placeholder="Paste news article here..."),
    outputs="text",
    title="📰 Fake News Detector",
    description="Enter a news article below to check if it's real or fake."
).launch(share=True)  # share=True creates a public URL
