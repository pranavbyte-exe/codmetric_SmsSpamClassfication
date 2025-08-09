
import pandas as pd
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.lower().split()
    return ' '.join([word for word in words if word not in stop_words])


df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])


df['clean_msg'] = df['message'].apply(clean_text)


le = LabelEncoder()
df['label_num'] = le.fit_transform(df['label'])


cv = CountVectorizer()
X = cv.fit_transform(df['clean_msg'])
y = df['label_num']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
