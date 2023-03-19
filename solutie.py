import re
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier


# nltk.download('stopwords') download it only the first time you use the application
stop_words = set(stopwords.words('english'))


df = pd.read_csv('fb_sentiment.csv')

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)
df['FBPost'] = df['FBPost'].apply(preprocess)

X = df['FBPost']
label_int = []
for i in df.columns:
    if i == "Label":
        for j in df[i]:
            if j == "O":
                label_int.append(0)
            elif j == "N":
                label_int.append(2)
            elif j == "P":
                label_int.append(1)
# print(label_int)
df['LabelInt'] = label_int
y = df['LabelInt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

knn = KNeighborsClassifier(n_neighbors=35)
knn.fit(X_train_tfidf, y_train)
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred = knn.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy is:', accuracy)

