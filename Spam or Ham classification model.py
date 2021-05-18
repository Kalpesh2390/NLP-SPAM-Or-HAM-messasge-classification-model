{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split \n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "import string\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "#nltk.download()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "Data = pd.read_table('C:/Users/91805/Documents/Kalpesh Mahajan/Natuaral Language Processing Class/sms.tsv',header=None,\n",
    "                     names=['Classification','Message'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Classification</th>\n",
       "      <th>Message</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Classification                                            Message\n",
       "0            ham  Go until jurong point, crazy.. Available only ...\n",
       "1            ham                      Ok lar... Joking wif u oni...\n",
       "2           spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3            ham  U dun say so early hor... U c already then say...\n",
       "4            ham  Nah I don't think he goes to usf, he lives aro..."
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ham     4825\n",
       "spam     747\n",
       "Name: Classification, dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Data['Classification'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5572, 2)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get the stopwords\n",
    "stopwords = nltk.corpus.stopwords.words('english')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_text(text):\n",
    "    print(text)\n",
    "    text = \"\".join([word.lower() for word in text if word not in string.punctuation])\n",
    "    tokens = re.split('\\W+', text)\n",
    "    text = [ps.stem(word) for word in tokens if word not in stopwords]\n",
    "    return text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'text' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-48-2ae44d9f1a9d>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[1;33m[\u001b[0m\u001b[0mword\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mlower\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mword\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mtext\u001b[0m \u001b[1;32mif\u001b[0m \u001b[0mword\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mstring\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpunctuation\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'text' is not defined"
     ]
    }
   ],
   "source": [
    "#sen = ''\n",
    "#[word.lower() for word in text if word not in string.punctuation]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "ps=nltk.PorterStemmer()\n",
    "tfidf_vect = TfidfVectorizer(analyzer=clean_text)\n",
    "X_tfidf = tfidf_vect.fit_transform(Data['Message'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<5572x8109 sparse matrix of type '<class 'numpy.float64'>'\n",
       "\twith 50185 stored elements in Compressed Sparse Row format>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_tfidf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_features = pd.DataFrame(X_tfidf.toarray())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>...</th>\n",
       "      <th>8099</th>\n",
       "      <th>8100</th>\n",
       "      <th>8101</th>\n",
       "      <th>8102</th>\n",
       "      <th>8103</th>\n",
       "      <th>8104</th>\n",
       "      <th>8105</th>\n",
       "      <th>8106</th>\n",
       "      <th>8107</th>\n",
       "      <th>8108</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 8109 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   0     1     2     3     4     5     6     7     8     9     ...  8099  \\\n",
       "0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  ...   0.0   \n",
       "1   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  ...   0.0   \n",
       "2   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  ...   0.0   \n",
       "3   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  ...   0.0   \n",
       "4   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  ...   0.0   \n",
       "\n",
       "   8100  8101  8102  8103  8104  8105  8106  8107  8108  \n",
       "0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  \n",
       "1   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  \n",
       "2   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  \n",
       "3   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  \n",
       "4   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  \n",
       "\n",
       "[5 rows x 8109 columns]"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_features.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>008704050406</th>\n",
       "      <th>0089mi</th>\n",
       "      <th>0121</th>\n",
       "      <th>01223585236</th>\n",
       "      <th>01223585334</th>\n",
       "      <th>0125698789</th>\n",
       "      <th>02</th>\n",
       "      <th>020603</th>\n",
       "      <th>...</th>\n",
       "      <th>zindgi</th>\n",
       "      <th>zoe</th>\n",
       "      <th>zogtoriu</th>\n",
       "      <th>zoom</th>\n",
       "      <th>zouk</th>\n",
       "      <th>zyada</th>\n",
       "      <th>é</th>\n",
       "      <th>ü</th>\n",
       "      <th>üll</th>\n",
       "      <th>〨ud</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5567</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5568</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.33524</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5569</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5570</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5571</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5572 rows × 8109 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "             0  008704050406  0089mi  0121  01223585236  01223585334  \\\n",
       "0     0.0  0.0           0.0     0.0   0.0          0.0          0.0   \n",
       "1     0.0  0.0           0.0     0.0   0.0          0.0          0.0   \n",
       "2     0.0  0.0           0.0     0.0   0.0          0.0          0.0   \n",
       "3     0.0  0.0           0.0     0.0   0.0          0.0          0.0   \n",
       "4     0.0  0.0           0.0     0.0   0.0          0.0          0.0   \n",
       "...   ...  ...           ...     ...   ...          ...          ...   \n",
       "5567  0.0  0.0           0.0     0.0   0.0          0.0          0.0   \n",
       "5568  0.0  0.0           0.0     0.0   0.0          0.0          0.0   \n",
       "5569  0.0  0.0           0.0     0.0   0.0          0.0          0.0   \n",
       "5570  0.0  0.0           0.0     0.0   0.0          0.0          0.0   \n",
       "5571  0.0  0.0           0.0     0.0   0.0          0.0          0.0   \n",
       "\n",
       "      0125698789   02  020603  ...  zindgi  zoe  zogtoriu  zoom  zouk  zyada  \\\n",
       "0            0.0  0.0     0.0  ...     0.0  0.0       0.0   0.0   0.0    0.0   \n",
       "1            0.0  0.0     0.0  ...     0.0  0.0       0.0   0.0   0.0    0.0   \n",
       "2            0.0  0.0     0.0  ...     0.0  0.0       0.0   0.0   0.0    0.0   \n",
       "3            0.0  0.0     0.0  ...     0.0  0.0       0.0   0.0   0.0    0.0   \n",
       "4            0.0  0.0     0.0  ...     0.0  0.0       0.0   0.0   0.0    0.0   \n",
       "...          ...  ...     ...  ...     ...  ...       ...   ...   ...    ...   \n",
       "5567         0.0  0.0     0.0  ...     0.0  0.0       0.0   0.0   0.0    0.0   \n",
       "5568         0.0  0.0     0.0  ...     0.0  0.0       0.0   0.0   0.0    0.0   \n",
       "5569         0.0  0.0     0.0  ...     0.0  0.0       0.0   0.0   0.0    0.0   \n",
       "5570         0.0  0.0     0.0  ...     0.0  0.0       0.0   0.0   0.0    0.0   \n",
       "5571         0.0  0.0     0.0  ...     0.0  0.0       0.0   0.0   0.0    0.0   \n",
       "\n",
       "        é        ü  üll  〨ud  \n",
       "0     0.0  0.00000  0.0  0.0  \n",
       "1     0.0  0.00000  0.0  0.0  \n",
       "2     0.0  0.00000  0.0  0.0  \n",
       "3     0.0  0.00000  0.0  0.0  \n",
       "4     0.0  0.00000  0.0  0.0  \n",
       "...   ...      ...  ...  ...  \n",
       "5567  0.0  0.00000  0.0  0.0  \n",
       "5568  0.0  0.33524  0.0  0.0  \n",
       "5569  0.0  0.00000  0.0  0.0  \n",
       "5570  0.0  0.00000  0.0  0.0  \n",
       "5571  0.0  0.00000  0.0  0.0  \n",
       "\n",
       "[5572 rows x 8109 columns]"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "feauture_name = tfidf_vect.get_feature_names()\n",
    "pd.DataFrame(X_tfidf.toarray(), columns = feauture_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X_features, Data['Classification'], \n",
    "                                                    train_size = 0.8, random_state = 123)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(4457, 8109)\n",
      "(4457,)\n",
      "(1115, 8109)\n",
      "(1115,)\n"
     ]
    }
   ],
   "source": [
    "print(X_train.shape)\n",
    "print(y_train.shape)\n",
    "print(X_test.shape)\n",
    "print(y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "rf = RandomForestClassifier(n_estimators=200, n_jobs = -1)\n",
    "rf_mdl = rf.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = rf_mdl.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "from sklearn.metrics import confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         ham       0.97      1.00      0.99       962\n",
      "        spam       1.00      0.81      0.90       153\n",
      "\n",
      "    accuracy                           0.97      1115\n",
      "   macro avg       0.99      0.91      0.94      1115\n",
      "weighted avg       0.97      0.97      0.97      1115\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[962,   0],\n",
       "       [ 29, 124]], dtype=int64)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_train = rf_mdl.predict(X_train)"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "print(confusion_matrix(y_train, y_pred_train))\n",
    "print(classification_report(y_train, y_pred_train))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Logistic Regression\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "LR = LogisticRegression()\n",
    "LogRegModel = LR.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = LogRegModel.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=====Classification Report for Test Data with Logistic Regression ==================\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "         ham       0.95      1.00      0.97       962\n",
      "        spam       0.98      0.67      0.79       153\n",
      "\n",
      "    accuracy                           0.95      1115\n",
      "   macro avg       0.97      0.83      0.88      1115\n",
      "weighted avg       0.95      0.95      0.95      1115\n",
      "\n",
      "=====Confusion Matrix for Test Data with Logistic Regression========================\n",
      "[[960   2]\n",
      " [ 51 102]]\n"
     ]
    }
   ],
   "source": [
    "print(\"=====Classification Report for Test Data with Logistic Regression ==================\")\n",
    "print(classification_report(y_test, y_pred))\n",
    "print(\"=====Confusion Matrix for Test Data with Logistic Regression========================\")\n",
    "print(confusion_matrix(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_train = rf_mdl.predict(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=====Classification Report for Train Data with Logistic Regression ==================\n",
      "[[3863    0]\n",
      " [   0  594]]\n",
      "=====Confusion Matrix for Train Data with Logistic Regression========================\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "         ham       1.00      1.00      1.00      3863\n",
      "        spam       1.00      1.00      1.00       594\n",
      "\n",
      "    accuracy                           1.00      4457\n",
      "   macro avg       1.00      1.00      1.00      4457\n",
      "weighted avg       1.00      1.00      1.00      4457\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"=====Classification Report for Train Data with Logistic Regression ==================\")\n",
    "print(confusion_matrix(y_train, y_pred_train))\n",
    "print(\"=====Confusion Matrix for Train Data with Logistic Regression========================\")\n",
    "print(classification_report(y_train, y_pred_train))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
