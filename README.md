# persian-embedding
The goal of this project is to learn how to train and test an embedding model (not to achieve high accuracy in testing).

## Requirements

  - gensim==3.8.3
  - hazm

## Training
According to the following parameters, we have trained 8 models with a given Persian corpus.

  - W2V1 = word2vec CBOW(D = 50, min_count = 20, Window Size = 5)
  - W2V2 = word2vec CBOW(D = 50, min_count = 20, Window Size = 10)
  - W2V3 = word2vec CBOW(D = 100, min_count = 20, Window Size = 5)
  - W2V4 = word2vec CBOW(D = 100, min_count = 20, Window Size = 10)
  - FT1 = FastText (D = 50, min_count = 20, sg= 1)
  - FT2 = FastText (D = 50, min_count = 20, sg= 0)
  - FT3 = FastText (D = 100, min_count = 20, sg= 1)
  - FT4 = FastText (D = 100, min_count = 20, sg= 0)

For train the models, type:
```python
python embedding_train.py
```
 
## Evaluation
I evaluated all models using Analogy test.
In this evaluation, to calculate the similarity of the vectors, I have calculated the distance of the vectors using 4 methods. These 4 methods are as follows:

  - Distance_by_vec
  - Normalized_distance_by_vec
  - Most_similar_distance 
  - Most_similar_cosmul_distance 

For test the models, type:
```python
python embedding_test.py
```
***
You can download the Persian corpus from the following link:

Train corpus: https://drive.google.com/file/d/1L1kAhb8NofQUJqLK05nj2YfPlynB_3m_/view?usp=sharing

Test corpus: https://drive.google.com/file/d/1mgxcenW9D28AfYlYwXgN-r0P3XWGz46z/view?usp=sharing
