# Persian Embedding

The goal of this project is to learn how to train and test an embedding model (without aiming for high testing accuracy).

## Requirements

- `gensim==3.8.3`
- `hazm`

## Training

I trained 8 models using a given Persian corpus with the following parameters:

- W2V1: Word2Vec CBOW (D = 50, min_count = 20, Window Size = 5)
- W2V2: Word2Vec CBOW (D = 50, min_count = 20, Window Size = 10)
- W2V3: Word2Vec CBOW (D = 100, min_count = 20, Window Size = 5)
- W2V4: Word2Vec CBOW (D = 100, min_count = 20, Window Size = 10)
- FT1: FastText (D = 50, min_count = 20, sg = 1)
- FT2: FastText (D = 50, min_count = 20, sg = 0)
- FT3: FastText (D = 100, min_count = 20, sg = 1)
- FT4: FastText (D = 100, min_count = 20, sg = 0)

To train the models, run the following command:

```shell
python embedding_train.py
```

Evaluation
I evaluated all models using the Analogy test. For vector similarity calculations, I used four methods:

- Distance_by_vec
- Normalized_distance_by_vec
- Most_similar_distance
- Most_similar_cosmul_distance
  
To test the models, run the following command:
```shell
python embedding_test.py
```
***
Data availability:

Train corpus: https://drive.google.com/file/d/1L1kAhb8NofQUJqLK05nj2YfPlynB_3m_/view?usp=sharing

Test corpus: https://drive.google.com/file/d/1mgxcenW9D28AfYlYwXgN-r0P3XWGz46z/view?usp=sharing
