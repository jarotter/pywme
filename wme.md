---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Word Mover's Embedding

```python
import numpy as np
import multiprocessing as mp
from scipy.spatial import distance_matrix
from itertools import repeat, starmap
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from fuzzywuzzy import process
from gensim.models import KeyedVectors
from pyemd import emd
```

```python
class WordEmbedding:
    """ Container for using different word embeddings.
    
    For now use gensim for wmd.
    """

    def __init__(self, model_path='keyed-es/complete.kv'):
        self.model = KeyedVectors.load(model_path, mmap='r')
        self.d = self.model.vector_size
        
    def embedd(self, text):
        """
        """
        words = text.split(' ')
        vecs = []
        repl = {}
        
        for w in words:
            try:
                vecs.append(self.model.get_vector(w))
            except KeyError:
                if len(repl) < 1:
                    vocab = list(self.model.vocab.keys())
                v, s = process.extract(w, vocab, limit=1)[0]
                repl[w] = {'closest':v, 'score':s}
                vecs.append(self.model.get_vector(v))
        return {'vectors':vecs, 'replacements':repl}
                
```

```python
class DocModel:
    """ Probabilistic model for generating random documents.
    """

    def __init__(self, length=None, rdoc=None, emb=None, **kwargs):
        if length is None:
            self.sample_length = self.default_length()
        else:
            self.sample_length = length
        if rdoc is None:
            self.random_docs = self.default_docgen()
        else:
            self.random_docs = rdoc
        if emb is None:
            self.embedding = WordEmbedding()
        else:
            self.embedding = emb

    def default_length(self):
        """ Uniform.
        """
        return lambda n: np.random.randint(low=1, high=10, size=n)

    def default_docgen(self):
        """ Independent uniform documents
        """
        def f(D, lo, hi):
            return np.random.uniform(low=lo, high=hi, size=(D, self.embedding.d))

        return f
```

```python
cv = CountVectorizer()
```

```python
[cv.fit_transform(t).toarray() for t in [['hola ke ase'], ['hola kla']]]
```

```python
class WordMoversEmbedding:
    """General class to implement the Word Mover's Embedding
    with a given probabilistic model and embedding.
    """

    def __init__(self, model=None, R=None, freq=None):
        self.model = DocModel() if model is None else model
        self.R = 1024 if R is None else R
        self.freq = self.default_freq if freq is None else freq
        
    def default_freq(self, x):
        """
        """
        cv = CountVectorizer()
        a = cv.fit_transform(x).toarray()
        return a/a.sum()
        
         

    def random_features(self, w, X, Fx, D, gamma=10):
        """
        X list of documents
        Fx list of nBOW representation of documents in X
        w one random doc (list of word vectors)
        """   
        
        fomega = np.ones(D)/D
        
        def phi_w(x, fx, fomega=fomega):
            """
            x = one real doc (list of word vectors)
            fx nBOW representation of x
            """
            joint_vocabulary = np.concatenate((x,w))
            fomega = np.concatenate((np.zeros(len(x)), fomega))
            fx = np.concatenate((fx, np.zeros(len(w))))
            dmat = distance_matrix(joint_vocabulary, joint_vocabulary)
            wmd = emd(fx, fomega, dmat)
            return np.exp(-gamma*wmd)
        
        _args = zip(X, Fx)
        return list(starmap(phi_w, _args))
    
    def regularize(self, sol_list):
        """
        """
        return 1/np.sqrt(self.R)*np.array(sol_list)

    def fit_transform(self, real_docs):
        """
        Compute WME.
        """
        
    
        # 1. Generar vectores
        ## 1.1 De los documentos reales
        
        temp = list(map(self.model.embedding.embedd, real_docs))
        replacements = [t['replacements'] for t in temp]
        real_vecs = [t['vectors'] for t in temp]
        del temp
        
        ## 1.2 De documentos aleatorios
        random_lengths = self.model.sample_length(self.R)
        vmin = np.nanmin([np.nanmin(t) for t in real_vecs])
        vmax = np.nanmax([np.nanmax(t) for t in real_vecs])
        args = zip(random_lengths, repeat(vmin), repeat(vmax))
        random_vecs = starmap(self.model.random_docs, args)
        
        # 2. Calcular representaciones
        Fx = [self.freq([t])[0] for t in real_docs]
        
        # 3. Calcular WMD
        args = zip(random_vecs, repeat(real_vecs), repeat(Fx), random_lengths)
        Z = list(starmap(self.random_features, args))

        return self.regularize(Z), replacements
```

---

# Test Afore

```python
texts = pd.Series.from_csv('data/nps.csv', header=None)
texts = [t for t in texts if type(t) is str and len(t)>0]
```

```python
xs_wme = WordMoversEmbedding()
Z, repls = xs_wme.fit_transform([texts[i] for i in range(5)])
```

```python
repls
```

```python
Z.shape
```

```python

```
