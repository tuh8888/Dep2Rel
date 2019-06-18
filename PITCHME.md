## Semi-supervised relation extraction using word vectors and syntax patterns

**Presenter:** Harrison Pielke-Lombardo

**Adviser:** Lawrence Hunter

@snap[south span-75]
@img[](assets/CUAnschutz_sl_clr.png)
@snapend


---
@title[Background]

## Some observations

- There's not many gold standard relation annotations
- We don't know how many different relations are in the scientific literature
- It's difficult to construct syntactic/semantic patterns for finding relations in 
scientific literature

---
@title[Method]

## Semi-Supervised Method
- Few seed sentences required
- Bootstrapping

---
@title[Context]

@snap[north-west span-50 text-center]
**Context Path:** 
</br>
Word vectors along the dependency path between two entities
@snapend

@snap[north-east span-50 text-center]
**Context Vector:** 
</br>
Combination of the word vectors along the context path
@snapend

@snap[south span-100]
@img[clean-img](/assets/dep_example.svg) 
@size[.6em](Depencency path for "Little is known about genetic factors affecting intraocular pressure [IOP] in mice and other mammals.")
@snapend

---
@title[Relation Extraction]

@img[clean-img height=500](assets/algorithm.svg)


---
@title[Table 1]

  | Group |  Num Sentences |
| -----------|------------|
|  Train |      18020 |
|   Test | 15764       |

| Relation Group | Num Sentences |
|---|---|
| CPR:3 | 762 |
| CPR:4 | 2251 |
| CPR:5 | 172  |
| CPR:6 | 232 |
| CPR:9 | 722 |
| NONE | 13881 |

---

@title[Context Path Lengths]

@img[border=none width=1000 height=400](assets/Training Context Path Lengths.svg) 

---
@title[Results]

@snap[north span-50]
#### Results
@snapend

@snap[west span-50]
@img[clean-img](assets/pca-all.png)
@snapend

@snap[east span-50]
@img[clean-img](assets/metrics.svg)
@snapend

---

|   | Dep2Rel 20 | Dep2Rel All | State of the Art |
|---|---|---|---|
| Precision | | 0.3283 | 0.4544 
| Recall | | 0.6195 | 0.5387
| F1-Score | | 0.3128 | 0.3729      

---
@title[Conclusion]

@snap[north-west span-100]
## Conclusion
@snapend

---
@title[Acknoledgements]

## Acknowledgements

UCD Computational Bioscience Program

Funding: T15 LM009451

Email: Harrison.Pielke-Lombardo@ucdenver.edu

GitHub: https://github.com/tuh8888

Project: https://github.com/tuh8888/Dep2Rel
