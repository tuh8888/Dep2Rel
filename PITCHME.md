## Semi-supervised relation extraction using word vectors and syntax patterns

**Presenter:** Harrison Pielke-Lombardo

**Adviser:** Lawrence Hunter

@snap[south span-75]
@img[](assets/CUAnschutz_sl_clr.png)
@snapend

---
@title[Background]

## Challenges

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
Tokens along the dependency path between two entities
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

@snap[north]
**BioCreative VI Task 4.2**
@snapend

@snap[west]
@table[table-header](assets/train-test.csv)
@snapend

@snap[east]
@table[table-header](assets/train-test-relations.csv)
@snapend

---
@title[PCA]

@img[clean-img width=750](assets/PCA for NONE and CPR:9.png)


---
@title[Results]

@table[table-header](assets/best-of-results.csv)

---

@title[Acknowledgements]
@snap[north]
## Acknowledgements
UC, Denver Computational Bioscience Program
</br>
Funding: T15 LM009451
@snapend

@snap[south span-100]

**Contact**
</br>
Email: Harrison.Pielke-Lombardo@ucdenver.edu
</br>
GitHub: https://github.com/tuh8888
</br>
Project: https://github.com/tuh8888/Dep2Rel
@snapend