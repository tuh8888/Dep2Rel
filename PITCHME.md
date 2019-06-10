## Semi-supervised relation extraction using word vectors and syntax patterns

**Presenter:** Harrison Pielke-Lombardo

**Advisor:** Lawrence Hunter

**Institution:** University of Colorado, Anschutz Medical Campus

---
@title[Background]

@snap[north-west span-100]
## Background
@snapend

@snap[west span-50]
@ul[](false)
- Can we make enough gold standard data?
- How many relations are there?
@ulend
@snapend

@snap[east span-40]
@img[](resources/dep_example.gif)
@snapend

---
@title[Method]

@snap[north-west span-100]
## Method
@snapend

@snap[west span-50]
@ul[](false)
- Dependency path between two entities -> context path
- Combine word vectors -> context vectors
- Bootstrapping
  - Seeds
  - Clustering -> patterns
  - Context similarity
@ulend
@snapend

@snap[east span-50]
@img[](resources/algorithm.svg)
@snapend

---
@title[Results]

@snap[north-west span-100]
## Results
@snapend

@snap[west span-50]
@ul[](false)
@ulend
@snapend

@snap[east span-50]
![]()
@snapend

---
@title[Conclusion]

@snap[north-west span-100]
## Conclusion
@snapend

---
@title[Acknoledgements]

@snap[north-west span-100]
## Acknowledgements
@snapend

@snap[west span-50]
@ul[](false)
- Advisor: Lawrence Hunter
- Funding: T15 LM009451
@ulend
@snapend

@snap[east span-50]
@ul[](false)
- Email: Harrison.Pielke-Lombardo@ucdenver.edu
- GitHub: https://github.com/tuh8888
- Project: https://github.com/tuh8888/Dep2Rel
@ulend
@snapend
