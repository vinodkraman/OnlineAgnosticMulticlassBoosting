# Online Agnostic Multiclass Boosting
A Python implementation of online agnostic multiclass boosting using online learners from the River Package.

The algorithms are described and theoretically analayzed in the following work. 
```
Vinod Raman and Ambuj Tewari. 
Online Agnostic Multiclass Boosting.
In Advances in Neural Information Processing Systems, 2022.
```

If you use this code in your paper, please cite the above work. Although it is based on this we cannot guarantee that the algorithm will work exactly, or even produce the same output, as any of these implementations.

For our weak learners, we used the HoeffdingTreeClassifier from the [River Package](https://github.com/online-ml/river).

```
Montiel, Jacob and Halford, Max and Mastelini, Saulo Martiello
and Bolmier, Geoffrey and Sourty, Raphael and Vaysse, Robin and Zouitine, Adil
and Gomes, Heitor Murilo and Read, Jesse and Abdessalem, Talel and others
River: machine learning for streaming data in Python
2021

```
