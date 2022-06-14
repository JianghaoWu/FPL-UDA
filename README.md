# FPL-UDA

The codes for the paper [*FPL-UDA:FILTERED PSEUDO LABEL-BASED UNSUPERVISED CROSS-MODALITY ADAPTATION FOR VESTIBULAR SCHWANNOMA SEGMENTATION*](https://ieeexplore.ieee.org/abstract/document/9761706) 

At the same time, this method has achieved good results in CrossMoDA 2021 Challenge (https://arxiv.org/pdf/2201.02831.pdf).

### Part 1: GAN-Based Data Augmentation for *G*

1. Train the *CycleGAN*

2. Train the *CUT*

3. Train the  pseudo label generator 

```sh
sh train_pseudo_label_generator.sh
```

### Part 2: Pseudo Label-Assisted Two-Stage Translation

### Part 3: **Uncertainty-Based Filtering of Pseudo Labels for** *S*

```sh
sh get_normalized_uncertainty.sh
```

```sh
sh train_final_segmentor.sh
```

### Citation

If you have any questions, please send us email jianghao.wu@88.com.

If you find our work is useful, please cite our work.

```bibtex
@inproceedings{wu2022fpl,
  title={FPL-UDA: Filtered Pseudo Label-Based Unsupervised Cross-Modality Adaptation for Vestibular Schwannoma Segmentation},
  author={Wu, Jianghao and Gu, Ran and Dong, Guiming and Wang, Guotai and Zhang, Shaoting},
  booktitle={2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2022},
  organization={IEEE}
}
@article{dorent2022crossmoda,
  title={CrossMoDA 2021 challenge: Benchmark of Cross-Modality Domain Adaptation techniques for Vestibular Schwnannoma and Cochlea Segmentation},
  author={Dorent, Reuben and Kujawa, Aaron and Ivory, Marina and Bakas, Spyridon and Rieke, Nicola and Joutard, Samuel and Glocker, Ben and Cardoso, Jorge and Modat, Marc and Batmanghelich, Kayhan and others},
  journal={arXiv preprint arXiv:2201.02831},
  year={2022}
}
```