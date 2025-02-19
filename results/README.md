# Test results

## GLISTER replication tests

DNA. Test accuracies of: (1)GLISTER-ONLINE models (internal optimization), (2) shallow net trained on core-sets of
10%, 30% and 50% of the train data set generated by the core-set selection methods, (3) shallow net train
on random subsets of the train data, (4) shallow net trained on full train sample:

<img src="https://github.com/rolkon/ML2021-Coreset-Selection/blob/43ac16c073fa3226670abd6f9990e38d3ba5944b/results/glister_replication/glister_dna.png" width="512">


Digits, GlisterRegular. Test accuracies of: (1)GLISTER-ONLINE models (internal optimization), (2) shallow net trained on core-sets of
10%, 30% and 50% of the train data set generated by the core-set selection methods, (3) shallow net train on
random subsets of the train data, (4) shallow net trained on full train sample:

<img src="https://github.com/rolkon/ML2021-Coreset-Selection/blob/43ac16c073fa3226670abd6f9990e38d3ba5944b/results/glister_replication/glister_digits_reg.png" width="512">

Digits, GlisterImage. Test accuracies of: (1)GLISTER-ONLINE models (internal optimization), (2) shallow net trained on core-sets of
10%, 30% and 50% of the train data set generated by the core-set selection methods, (3) shallow net train on
random subsets of the train data, (4) shallow net trained on full train sample:

<img src="https://github.com/rolkon/ML2021-Coreset-Selection/blob/43ac16c073fa3226670abd6f9990e38d3ba5944b/results/glister_replication/glister_digitis_image.png" width="512">

## K-Center replication tests

Test  accuracies  of  Resnet18,  trained  on  core-sets  of 10%, 30% and 50% of the total data set,
generated by Greedy K-Centers and random sampling:

<img src="https://github.com/rolkon/ML2021-Coreset-Selection/blob/b2614099ddf708446f7c11a6a667ff131ff4e594/results/k_center_replication/accuracy_over_setsize.png" width="512">

## Model generalization tests

Test accuracies of neural networks after 100. epoch, trained on core-sets of 10%, 30% and 50% of the total data set, generated by the core-set selection methods and plotted over the core-set selection methods:

<img src="https://github.com/rolkon/ML2021-Coreset-Selection/blob/b2614099ddf708446f7c11a6a667ff131ff4e594/results/models_generalization/generalization_ability_16x10.png" width="512">

Test accuracies of neural networks after 100. epoch, trained on core-sets of 10%, 30% and 50% of the total data set, generated by the core-set selection method and plotted over the used neural networks:

<img src="https://github.com/rolkon/ML2021-Coreset-Selection/blob/b2614099ddf708446f7c11a6a667ff131ff4e594/results/models_generalization/generalization_accuracy_comparison_16x10.png" width="512">
