
**💡 Collated best practices from most popular ML research repositories - *now official guidelines at NeurIPS 2021!*** 

For reprodicibility, we offer the code for "MAML is a noisy contrastive learning" submitted for NeurIPS 2021.

## Cosine similarity analysis during training.


As in our paper where we show that the encoder is updated so that the query features are pushed towards the direction derived from the support features of the same class and pulled away from the direction built upon the support features of different classes.
We are to verify this supervised contrastiveness experimentally. 

The computed cosine similarity matrix are calcuated in `contrastivemess_visualization.ipynb` and stored in `./pickles`.
The figures are drawn in `contrastivemess_visualization_results.ipynb`.

For calculating the cosine similarity matrix, one need to download the miniImagenet dataset as described in the main page.
![image](https://github.com/IandRover/Noisy-MAML/edit/main/contrastiveness_visualization/Contrastiveness.png)
