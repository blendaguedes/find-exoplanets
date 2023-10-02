<h1> Discovering Exoplanets </h1>

This project was made to complete the credits for the discipline of Neural Networks in the master's program in Applied Computing at PPGIA/UFRPE. 
As I had to present an experiment and an article, this repository will have all the experimental code, it will also present most of what is valuable to share with the Git community about the article.

<h3>Introduction</h3>

In 2009 the Kepler Space Observatory was launched by NASA, with the aim to answer the question: ”How frequent are other Earths in our galaxy?” [2]. The observatory is on its second mission since 2014, and had labeled more than 10, 000 possible exoplanets [3].

Kepler scanned 1,284 new exoplanets in 2016. As in 2017, there were more than 3,000 exoplanets confirmed in total (using all detection meth- ods, including ground-based ones). The observatory still works and keep on searching for other exoplanets [3].

Our goal with this project is to use the data collected by the Kepler Space Observatory during all these years and build a Neural Networks model solution that, given the characteristics collected by Kepler, is feasible to identify exoplanets.

<h3>Multilayer Perceptron</h3>

Multilayer Perceptron (MLP) type RNA is a type of supervised net- work that uses a set of perceptron and direct feed. Its basic structure is called in three points: each neuron has a function of activation nonlinear and differentiable; there are one or more layers (see Figure 2) intermediate or hidden for both input and output layers; and has a high number of con- nections where their amplitudes are calculated by the weight of each of the connections.
The input and output layers work similarly to a perceptron, this time having their connections pointed at more than one neuron. The intermediate layers work with a network of perceptrons, having exits of a neuron in any layer with the entrance of the neurons of the following layer. Finally the output layer delivers the result of the network.
The Backpropagation technique is used to adjust the weights of the net- work connections. This technique uses with parameter the network prediction error propagating this error retroactively, so it receives this name.
The weight adjustment is done using the Generalized Delta Rule which is an adaptation of the Delta Rule. [7], where the network weights are suitable through the descending gradient:

<h3>Methodology</h3>

The machine used was a MacBook Pro 2017, with a Mac OS Mojave operating system, 500Gb SSD, RAM 16Gb, processor of 2.9 GHz Quad-Core Intel Core i7.
The programming language used was Python 3 and PyTorch [9] library was used to built the Neural Networks.

<h4>Database</h4>

The database was taken by the Kaggle [3] platform. And the data cleaning was based in the work referenced in [10]. After that we have 7803 samples. It was divided in three parts where 70% was to training, 15% to validation and 15% to testing. The features were normalized using min-max scaling.
Finally we had 37 different characteristics to use as a input and a binary output.

<h4>Neural Network Configuration</h4>

Five different Neural Network were built, Table 1 has all the Neural Networks configurations:

<img WIDTH="600" alt="portfolio_view" src="https://github.com/blendaguedes/find-exoplanets/blob/main/article/pics/conf.png">

All the Neural Networks used had on their output layer a Sigmoid Function, a threshold of 0.5 was used to the final classification. 
The images bellow show the representation of the three different architectures used on this project:


<img height="300" alt="portfolio_view" src="https://github.com/blendaguedes/find-exoplanets/blob/main/article/pics/perceptron.png" title="Figure 1: Perceptron Topology">
Figure 1: Perceptron Topology

</br>

<img height="300" alt="portfolio_view" src="https://github.com/blendaguedes/find-exoplanets/blob/main/article/pics/NN2.png" title="Figure 2: MLP with 2 Hidden Layers Topology">

Figure 2: MLP with 2 Hidden Layers Topology


<img height="300" alt="portfolio_view" src="https://github.com/blendaguedes/find-exoplanets/blob/main/article/pics/NN3.png" title="Figure 3: MLP with 3 Hidden Layers Topology">

Figure 3: MLP with 3 Hidden Layers Topology

The training was made using the Adam as optimizer, using 0.01 as the learning rate and using Mean Squared Error as loss function. Two different training were made, one with 100 epochs and another with 50 epochs. All the other configurations of the training remained the same. We have the total of 10 experiments.

<h3>Results</h3>

Here we will call the experiment with 100 epochs Experiment A, and the experiment with 50 epochs Experiment B.
Above you will find in Table 2 the results using different metrics, described in [11], of the test data after the training. After that he validation vs training loss graph during training of Experiment A are presented.

<img WIDTH="600" alt="portfolio_view" src="https://github.com/blendaguedes/find-exoplanets/blob/main/article/pics/results.png">

<img height="300" alt="portfolio_view" src="https://github.com/blendaguedes/find-exoplanets/blob/main/article/pics/a_all_train.png" title="Figure 4: Training loss during Experiment A">
Figure 4: Training loss during Experiment A

<img height="300" alt="portfolio_view" src="https://github.com/blendaguedes/find-exoplanets/blob/main/article/pics/all_train.png" title="Figure 5: Training loss during Experiment A">
Figure 5: Training loss during Experiment B

<h3>Conclusion</h3>
As we could see that the results of the experiments had all very close values of performance, talking the conclusion that least complex Neural Net- work could bring us a very similar result to the ones with more layers. What makes us believe that a Perceptron would perform a good analysis algorithm to the problem we have.

<h3>References</h3>

[1] L. du Buisson; N. Sivanandam; Bruce A. Bassett; M. Smith, Machine learning classification of sdss transient survey images, Monthly Notices of the Royal Astronomical Society, 2015, 454 (2): 2026-2038 (Aug. 2015). doi:10.1093/mnras/stv2041.

[2] M. Johnson, Kepler and k2, (August 2018).

[3] NASA, Kepler exoplanet search results,
    https://www.kaggle.com/nasa/kepler-exoplanet-search-results/metadata
(2017).

[4] A. Braga, T. Ludemir, A. Carvalho, Redes Neurais Artificiais - Teoria e aplica ̧ca ̃o, 1st Edition, Livros t ́ecnicos e cient ́ıficos editora, 2000.

[5] C. Gershenson, Artificial neural networks for beginners (2003). arXiv:cs/0308031. URL https://arxiv.org/abs/cs/0308031

[6] D. V. Leal, Estudo sobre evasa ̃o escolar via aprendizado de m ́aquina (2017).

[7] C. M. Bishop, Neural Networks for Pattern Recognition, Oxford Uni- versity Press, Inc., New York, NY, USA, 1995.

[8] S. Haykin, Neural Networks and Learning Machines, 3rd Edition, Pear- son Education, Inc., 1999.

[9] A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, A. Lerer, Automatic differentiation in pytorch (2017).

[10] I. Araujo, Using machine learning to find exoplanets with nasa’s data, (November 2020). https://towardsdatascience.com/using-machine-learning-to-find-exoplanets-with-nasas-dataset-bb818515e3b3

[11] A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, A. Lerer, Automatic differentiation in pytorch, International Journal of Data Mining Knowledge Management Process (IJDKP) 5 (2017).
URL https://www.researchgate.net/profile/MohammadH ossin/publication/275224157ARe 

