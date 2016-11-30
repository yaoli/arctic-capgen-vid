This package contains the accompanying code for the following paper:

* \[1\] Li Yao, Atousa Torabi, Kyunghyun Cho, Nicolas Ballas, Christopher Pal, Hugo Larochelle, and Aaron Courville [Describing Videos by Exploiting Temporal Structure](http://arxiv.org/abs/1502.08029). ICCV 2015.

[PDF](http://arxiv.org/pdf/1502.08029v4.pdf)

[BibTeX](https://raw.github.com/yaoli/arctic-capgen-vid/master/reference.bib)

[Video](https://youtu.be/Q6BiLAxJtXk)

[Poster with follow-up works](https://drive.google.com/open?id=0B_l-xTvrRJiESnJ0SXVMWGFvemc) that include 

* \[2\] Li Yao, Nicolas Ballas, Kyunghyun Cho, John R. Smith, Yoshua Bengio [Oracle performance for visual captioning](http://arxiv.org/abs/1511.04590).  BRITISH MACHINE VISION CONFERENCE (BMVC) 2016 (oral).

* \[3\] Nicolas Ballas, Li Yao, Chris Pal, Aaron Courville [Delving Deeper into Convolutional Networks for Learning Video Representations](http://arxiv.org/abs/1511.06432). International Conference of Learning Representations (ICLR) 2016. (conference track)

With the default setup in `config.py`, you will be able to train a model on YouTube2Text, reproducing (in fact better than) the results corresponding to the 3rd row in Table 1 where a global temporal attention model is applied on features extracted by GoogLenet. 

Note: due to the fact that video captioning research has gradually converged to using [coco-caption](https://github.com/tylin/coco-caption) as the standard toolbox for evaluation. We intergrate this into this package. In the paper, however, a different tokenization methods was used, and the results from this package is *not* strictly comparable with the one reported in the paper. 

#####Please follow the instructions below to run this package
1. Dependencies
  1. [Theano](http://deeplearning.net/software/theano/) can be easily installed by following the instructions there. Theano has its own dependencies as well. The simpliest way to install Theano is to install [Anaconda](https://store.continuum.io/cshop/anaconda/). Instead of using Theano coming with Anaconda, we suggest running `git clone git://github.com/Theano/Theano.git` to get the most recent version of Theano. 
  2. [coco-caption](https://github.com/tylin/coco-caption). Install it by simply adding it into your `$PYTHONPATH`.
  3. [Jobman](http://deeplearning.net/software/jobman/install.html). After it has been git cloned, please add it into `$PYTHONPATH` as well. 
2. Download the preprocessed version of Youtube2Text. It is a zip file that contains everything needed to train the model. Unzip it somewhere. By default, unzip will create a folder `youtube2text_iccv15` that contains 8 `pkl` files. 

[preprocessed YouTube2Text download link](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/yaoli/youtube2text_iccv15.zip)

3. Go to `common.py` and change the following two line `RAB_DATASET_BASE_PATH = '/data/lisatmp3/yaoli/datasets/'` and `RAB_EXP_PATH = '/data/lisatmp3/yaoli/exp/'` according to your specific setup. The first path is the parent dir path containing `youtube2text_iccv15` dataset folder. The second path specifies where you would like to save all the experimental results.
4. Before training the model, we suggest to test `data_engine.py` by running `python data_engine.py` without any error.
5. It is also useful to verify coco-caption evaluation pipeline works properly by running `python metrics.py` without any error.
6. Now ready to launch the training
  1. to run on cpu: `THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python train_model.py`
  2. to run on gpu: `THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_model.py`

#####Notes on running experiments
Running `train_model.py` for the first time takes much longer since Theano needs to compile for the first time lots of things and cache on disk for the future runs. You will probably see some warning messages on stdout. It is safe to ignore all of them. Both model parameters and configurations are saved (the saving path is printed out on stdout, easy to find). The most important thing to monitor is `train_valid_test.txt` in the exp output folder. It is a big table saving all metrics per validation. Please refer to `model_attention.py` line 1207 -- 1215 for actual meaning of columns. 


#####Bonus
In the paper, we never mentioned the use of uni-directional/bi-directional LSTMs to encode video representations. But this is an obvious extension. In fact, there has been some work related to it in several other recent papers following ours. So we provide codes for more sophicated encoders as well. 

#####Trouble shooting
This is a known problem in COCO evaluation script (their code) where METEOR are computed by creating another subprocess, which does not get killed automatically. As METEOR is called more and more, it eats up mem gradually. 
To fix the problem, add this line after line https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/meteor/meteor.py#L44
`self.meteor_p.kill()`

If you have any questions, drop us email at li.yao@umontreal.ca.

