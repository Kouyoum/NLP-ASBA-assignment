# Introduction to NLP 
# Assignment  2 - Intro to NLP - CentraleSupélec

### Assignment Links
- Description of the assignment: https://drive.google.com/file/d/1vTmsTm0b0q_6ZBzKPZ8NHlxDS4u3proo
- Data given: https://drive.google.com/file/d/1dfThUf4lCey-yZ1R4yeR8hkXE3nnOEkj
- Group to ask questions: https://groups.google.com/g/centralesupelec_nlp2021

### Name of the students
- Tom Terrier-Sarfati
- Armand Kouyoumdjian

### Sources: 
- https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
- https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/
- https://huggingface.co/activebus/BERT_Review

We inspired ourselves, and reused some code of "Curiously"'s implementation of BERT for sentiment classification (link above).

### Summary of final model

<<<<<<< HEAD
Our final system uses a BERT transformer network from the transformers library, BERT_Review, which is pretrained on a corpus of Amazon and Yelp reviews.

Before passing the review into BERT, some **preprocessing** is done on the reviews (functions in preprocessing.py): 
=======
**Accuracy on the dev dataset**: XX % 

Our final system uses a BERT transformer network from the transformers library, BERT_Review, which is pretrained on a corpus of Amazon and Yelp reviews.

Before passing the review into BERT, some preprocessing is done on the reviews (functions in preprocessing.py): 
>>>>>>> 0765d4d0a175eb731b822497a76209d00834d1f1
- reviews are concatenated with their associated aspect category, and target term 
-  [CLS] and [SEP] token are added to separate and enclose the "new" review

Then, the pretrained tokenizer is applied on the reviews, which ouputs the necessary input for BERT.
This includes the review, the input ids (embeddings to represent the tokens), attention masks (padded sequence, with 0s added, to have equal sized sequences), and the target (sentiments we want to predict).
<<<<<<< HEAD
We apply this in the MyDataset.py, to create the Dataset. After the dataloader is created, it is passed to BERT. 

From BERT's output, we keep only the pooled output, which is the **feature representation**, a 768 long tensor, of the [CLS] token. Because of the transformer's architecture, this token's output embedding holds already holds a lot of information about the whole review, and it suffice for the classification. 

Regarding the **classifier**, we pass BERT's pooled output through 2 layers:
=======
We apply this in the MyDataset.py, to create the Dataset. 

After the dataloader is created, it is passed to BERT. 
From BERT's output, we keep only the pooled output, which is the feature representation, a 768 long tensor, of the [CLS] token. Because of the transformer's architecture, this token's output embedding holds already holds a lot of information about the whole review, and it suffice for the classification. 

Regarding the classifier, we pass BERT's pooled output through 2 layers:
>>>>>>> 0765d4d0a175eb731b822497a76209d00834d1f1
- a dropout layer, which freezes some of the neurons: this one way of doing regularization, and can help the model learn robust relationships (generalizable to other datasets)
- a fully-connected-layer, with 3 output neurons, for the 3 sentiments that need to be predicted
Finally, a softmax is applied to obtain probabilities summing up to 1 for each review. 

Another thing worth noting in our implementation is the use of weights for each review. As we noticed that the training dataset was fairly imbalanced, we added some weighting in the cross-entropy loss. 
<<<<<<< HEAD

**Accuracy on the dev dataset**: around **86 %** for different random seeds. 

### Additional work including parsing

We also wanted to implement a solution using dependency parsing. To do so, we used the NLTK Stanford CoreNLP Dependency Parser (https://stanfordnlp.github.io/CoreNLP/index.html). The solution implemented and the performance achieved is described below, but we realized shortly before the deadline that this model would not be able to be ran directly from the terminal, because a connection to the Stanford server is needed in order to compute the parsings. 

#### Use of parsing
The idea behind our model was to filter the sentence with only words which have a link to the aspect given. We did this in two steps:
•	Step 1, getting words directly linked to the aspect: 
o	Take all the words that point to the aspect words
o	Take all the words that the aspect words point to 
•	Step 2, getting words with 2nd order links to the aspect: 
o	Take all the words that point to the words collected in step 1, as long as they are not nouns or determinants
o	Take all the words that the words in step 1 point to, as long as they are not nouns or determinants

#### Model
Once we had the filtered sentence with only the words of interest for the given aspect, we transformed it in order to get the BERT representation of the sentence (768x1 vector). We then fed that representation to a shallow neural network (1st layer: 768x156, 2nd layer: 156x3). We used the Adam optimizer and Cross Entropy as loss function.

#### Results:
Train set: 81% accuracy
Dev set: 79% accuracy
=======
- Mention the weights
- Classifier: Dropout for regularization, robustness, softmax (output probabilities)





### Additional work (model with parsing)
>>>>>>> 0765d4d0a175eb731b822497a76209d00834d1f1
