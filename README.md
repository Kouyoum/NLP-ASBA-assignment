# Introduction to NLP 
Assignment  2 - Intro to NLP - CentraleSupélec

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

### Summary of final model
2. A couple of paragraphs describing your final system (type of
classification model, feature representation, resources etc.)
We inspired ourselves from 

Our final system uses a BERT transformer network from the transformers library, BERT_Review, which is pretrained on a corpus of Amazon and Yelp reviews.

Before passing the review into BERT, some preprocessing is done on the reviews (functions in preprocessing.py): 
- reviews are concatenated with their associated aspect category, and target term 
-  [CLS] and [SEP] token are added to separate and enclose the "new" review

Then, the pretrained tokenizer is applied on the reviews, which ouputs the necessary input for BERT.
This includes the review, the input ids (embeddings to represent the tokens), attention masks (padded sequence, with 0s added, to have equal sized sequences), and the target (sentiments we want to predict).
We apply this in the MyDataset.py, to create the Dataset. 

After the dataloader is created, it is passed to BERT. 
From BERT's output, we keep only the pooled output, which is the feature representation, a 768 long tensor, of the [CLS] token. Because of the transformer's architecture, this token's output embedding holds already holds a lot of information about the whole review, and it suffice for the classification. 

Regarding the classifier, we pass BERT's pooled output through 2 layers:
- a dropout layer, which freezes some of the neurons: this one way of doing regularization, and can help the model learn robust relationships (generalizable to other datasets)
- a fully-connected-layer, with 3 output neurons, for the 3 sentiments that need to be predicted
Finally, a softmax is applied to obtain probabilities summing up to 1 for each review. 

Another thing worth noting in our implementation is the use of weights for each review. As we noticed that the training dataset was fairly imbalanced, we added some weighting in the cross-entropy loss. 
- Mention the weights
- Classifier: Dropout for regularization, robustness, softmax (output probabilities)


**Accuracy on the dev dataset**: XX % 


### Additional work (model with parsing)