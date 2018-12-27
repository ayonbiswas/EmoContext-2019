# EmoContext-2019
EmoContext-A-SHARED-TASK-AT-SEMEVAL-2019

Information regarding the Task: https://www.humanizing-ai.com/emocontext.html

Dataset can be obtained by joining their linkedin group

**Proposed Architecture**
![alt text](https://raw.githubusercontent.com/nier79/EmoContext-2019/model.PNG)

**Abstract**
With the advent of text messaging applications and digital assistants in conversations, it has become inevitable for these digital agents to recognize the emotions and appropriately give a response. However, emotions are complicated to understand in a discussion due to the presence of ambiguity. A reaction generated in the human physiological state as a result of internal or external factors is called Emotions. 

In this research work, we propose to detect emotions like Happy, Sad or Angry emotions from three sentenced textual dialogue between two entities in a conversation using a Long Short-Term Memory Network-based approach branching from Deep Learning Techniques. Our proposed approach benefits from semantic and sentiment based embeddings which would be used to come up to a final solution. Evaluation of our approach has been carried out on real-world conversations.

**Problem Description**
- EmoContext deals with understanding the sentiments from textual conversions. 

- It is a hard problem in the absence of metadata like facial expressions and voice modulations.

- In EmoContext: A Shared Task at SemEval 2019, the task is that 
“given a textual dialogue, i.e., a user utterance along with two turns of context, you have to classify the emotion of user utterance as one of the emotion classes: Happy, Sad, Angry or Others​”

- The provided training dataset will contain 15K records for emotion classes, i.e., Happy, Sad and Angry; all combined and 15K records not belonging to any of the emotion categories as mentioned earlier. 

**Proposed approach**
Pre-processing was performed to remove punctuations. Emojis that were present in the conversations were replaced with the text representation of those emojis (😍 is represented as “smiling face with heart eyes”).
For Embeddings, using pre-trained models wasn’t a good idea as the language was text conversation and isn’t grammatical. So, we took the embeddings of the words and averaged them for a sentence to get the embedding of a sentence.
The embedding of the three sentences are fed to the LSTM cell, one at each timestep. The hidden state of the last cell is used as the overall context of the conversation. 
The last hidden state is then fed to a connected layer followed by softmax.  The hidden state vector has a size of 64. The dimension of the average word2vec feature is 100.
The model was implemented in Pytorch. Adam optimiser was used for training. The number of epochs run was 150.

**Results**
- Our model got an accuracy score of 67.8%.

- Our train accuracy was close to 72%.

- The submission with the highest accuarcy score is around 75%

**Challenges**
- We can get better embeddings by training them with large datasets of twitter data and other conversational datasets that are similar to our dataset. This might help us improve our model.

- Using an LSTM + CNN model might help us get better classes.

- One of the challenge is that the abbreviations used in the text language such as LOL, TTYL, ROFL etc. are difficult to understand.

- Another challenge is the use of english along with another language written in english  text. 
example - Hinglish (Hindi + English), Tanglish (Tamil + English).

**References**

[1] Gupta, Umang et al. “A Sentiment-and-Semantics-Based Approach for Emotion Detection in Textual Conversations.” CoRR abs/1707.06996 (2017): n. pag.

[2] Tang, Duyu, et al. “Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification.” ACL Anthology, 1 Jan. 1970, aclanthology.coli.uni-saarland.de/papers/P14-1146/p14-1146.

[3] Hamid Palangi, Li Deng, Yelong Shen, Jianfeng Gao, Xiaodong He, Jianshu Chen, Xinying Song, and Rabab Ward. 2016. Deep sentence embedding using long short-term memory networks: analysis and application to information retrieval. IEEE/ACM Trans. Audio, Speech and Lang. Proc. 24, 4 (April 2016), 694-707. DOI: https://doi.org/10.1109/TASLP.2016.2520371

[4] Rong, X. (n.d.). word2vec Parameter Learning Explained, 1–21.
