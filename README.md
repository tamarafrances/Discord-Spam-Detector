# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Discord Spam Detector
#### Capstone Project
#### Tamara Frances

<br>

### Background

#### What is Discord?
Discord is a social media platform with over 150 million active users. It consists of public and private groups that users with common interests can join, and you can use both public and private messaging and voice chat to connect with others ([source](https://en.wikipedia.org/wiki/Discord)).

#### Spam on Discord
In November 2021, in response to an increase of spam on the platform, Discord’s policy and safety team put out a detailed report on steps being taken to fight it ([source](https://discord.com/blog/how-discord-is-fighting-spam)). Some spam messages are fairly “innocent” like getting people to join a discord group, whereas others are more dangerous, like tricking users into connecting their crypto wallets to fake websites to buy NFTs that don't exist.

According to Discord, spam generally comes from 3 areas:
- Generated Accounts aka bots that make up the "bulk majority" of Discord spam.
- Compromised Accounts that cause "the highest user-impact spam" because they're using 'real' accounts.
- Human Operations Accounts that are made and operated by real people.


### Problem Statement

My goal was to create a classification model that determined whether or not a Discord message was spam in order to reduce user interaction with potentially malicious messages.

### Data Collection
##### Data folder
There is no easy way to pull messages sent through Discord from their website or app. Therefore, I manually copied and pasted around 1,000 messages from various Discord groups I am part of. I also manually categorized the messages into 'spam' or 'non-spam'.
<br>

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**text**|*object*|data_for_capstone.xlsx|Text messages sent on Discord|
|**spam**|*object*|data_for_capstone.xlsx|Classification of spam or non-spam|
<br><br>

---

<br>

### Exploratory Data Analysis
##### EDA folder
This dataset has about 1,000 messages and comes from various discord groups I am a part of. They’re generally focused on reselling items, like sneaker betting, investing in sports cards, sports betting, etc.
- Around 86% of messages in the dataset were non-spam and 14% were spam
- Most common words with stop words removed don’t reveal much information
- The average word count of non spam was 14 words vs. 16 words for spam
- The average text length of non spam messages was 72 characters vs. 93 characters for spam
- Polarity and subjectivity were generally the same - this doesn’t surprise me because if the goal is to get spam through the filters

The difference between bigrams and trigrams for non-spam and spam provided some interesting information. The bigrams and trigrams for spam were overwhelmingly cryptocurrency focused vs. more variation in the non-spam bigrams and trigrams.

<br>

---

<br>

### Modeling
##### Preprocessing and Modeling folder
I created and finetuned three models: logistic regression, random forest classifier, and multinomial naive bayes. 

All three models beat the baseline model’s accuracy score of 0.86. Ultimately, I chose Multinomial Naive Bayes as my best model. It edged out the other two models when it came to f1-score and had much better performance in another metric I was focused on, which was recall. I also tried some balancing techniques due to the class imbalance of the dataset I was using, but these techniques didn't surpass scores on the imbalanced data.

In the case of discord spam in particular, I personally paid attention to which model minimized false negatives the most (recall), because I’d rather over index on non-spam potentially getting flagged as spam vs spam not getting flagged because falling to the scams in the spam can be so detrimental.

Multinomial Naive Bayes (imbalanced data):
- Accuracy score: 0.92
- **F-1 score: 0.75**
- **Recall: 0.81**

Random Forest Classifier (imbalanced data):
- Accuracy score: 0.93
- F-1 score: 0.74
- Recall: 0.68

Logistic Regression (imbalanced data):
- Accuracy score: 0.92
- F-1 score: 0.6
- Recall: 0.45

<br>

##### Evaluation folder
In the "Evaluation" notebook, I took a look at the model's misclassifications to get a better sense of most common misclassified words and view the specific messages that were misclassified. It did not surprise me that most of the misclassified non-spam messages involved words related to NFTs and cryptocurrency since those topics were the focus of most of the spam messages in the dataset.

<br>

##### Streamlit folder
I built a streamlit app to highlight some of the exploratory data analysis I conducted, the different models I tested and their respective success metric results, some of the code I used in this process, and the specific classifications that my best performing model got wrong. The front page also has a demo to show whether or not my model would flag a message as spam. The streamlit app can be found ([here](link)).
<br>

---

<br>

### Future Considerations

I think these models could be improved even further if extra data became accessible. Data such as age of the discord account sending the message, whether someone is friends or shares groups with the person they’re direct messaging, and the average rate of messaging could contribute to more accurate predictions.