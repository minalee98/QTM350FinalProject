# Detecting Accuracy of the Transcribe Service using Different Songs and Audio

Hello Everybody! Welcome to the blog that will walking you through our project in its entirety. If you are reading this, we are glad you decided to embark on this journey with us on our analysis of the Transcribe AWS ML API.

## Using AWS Transcribe to detect song lyrics

How many times have you been listening to a song and wondered what the artist is even saying? 

The answer is too many times. Whether it be because of their accent, the background noise being too loud, or simply that they are singing so fast, it can be hard to pick-up on the lyrics and sing your heart out with them. This forces you to search the lyrics on google. But, how do we know if these lyrics are accurate?

AWS Transcribe is an Amazon Machine Learning API that is supposed to have the answers for these issues. Our project will be centered around using the Transcribe ML API to transcribe different songs and compare the lyrics that it outputs to the lyrics that anyone can search for on google. We want to understand just how accurate the Transcribe service is and how accurate it can be given songs with different speeds, levels of sound regarding background instruments, and artist accents. 

## Our Hypothesis

We believe that the Transcribe service will generate lyrics with a high-level of accuracy for straight-forward songs at regular speeds. However, we believe that increasing the speed of the song will reduce the accuracy of the transcription along with other variables such as: artist accent, and background noise. Often, in order to keep a song in rythm, artists may pronounce words slightly differently in order to keep the song flowing properly. It will be interesting to see if the accuracy of the transcription is effected at all by the phonetics of the artist. Background noice is also a factor. Often in country or rock songs there are many instruments being played while the lead artist is singing and this could potentially effect the accuracy of the transcription. Our analysis will answer all of these questions and show whether AWS Transcribe is useful for finding the lyrics of your favorite songs. 

## Our Data

The data we will collect for our analysis will be comprised of many songs from different genres such as: rap, rock, country, and pop. For each song we will adjust the audio to have three different play-back speeds of: 0.5x, 1x, and 2x. This will allow us to see how the speed of the audio effects the transcription. The different genres of songs will allow us to test different variables such as: phonetics of the artist, artist accent, and background music. Rap songs tend to have a lot of background vocals (ad-libs), rock songs tend to have a lot of background instruments playing, country songs have different accents and phonetics, and pop songs will be our base genre that, we believe, will produce the most accurate transcriptions. 

Now that you have a general understanding of our project, let me explain how AWS Transcribe works. 

## What Is AWS Transcribe?

As per the [AWS Documenation](https://docs.aws.amazon.com/transcribe/?id=docs_gateway), AWS Transcribe is used to  provide transcription services for your audio files. It uses advanced machine learning technologies to recognize spoken words and transcribe them into text. It can transcribe audio files from different languages and also different accents. The latter proved to be an important aspect for our analysis. 

## Getting Started

Now that we have some background knowledge and a basic understanding of the Transcribe API, lets call the service using SageMaker and the CLI to see how it operates. 



### Setting up an IAM Role

In order to use this API within Sagemaker, we will need to update the Role we have been using to control Sagemaker permissions. Recall, when you created your Sagemaker instance, one of the steps was creating a new IAM Role.



## Speech Input

As the documentation says, the first step to transcribing an audio file is to store that file in an S3 Bucket. S3 is another service in AWS that you can learn more about through its [documentation](https://docs.aws.amazon.com/AmazonS3/latest/gsg/GetStartedWithS3.html). It is used for internet storage and can be used in coinjuction with other AWS services. 

##Using the AWS Transcribe from the SageMaker

To find the detailed walk through of how we save the mp3 file into the bucket S3 and operate AWS Transcribe from the SageMaker, check the following [GitHub](https://github.com/ally-jin/QTM-350-Final-Project-.git).






# The Final Analysis

Hello, Everyone. This notebook will be walking you step-by-step through our final analysis and will end with a detailed description of our results. As a result of following through this notebook, you will learn how to import your data into the environment, create summary statistics, fit a linear regression model to calculate statistical correlation, and finally, visualize your results in a clear and concise manner. 

## Importing the Data

Before we can start the analysis, we need to import the data into our environment. 

# Import Pandas package for later use
import pandas as pd

# Import the data
from google.colab import files
uploaded=files.upload()

import io
df=pd.read_excel(io.BytesIO(uploaded['FinalProjData.xlsx']))

# Check if the data is there

Now that we know that our data has been successfully loaded into the environment, we can start our analysis.

## Summary statistics

We want to get a better understanding of our data. To do so, we will find the average accuracy for the different Genres and for the different speeds.

# Basic summary statistics
df.describe()

# Importing matplotlib package
import matplotlib.pyplot as plt

# Visualizing the distribution of Accuracy 
plt.hist(df['Accuracy'], bins = 25)
plt.xlabel('Accuracy')
plt.ylabel('Distribution Density')
plt.title('Distribution of Accuracy Scores')


# Use pandas to find average accuracy by genre 
df_genre=df.groupby(['Genre'], as_index=False).mean()
df_genre

# Visualizing the correlation between genre and accuracy
plt.bar(df_genre['Genre'], df_genre['Accuracy'])
plt.xlabel('Genre')
plt.ylabel('Average Accuracy (%)')
plt.title('Genre vs. Average Accuracy')

# Use pandas to find average accuracy by speed
df_speed=df.groupby(['Speed'], as_index=False).mean()
df_speed

# Visualizing the correlation between speed and accuracy
plt.bar(df_speed['Speed'], df_speed['Accuracy'])
plt.xlabel('Speed')
plt.ylabel('Average Accuracy (%)')
plt.title('Speed vs. Average Accuracy')

# Use pandas to find average accuracy by genre and speed
df_both=df.groupby(['Speed', 'Genre'], as_index=False).mean()
df_both

As shown from the tables above, the different genres and speeds definitely make a difference in how accurate Amazon Transcribe's transcription is. In order to know the magnitude of the effect, however, we will need to create a model a fit a linear regression on our data.

## The Linear Regression

Our model is very simple. We want to know just how much genre and speed effects the accuracy of the transcription. Therefore, our model will look like this:

$Accuracy_{estimate} = B_0 + B_1*Genre + B_2*Speed + residual$

# Import the package linregress from scipy.stats
import statsmodels.formula.api as smf

# Run the model
results = smf.ols('Accuracy ~ Speed+Genre', data=df).fit()
results.params

results.tvalues

## The results 

Our model has returned some interesting results. As seen above, songs displayed at the original speed are roughly 17.88% more accurate than those at half  speed (0.5x) and 16% more accurate than those at double speed (2.0x). We initally hypothesized that transcribe performance would be better at the half the original speed. This is because most songs have quite speedy tempo, which makes the system hard to pick up the lyrics. Hence, we believed that the slowed down audio would faciliate the transcription as the service might be able to pick up more lyrics than either from original or faster speed. 

It is important to note that only the coefficient on the original speed is statistically significant (1% level). 

The genres also produced very interesting results. We hypothesized that rock and rap would both be very hard to transcribe because of background sounds and speed of tempo, respectively. However, to our surprise, rap had the best average accuracy and was generally 11.42% more accurate than pop and 16.98% more accurate than rock. It is understandable that rock has the worst accuracy out of the three genres. We believe this has been the case for the following reasons: 

1. Rap songs (at least most of the rap songs we used) were full of lyrics unlike other genre songs. This means that lyrics were emphasized rather than the melody or other noise; therefore, the consistent presence of the words significantly improves AWS Transcribe performance.

2. Songs that have lot of melody/MR: when the singer sings words in rhythmic tone or don't pronounce the words precisely & distinctly, the lyrics tend not to be picked up well. 

3. Songs with lot of non-vocabulary words (i.e ooh, woahh, uhmmm mmhmm, etc) barely got picked up.

4. Rock songs tend to have a lot of background music, which could make it hard for the Transcribe service to properly distinguish words from the noise in the audio.

5. Rock artists are often yelling or speaking in a unorthodox manner that can make them hard to understand. 

From these two coefficients, only the one on the genre Rap is statistically significant and it is also at the 1% level. 

## Conclusion 

In conclusion, the actual results was not in align with our hypothesis. However, we believe there are many factors that resulted in these results. 

- For most of the time, AWS Transcribe service was not able to formulate text for the entire song. 
- For uncler reasons, the transcription tended to be a lot shorter (150 words on average) than the actual song. 
- Such deficency was reflected in the accuracy of the transcriptions for all levels of speed and genre. 
- The average accuracy was very low (only 36.43%); therefore it is very clear that AWS Transcribe was having trouble with all speeds and genres. 

Based on the outcome and analysis, AWS Transcribe service is not the most adequate method for transcribing songs. Rather, it would be more useful for  different types of audio files that primarly involves words (i.e speeches, interviews, testimony, etc).


Thank you everyone for following our analysis. We hope you learned a bit about AWS Transcribe and how it should NOT be used to transcribe song lyrics. Your are much better of just looking for lyrics on google
