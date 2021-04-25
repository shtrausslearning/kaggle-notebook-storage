#  **HUMMINGBIRD CLASSIFICATION**
**Preword**

[View Notebook on Kaggle](https://www.kaggle.com/shtrausslearning/hummingbird-classification-with-cnn) for nicer formatting output & interative plots.
Wonderful Photo by [Zdeněk Macháček](https://unsplash.com/@zmachacek)

**Still on ToDo List:**
- Video Image Test Inference.
- Section about the influence of image clarity for better classification.

# 1. <span style='color:#B6DA32 '> INTRODUCTION </span>

![banner](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dehie0n-a2e641c6-af9a-47cf-832c-8015e86b7347.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOiIsImlzcyI6InVybjphcHA6Iiwib2JqIjpbW3sicGF0aCI6IlwvZlwvOGNjMWVlYWEtNDA0Ni00YzRhLWFlOTMtOTNkNjU2ZjY4Njg4XC9kZWhpZTBuLWEyZTY0MWM2LWFmOWEtNDdjZi04MzJjLTgwMTVlODZiNzM0Ny5qcGcifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6ZmlsZS5kb3dubG9hZCJdfQ.CnD5mRUqdz93URXVhEPqPNQLlY5U4bq-uDVWCZ2Bn4g)

<b>Bird Monitoring </b>

- Having conducted a quick background investigation, it has become evident that a sizable portion of the monitoring & observation surrounding birds is mostly done manually by researchers & bird monitoring enthusiasts (Eg. you can submit your own sightings to [Hummingbirdcentral](https://www.hummingbirdcentral.com/) & you wouldn't need to look far for bird sighting in general, take [Xeno-Canto](https://www.xeno-canto.org/) for example.
- The key the points outlined in <b>Importance of this work</b> is quantity, & with the evolution of monitoring technology, it has become quite accessable for anyone to acquire a camera/videocamera & start contributing to the overall process of bird monitoring.
- When it comes to hummingbirds, there is a lot of interest across various parts of the United States. Enthusiasts and bird watchers often <b>set up bird feeders</b> to attract birds & one of the feeding spots ( which is used in this dataset ) is located in the state of Colorado.
- To monitor the feeder we simply need a digital camera/recorder pointed at the feeder, the recording data can then be extracted and analysed autonomously to identify the species that were present during recording. Via separate automation or visual inspection, frames can be extrated when birds are present and a <b>classifier can be used to identify them autonomously</b>.

<b>Where Machine Learning Comes In</b>
- A key component in the bird identification process is the actual model that will need to be trained, for it to be able to identify and classify a particular bird present at the feeder.
- Machine Learning & in particular the use of Convolution Neural Network can be used to build a classification model, so we can identify the species present at the feeder.
- The photos collected from [akimball002](https://www.kaggle.com/akimball002) feeders' cameras with a binary bird-finder classification model to create a CNN multi-species and gender classification model. 

<b>Purpose, Goal & Application</b>

- The <b>purpose of this project</b> is to create an <b>image classification for hummingbird species</b> and <b>genders</b> that visit feeders in Colorado. 
- The <b>ultimate goal</b> is to have a <b>classification system that can be deployed at any feeder</b>. 
- It is <b>applicable to anywhere that hummingbirds migrate or breed</b> given additional datasets for those species. 

<b>Importance of this work</b>

- This is <b>important to the continued monitoring of hummingbird species</b> and <b>migration patterns</b>. 
- Humming bird <b>migration is otherwise reliant on individual bird watchers</b> to see and report their observations. 
- If avid bird lovers setup a similar system, then <b>conservation organizations would have better information on migratory and breeding patterns</b>. 
- This knowledge can be used to determine if specific changes in the environment or ecology has positively or negatively impacted bird life.

<b>Interesting References</b>
- [Hummingbirds 101](https://www.perkypet.com/advice/hummingbirds-101)
- [Hummingbird Nectar Recipes](https://nationalzoo.si.edu/migratory-birds/hummingbird-nectar-recipe)
- [Keras Image Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)

# 2. <span style='color:#B6DA32 '>DATASET </span>

Applicable Dataset can be accessed on Kaggle; [Hummingbirds at my feeders](https://www.kaggle.com/akimball002/hummingbirds-at-my-feeders) dataset

- In the context of bird monitoring, what this dataset outlines more than anything else is that you don't need to place cameras right next to the feeder, which for some species can be offputting & the images don't need to be of perfect quality in order to create a classifier that can identify hummingbird species accurately.
- We will go through the images images in <b>Training Image Exploration</b>
- The primary birds at the feeders this year are [broad-tailed (Selasphorous platycerus)*](https://www.allaboutbirds.org/guide/Broad-tailed_Hummingbird/id ) and a few [rufous (Selasphorous rufus)*](https://www.allaboutbirds.org/guide/Rufous_Hummingbird/) hummingbirds</b>.

<b>Folder Layout</b>
- Let's see what kind of dataset we are dealing with; there seems to be a specific folder already assembled specifically for modelling; <b>hummingbirds</b>, which contains subfolders for each separate class. When it comes time to create a dataset, this will be very convenient since we can just call the <b>.flow_from_directory</b> function.
- There also is a folder containing 
- As well as two other folder associated with a video recording (collection of still images); <b>video_test</b> & All images in the dataset; <b>All_images</b>

**RUFOUS FEMALE EXAMPLES**
![rufousfemale](https://www.kaggleusercontent.com/kf/60030590/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..c5xGz3aVDoKVE-KKZHD0Zg.CJ_XkfwO7Qv1yB97VnBKKwRAiUmci3Zm6pHNca6eoq6Lt15rA3rsaMovWvNy87BX1gqF6OiG0Nh3FHrEdwCnS6-m09cXI3b50Zi8jUsT5oL2uzSXtroncQkqoJ8DkqOjiP9DUEmKv7kmfz8ZLB-NuE4pMhvaGWp_rFd7CkmuIuw4sgxxYmww-3oDXjUeh3M2RdXo56VsixIWQ70v3SmUhyumwO8r9cwA7GmtfKWHAh7HwKj4CZraWIKj22Pb-57Mxf5nsweEaZ4lXBSg5neaFT1RFUHrJAmTFP1IBUicYpyhRkUaynFVZv9uhT5BFhiRSUdKO0gVk4krC1IBjhJyysFSBlnA2jodLAI-bj09S9M8jCqagj4flcrcS2S5N0H6FrpcQ3w2Jbx83nB0dOH0MPXeMb29zbRKabStZ7tEBNgjxx2cNPCXjgvBgwlXGMFbKVISOl_Vj3-QCXgnQTnTRGc4a5KQnh7BjcSxea4zKWGqvrOvO4Jd878e-vjC7wGgSefOzbsozjfAT-u6KV0cCoSyUphB_wGBvxQn_NC7lSp8lgxL-a7w39j85VqwYCmcJviPQzuQgruQc7P56d0y0kI9lG52bVQt8a1bwrW6_TKfSCWi-cc_A2giKGMFEIHhSDh9y5FHrUS7h1kd2LHEh74I6lZPeTqiqksKyzkNehECPs74ivvO_GbPAJuZleyc.wusxoF0JChbN5Idwy55TwQ/__results___files/__results___8_0.png)

