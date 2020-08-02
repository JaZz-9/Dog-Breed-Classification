# Dog Breed Classification
**Who's a good dog? Who likes ear scratches? Well, it seems those fancy deep neural networks don't have all the answers. However, maybe they can answer that ubiquitous question we all ask when meeting a four-legged stranger: what kind of good pup is that?** 

# Demo 
<div align = "center" >
<img src='https://github.com/TheLethargicOwl/Dog-Breed-Classification/blob/master/images/demo.gif'/>
<br></br>
Link to full video : https://www.youtube.com/watch?v=5vY9zWx0pJU
</div>

# Table of Contents:-
* [How to get the webapp running for you?](#how-to-get-the-webapp-running-for-you)
* [Solution Approach](#solution-approach)
* [Important Links and Requirements](#important-links-and-requirements)
 
## How to get the webapp running for you
 * Fork the repo 
 * Open [web-app-runner](https://github.com/TheLethargicOwl/Dog-Breed-Classification/blob/master/DogBreedStreamLit.ipynb)
 * Run the web-app runner and get going 

## Solution Approach 
 * Trained several models of **Resnet50, Resnet101, efficientnet b0, b1, b2, b3, SeREsNEXT** on Pytorch
 * **Ensembled** over the best performing models 
 * Tried **Cutmix, Cutout, Fmix** - new augmentaion techniques to achieve better generalization and results 
 * Performed **Test Time Augmentations** to get better results
 * **Metric used** : *Accuracy* 
 * **Loss Function** : *BCE with Logits*, *Categorical Crossentropy* 
 * **Optimizers** : *Adam*, *Ranger*, *SGD*
 * **Best Accuracy** : *81.90 percent*
 
## Important Links and Requirements
 * EfficientNet B0 trained model - https://drive.google.com/file/d/13PYBuRvWWHH9JjIbVY4imkRFo5Az5Zyw/view?usp=sharing
 * Link to competition : https://www.kaggle.com/c/dog-breed-identification
 * Link to data : https://www.kaggle.com/c/dog-breed-identification/data
 
