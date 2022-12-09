# Lives of Plants
## Identifying Plants in the Wild
## Final Write Up/ Refelction

### Introduction
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The goal of this project is to implement a model that can effectively predict a plant species given a picture of a plant. We referenced a paper </br>
(https://bmcecolevol.biomedcentral.com/articles/10.1186/s12862-017-1014-z) </br>
where the researchers utilized herbarium specimens to develop a model to predict plants. At the time of the paper, a herbarium specimen model is something that has not been done in the past. A lot of the specimens are currently being digitized thus using the specimens for plant identification is a relatively new opportunity. Identifying new specimens by hand is quite difficult and time consuming, thus automating the process may help in identifying any specimens that have not been identified yet. The researchers also expanded on this by utilizing the model to identify plants in the wild. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We chose this paper because it not only explores identifying plants in a confined space, but talks about the possible expansion of the model. It is said to be the first  use of deep learning to identify plants and we saw it as a good fit. Both of us were in a first year seminar about plants and we drew inspiration from the class. Using that inspiration we ran into this paper and felt that its research fit closely with our ideas. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Initially we planned on using an herbarium specimen, however, our initial dataset got compromised by ransomware, thus we had to switch datasets. The new dataset consisted of plants in the wild as opposed to herbarium specimens. With this new dataset we expected some more difficulty due to there being more noisy data. However, this aligns better with our future goals. After building a model utilizing herbarium specimens, we considered attempting to use our model on plants in real life. With the difficulties with our original dataset, we decided to implement a model to predict plants in the wild from the start.

### Methodology
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Our dataset as explained in the introduction had to be changed from herbarium specimen to plants in the wild. We obtained our dataset from We acquired our data from inaturalist, which is a database consisting of verified research grade images of plants, taken in the wild. We chose 10 species to train on. The 10 species are as follows:Acer Macrophyllum, Arnica, Lewisia, Lupinus latifolious, Salal, Salmonberry, Trillium, Vine maple, Western Pasqueflower, and Western Red Cedar. Although there was change in the dataset, our approach to the model remained fairly similar. For preprocessing the data, we first collected the data, turned it into a tensorflow dataset then split it into training and validation datasets. We then normalized the images and randomly flipped and rotated the images to mitigate overfitting. We also OHE our labels which were handled when creating the tensorflow dataset. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  For our model, we utilize 6 Conv2d layers, all with kernel regularizers, to reduce overfitting. We also max pooling, batch normalization, flatten, and dropout layers. The dropout layer played a significant part in reducing noisy data and thus reduced overfitting. We used two dense layers with a leaky relu activation and our final layer is a softmax layer. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To train our model, we used an Adam optimizer with a learning rate of 0.0001. After attempting different learning rates, we felt that this value performed the best. Our loss function is Categorical Cross Entropy and we ran 75 epochs a batch size of 3.  </br>

### Results
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; After working through and implementing different models, our best model is the one described above (named Plant_Model in our code). This model resulted in a training accuracy of 73.21%, training loss of 1.5475, validation accuracy of 71.25% and validation loss of 1.9150.  The validation accuracy fluctuates more than the training accuracy which we believe is due to overfitting. We also attempted transfer learning, but did not yield the results we wanted, which is likely due to the lack of a proper base model. </br>

### Challenges
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In this project we ran into many challenges. Our first challenge was the loss of our original dataset which forced us to find and use another dataset. Because the new dataset consisted of plants in the wild and not herbarium specimens, we knew that the data would be more inconsistent. Due to the nature of the dataset, preprocessing was also a challenge. Building off this new dataset, the other issue we ran into was overfitting. This was due to a multitude of factors. Our two biggest factors were a small dataset and too much noise in our data. We did our best to acquire as much data as possible, however ,this was a challenge, due to using a different dataset than originally anticipated. Because these images were not herbarium specimens, they contained a lot more noise which we believe made it more difficult for our model to extract the features. We mitigated this to the best of our ability by utilizing Dropout layers and implementing regularization in our Conv2d layers. Our model relies heavily on a strong dataset, and while we were able to get a fairly robust dataset, more data would have helped our model perform more consistently. We also attempted to implement transfer learning to help make up for the smaller dataset, however, due to a lack of a strong base model, the model did not perform as well. Given these challenges we minimize overfitting and are able to create a strong model to identify plants. 

### Reflection
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We feel that our project turned out fairly well. Our goal was to train a model on 10 species of plants and we were able to do so with a relatively high accuracy. Our stretch goal was to train a model on plants in the wild, but due to the change in dataset, this became our base goal. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Our model works as expected. We are able to feed in a plant of the 10 species and can predict the species of the plant just over 70% of the time. Our approach changed due to the change in the dataset. Herbarium specimens are consistent in their format which would have made it easier to train. However, since we changed datasets to plants in the wild, we knew that there would be noise and inconsistencies. We were wary of overfitting and took steps to mitigate this as much as possible. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  If we did this project again, we would like to have started with the herbarium specimen and build off of that. We felt that the herbarium specimens would have given us a far better and more consistent baseline to build off of.  If we had more time, we would have liked to use the herbarium dataset to train the model, and use the model to implement transfer learning which we feel would have better trained our current dataset. This implementation was observed in our reference paper and is something we would like to consider implementing in the future. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Our biggest takeaways from this project are the dangers of overfitting and the ability to be flexible. We did not know that our original dataset would get compromised, however when it did, we feel that we handled it gracefully and were able to approach our problem from a different angle. We also had not run into a lot of issues with overfitting in our assignments, thus we learned a lot from this project. We realized the importance of preprocessing and creating a model in such a way that minimizes noise. This project was a learning opportunity for the both of us and we gained skills that we will take outside the scope of this class. </br>
