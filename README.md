# COVID-19, Pneumonia and Normal X-Ray Image Classification

<!-- wp:paragraph -->
<p>The novel Coronavirus also called COVID-19 originated in Wuhan, China in December 2019 and spread across the world. As the symptoms are also related to pneumonia patients, it is badly needed to diagnose accurately whether it is COVID-19 or Pneumonia for better treatment and avoid the transmission of COVID-19. The limited quantity of resources and lengthy diagnosis process encouraged us to come up with a Deep Learning model that can aid radiologists and healthcare professionals. In this study, we proposed a Deep Learning Model to automatically detect COVID-19, Pneumonia, and Normal patients from chest X-ray images. The proposed model is based on CNN architecture on a secondary dataset. The model has been trained and tested on the prepared dataset and the experimental results show an overall accuracy of 95.16%, and more importantly, the precision and recall rate for COVID-19 cases is 93.0% and 96.5% for 3-class cases (COVID vs Pneumonia vs Normal). The preliminary results of this study look promising and can be further improved by the different architecture and more training data.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2>Objectives</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>The specific objectives of the study are as follows:</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li>To detect 3 classes of patients using chest X-ray images</li><li>Fast and accurate diagnosis of the disease to avoid transmission of covid-19</li><li>Decrease the loss of life to faulty diagnosis</li><li>Getting proper treatment for the proper disease</li></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2>Dataset</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>For the purpose of the experiments, X-ray images were collected from secondary sources. A collection of X-ray images from selected from Kaggle’s repository was selected. This dataset consists of X-Rays from 3617 individuals with COVID-19, 4991 from healthy individuals which are labeled as normal, and 1346 X-Rays from individuals with viral pneumonia. All the images are in the Portable Network Graphics (PNG) file format, and with a resolution of 299-by-299 pixels with 3 color channels red, green, and blue.</p>
<!-- /wp:paragraph -->

<!-- wp:image {"align":"center","id":79,"sizeSlug":"full","linkDestination":"none"} -->
<figure class="wp-block-image aligncenter size-full"><img src="https://shahriaralamrakib.com/cv/wp-content/uploads/2022/09/x-ray.png" alt="" class="wp-image-79"/><figcaption><em>X-Ray Dataset</em></figcaption></figure>
<!-- /wp:image -->

<!-- wp:heading -->
<h2>Image Processing</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Using raw images in deep learning models leads to poor performance in classification, whereas preprocessing techniques increase the performance. The preprocessing techniques are also essential to speed up the training procedure. All the images were resized to 150x150 pixels for fast computation purposes. At this stage the amount of data is divided into training and testing with a data division of 80:20. After splitting the data training set contains 7960 images and 1991 for testing. The distribution of data used is 80:20 because other proportions will not be sufficient for the validation process. All images were normalized according to the pre-trained model standards.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2><strong>CNN Model Architecture</strong></h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Our proposed model is based on five basic components, namely convolutional layer, pooling layer, flatten layer, dense layer, and activation function. The components are used in different layers of our proposed model. A detailed discussion of each basic component is given below.</p>
<!-- /wp:paragraph -->

<!-- wp:image {"id":80,"sizeSlug":"large","linkDestination":"none"} -->
<figure class="wp-block-image size-large"><img src="https://shahriaralamrakib.com/cv/wp-content/uploads/2022/09/covid-model-1024x523.png" alt="" class="wp-image-80"/><figcaption><em>Proposed Model Architecture</em></figcaption></figure>
<!-- /wp:image -->

<!-- wp:heading -->
<h2>Model Summary</h2>
<!-- /wp:heading -->

<!-- wp:image {"align":"center","id":81,"sizeSlug":"large","linkDestination":"none"} -->
<figure class="wp-block-image aligncenter size-large"><img src="https://shahriaralamrakib.com/cv/wp-content/uploads/2022/09/Screenshot_952-1024x474.png" alt="" class="wp-image-81"/><figcaption><em>Model Summary</em></figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>We have used the Adam optimizer for weight updates, categorical cross-entropy loss function, and selected learning rate to compile the model. The model was trained with 20 epochs and batch size 64.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2>Results </h2>
<!-- /wp:heading -->

<!-- wp:image {"align":"center","id":83,"sizeSlug":"large","linkDestination":"none"} -->
<figure class="wp-block-image aligncenter size-large"><img src="https://shahriaralamrakib.com/cv/wp-content/uploads/2022/09/Screenshot_953-1024x377.png" alt="" class="wp-image-83"/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>The model achieved the best classification accuracy of 98.90% on train data after 20 epochs. Each epoch was done by batch size 64. The model showed a validation accuracy of 95.16% with the same epoch and batch size. The validation accuracy is the accuracy based on the test data.</p>
<!-- /wp:paragraph -->

<!-- wp:image {"align":"center","id":82,"sizeSlug":"full","linkDestination":"none"} -->
<figure class="wp-block-image aligncenter size-full"><img src="https://shahriaralamrakib.com/cv/wp-content/uploads/2022/09/download-5.png" alt="" class="wp-image-82"/><figcaption><em>Loss &amp; Accuracy over epochs</em></figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>The figure displays the training and validation loss and accuracy of the model over epochs. It indicates a smooth training process during which the loss gradually decreases and the accuracy increases. Moreover, it can be observed that the accuracy of both training and validation do not deviate much from one another in most cases, a phenomenon that can also be observed for the training and validation loss, indicating that the model is not overfitting.</p>
<!-- /wp:paragraph -->

<!-- wp:image {"align":"center","id":84,"sizeSlug":"full","linkDestination":"none"} -->
<figure class="wp-block-image aligncenter size-full"><img src="https://shahriaralamrakib.com/cv/wp-content/uploads/2022/09/download-6.png" alt="" class="wp-image-84"/><figcaption><em>confusion matrix</em></figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>The performance metrics mentioned in the figure are the top metrics used to measure the performance of classification algorithms. The proposed model achieved an average accuracy of 92.95%, while the data is actually COVD-19 and predicted as COVID-19. The percentage of actually Normal and also predicted as normal is 97.01%. The next class that is actually Pneumonia and Classified as Pneumonia is 94.80%.</p>
<!-- /wp:paragraph -->

<!-- wp:image {"align":"center","id":85,"sizeSlug":"full","linkDestination":"none"} -->
<figure class="wp-block-image aligncenter size-full"><img src="https://shahriaralamrakib.com/cv/wp-content/uploads/2022/09/Screenshot_954.png" alt="" class="wp-image-85"/><figcaption><em>Performance Evaluation</em></figcaption></figure>
<!-- /wp:image -->

<!-- wp:heading -->
<h2>Discussion</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Based on the results, it is demonstrated that the deep learning model may have significant effects on the automatic detection of covid-19 and pneumonia from X-ray images, related to the diagnosis of Covid-19.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The present work contributes to the possibility of a low-cost, rapid, and automatic diagnosis of the Coronavirus disease. It is to be investigated whether the extracted features performed by the CNNs constitute reliable biomarkers aiding in the detection of Covid-19. Also, despite the fact that the appropriate treatment is not determined solely from an X-ray image, an initial screening of the cases would be useful, not in the type of treatment, but in the timely application of quarantine measures in the positive samples, until a more complete examination and specific treatment or follow-up procedure are followed.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2>Limitations &amp; Future Work</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Some limitations of the particular study can be overcome in future research.</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li>The different architectures of deep learning like VGG16 / VGG19 / Alexnet or others can be implemented.</li><li>Due to time limitations and lack of resources can’t compare with multiple models</li><li>In the future, we can implement the model with real data.</li></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2>Conclusion</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>X-ray images play an important role in the diagnosis of COVID-19 infection from other pneumonia as advanced imaging evidence. Artificial Intelligence (AI) algorithms and radionic features derived from chest X-rays can be of huge help to undertaking massive screening programs that could take place in any hospital with access to X-ray equipment and aid in the diagnosis of COVID-19, as all the processes can be done automatically, the cost is significantly decreased compared with the traditional method. In order to speed up the discovery of disease mechanisms, this research developed a deep CNN-based chest X-ray classifier to detect COVID-19, Pneumonia, and Normal X-Ray images. The classification accuracy of the proposed model is 95.16% for 3 classes which is the highest achieved accuracy to the best of our knowledge on the datasets used in the experiments. Our future goal is to overcome hardware limitations and implement multiple models to compare the performance with a greater number of existing methods.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://github.com/mohammad-rakibul-alam/COVID-19-Pneumonia-Normal-Xray-Image-Classification" target="_blank" rel="noreferrer noopener">See the Full Project on GitHub</a></strong></p>
<!-- /wp:paragraph -->
