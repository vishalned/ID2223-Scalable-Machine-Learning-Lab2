# ID2223-Scalable-Machine-Learning-Lab2

Lab 2 of the course ID2223 Scalable Machine Learning at KTH.\
Group members: Ennio Rampello, Vishal Nedungadi -- Group 50

The objective of this work is the implementation of a complete Machine Learning pipeline for both training and inference. The pipeline consists of different components with different purposes, where each one can be scaled independently of the others, based on the specific load.
We have fine-tuned the Whisper model to work with the Swedish language.
The following is a description of all the components and procedures that we have implemented in order to carry out the lab work.

   1. **Preprocessing and storage**: the preprocessing is performed on a GCP Instance with a NVIDIA T4 GPU. The preprocessed data is then stored in a Storage Bucket in GCP. The preprocessing code is available in `preprocess.py`.
   2. **Training pipeline**: after downloading the data from the Storage Bucket, the training is performed on the same GCP instance with a T4 GPU. The model is then stored in Hugging Face. The code for training is in `train.py`.
   3. **Inferencing pipeline**: after the models have been uploaded to huggingface, we just load this onto our gradio ui and host this ui on huggingface. The url is provided below. The code to run the UI is in `hugging_face/app.py`.
   4. **Model optimizations**: To improve the performance of the model, we read the original paper, and used some of the hyper parameters used by them. We also found another medium blog post, that ran a hyper parameter grid search to find the most important parameters. From this, we found that the most important is the learning rate and weight decay. So we changed these 2 parameters and acheived a total WER of 19.6. When we ran the base code provided in the course repo, we get a WER of 20.04. This shows that there was some improvement. Ofcourse we could have done more, but each training attempt took 9 hrs to train, and hence we only had time to run 3 tests. 

### Repository Contents

   1. `hugging_face/` contains the code for inferencing on a audio from a user input. [URL link](https://huggingface.co/spaces/vishalned/scalable_ml_lab2)
   
   
