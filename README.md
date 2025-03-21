## ["A Hierarchical Deep Temporal Model for Group Activity Recognition."]


## Contents
0. [Abstract](#abstract)
0. [Model](#model)
0. [Baselines and Experiments](#baselines-and-experiments)  
0. [Dataset](#dataset)
0. [Installation](#installation)  
0. [Key Insights](#key-insights)
0. [Results](#results) 
0. [Future Improvements](#Future-Improvements)    


## Abstract
This project investigates **group activity recognition** in volleyball games through an extensive ablation study across **multiple baselines**. 

Using **temporal and spatial information alongside deep learning models** , the experiments emphasize learning representations at both individual player and team levels. 

This repository explores the progression of methods, from naive image classification to hierarchical models integrating LSTMs for temporal dependencies, achieving a best accuracy of **82.42%** on the final baseline **B8**.
  
This Project is a PyTorch implementation of the [Original Paper](https://arxiv.org/abs/1511.06040) ,which was originally implemented in C++ and the Caffe deep learning framework by the authors.

## Model
<img src="https://github.com/mostafa-saad/deep-activity-rec/blob/master/img/fig1.png" alt="Figure 1" height="400" >

**Figure 1**: High level figure for group activity recognition via a hierarchical model. Each person in a scene is modeled using a temporal model that captures his/her dynamics, these models are integrated into a higher-level model that captures scene-level activity.

<img src="https://github.com/mostafa-saad/deep-activity-rec/blob/master/img/fig2-b.png" alt="Figure 2" height="400" >

**Figure 2**: Detailed figure for the model. Given tracklets of K-players, feeding each tracklet in a CNN, followed by a person LSTM layer to represent each player's action.Then pooling over all people's temporal features in the scene. The output of the pooling layer is feed to the second LSTM network to identify the whole teams activity.

<img src="https://github.com/mostafa-saad/deep-activity-rec/blob/master/img/fig3.jpg" alt="Figure 3" height="400" >

**Figure 3**: Previous basic mode drops spatial information. In updated model, 2-group pooling to capture spatial arrangements of players.

<img src="https://github.com/user-attachments/assets/e9d8aa64-859f-4052-9a6e-09f45aa29d8a" alt="Figure 3" height="400" >

**Figure 4**: Hierarchical Deep Temporal Model


## **Baselines and Experiments**  

<img src="https://github.com/user-attachments/assets/7b3f45e9-5db8-4700-bd89-25f5d36bf95b" alt="Figure 3" height="400" >

**Figure 5** :  Baselines

### **Baseline B1-Tuned**  
- **Objective:** Fine-tune an image classifier for 8 activity classes using only the middle frame of each clip.  
- **Implementation:**  
  - Fine-tuned ResNet50 on the middle frame.  
  - Extended experimentation by using a sequence of 9 frames (5 before, 4 after the target).  
- **Result:** A foundational accuracy benchmark for subsequent baselines.  

---

### **Baseline B3**  
A three-step model emphasizing individual player-level classification:  
1. **Step A:** Train ResNet50 on cropped player images for 9 individual actions.  
2. **Step B:** Infer representations by extracting player features (2048 dimensions per player) and max-pooling them into a single scene-level feature.  
3. **Step C:** Train a classifier using these pooled representations for 8 activity classes.  

---

### **Baseline B4**  
- **Implementation #1:**  
  - Extract 9-frame sequences using the B1-tuned classifier.  
  - Train an LSTM to classify activities based on these temporal representations.  
    

- **Implementation #2:**  
  - Augment the ResNet50 backbone with an LSTM layer for end-to-end training.  

---

### **Baseline B5**  
- **Objective:** Model temporal information at the player level.  
- **Approach:**  
  - Train an LSTM for player-level sequences (9 frames).  
  - Max-pool player representations into scene-level features and train a classifier.  

---

### **Baseline B6**  
- **Enhancement:** Incorporate LSTM layers at the scene level to classify sequences of 9 frames.  
- **Key Insight:** Combining temporal modeling at both individual and scene levels significantly boosts performance.  

---

### **Baseline B7**  
- **Full Model (Version 1):**  
  - **LSTM-1:**  
    - Input: Features (9 × 12 × 2048) for 12 players across 9 frames.  
    - Output: Temporal player-level features max-pooled into a scene-level representation (9 × 2048).  
  - **LSTM-2:**  
    - Input: Scene-level representations (9 × 2048).  
    - Output: Classification over 8 activity classes.  

---

### **Baseline B8**  
**Final Model (Best Accuracy: 82.42%)**  
- **Key Innovation:**  
  - Separate player representations into **two teams** before pooling.  
  - Scene representation: Concatenate max-pooled features for team 1 and team 2.  
  - Input to LSTM-2: 9 × 4096 (2048 per team).  
- **Impact:** Improved understanding of team dynamics resulted in the highest accuracy across all baselines.  


## Dataset

### [NEW Download Link (all below combined google drive](https://drive.google.com/drive/folders/1rmsrG1mgkwxOKhsr-QYoi9Ss92wQmCOS?usp=sharing). 

They collected a new dataset using publicly available **YouTube volleyball** videos. They annotated **4830 frames** that were handpicked from **55 videos** with 9 player action labels and 8 team activity labels. 

<img src="https://github.com/mostafa-saad/deep-activity-rec/blob/master/img/dataset1.jpg" alt="Figure 3" height="400" >

**Figure 5**: A frame labeled as Left Spike and bounding boxes around each team players is annotated in the dataset.


<img src="https://github.com/mostafa-saad/deep-activity-rec/blob/master/img/dataset2.jpg" alt="Figure 4" height="400" >

**Figure 6**: For each visible player, an action label is annotaed.

They used 3493 frames for training, and the remaining 1337 frames for testing. The train-test split of is performed at video level, rather than at frame level so that it makes the evaluation of models more convincing. The list of action and activity labels and related statistics are tabulated in following tables:

|Group Activity Class|No. of Instances|
|---|---|
|Right set|644|
|Right spike|623|
|Right pass|801|
|Right winpoint|295|
|Left winpoint|367|
|Left pass|826|
|Left spike|642|
|Left set|633|

|Action Classes|No. of Instances|
|---|---|
|Waiting|3601|
|Setting|1332|
|Digging|2333|
|Falling|1241||
|Spiking|1216|
|Blocking|2458|
|Jumping|341|
|Moving|5121|
|Standing|38696|

**Further information**:
* The dataset contains 55 videos. Each video has a folder for it with unique IDs (0, 1...54)
 * **Train Videos**: 1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54
 * **Validation Videos**: 0 2 8 12 17 19 24 26 27 28 30 33 46 49 51
 * **Test Videos**: 4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47
* Inside each video directory, a set of directories corresponds to annotated frames (e.g. volleyball/39/29885)
  * Video 39, frame ID 29885
* Each frame directory has 41 images (20 images before target frame, **target frame**, 20 frames after target frame)
  * E.g. for frame ID: 29885 => Window = {29865, 29866.....29885, 29886....29905}
  * Scences change quite rapidly in volleyball, hence frames beyond that window shouldn't represent belong to target frame most of time.
  * In our work, we used 5 before and 4 after frames.
* Each video directory has annotations.txt file that contains selected frames annotations.
* Each annotation line in format: {Frame ID} {Frame Activity Class} {Player Annotation}  {Player Annotation} ...
  * Player Annotation corresponds to a tight bounding box surrounds each player
* Each {Player Annotation} in format: {Action Class} X Y W H
* Videos with resolution of 1920x1080 are: 2 37 38 39 40 41 44 45 (8 in total). All others are 1280x720.


<img src="https://github.com/user-attachments/assets/2767214f-8cc7-4be9-a2f4-b381bfbccf4f" alt="Figure 3" height="400" >

**Figure 7**:    9 person level labels, and 8 group activity labels.


---

## **Key Insights**  
1. **Temporal Information Matters:** Adding LSTMs significantly enhanced model performance.  
2. **Team Representations Are Crucial:** Separating and pooling player features by teams reduced confusion between activities.  
3. **Naive Classifiers Fall Short:** Scene-level classification benefits from hierarchical modeling of players and teams.  

---

## **Results**  
| **Baseline** | **Methodology**                     | **Test Accuracy** |  
|--------------|-------------------------------------|-------------------|  
| **B8**       | Hierarchical + Team separation      | **82.42%**        |  

---

## **Installation**  
1. Clone this repository:  
   ```bash  
   git clone https://github.com/MohamedTharwat21/deep-activity-rec


## **Future Improvements**
While the current implementation leverages ResNet50 and LSTM for temporal modeling, future enhancements could incorporate the following state-of-the-art methodologies to push the boundaries of group activity recognition:

### 1. Transformer-based Models
- **Vision Transformers (ViTs)**: Employ ViTs for feature extraction to capture global contextual relationships across frames.  
- **Temporal Attention Mechanisms**: Introduce attention mechanisms to better model inter-frame dependencies and long-range temporal patterns.  
- **Multi-modal Transformers**: Explore transformers that integrate spatial, temporal, and team-level dynamics effectively.

### 2. Graph Neural Networks (GNNs)
- **Graph Structure**: Model player interactions as a graph where nodes represent players, and edges capture their spatial and temporal interactions.  
- **Spatio-temporal GNNs**: Dynamically learn relationships between players within a frame and across sequential frames.  
- **Graph Pooling**: Experiment with graph pooling techniques to generate scene-level embeddings for enhanced activity recognition.

### 3. Self-Supervised Learning
- **Unlabeled Data Pretraining**: Integrate self-supervised pretraining techniques to leverage unannotated data for better feature representations.  
- **Contrastive Learning**: Utilize contrastive learning frameworks to pretrain models on frame sequences, improving performance on downstream tasks.

### 4. Data Augmentation and Synthetic Data
- **Advanced Augmentation**: Incorporate techniques such as frame-level adversarial perturbations and time-warping.  
- **Generative Adversarial Networks (GANs)**: Use GANs to synthesize player movements and activities, expanding dataset diversity.

### 5. Ensemble Methods
- **Model Integration**: Combine ResNet-LSTM architectures with GNN and Transformer-based models for robust predictions.  
- **Stacking and Voting**: Use stacking techniques or voting mechanisms to integrate multiple model predictions effectively.

### 6. Real-time Inference
- **Optimization for Edge Devices**: Optimize the model pipeline for deployment on edge devices for **real-time group activity recognition** in dynamic settings like sports analytics or surveillance systems.

### Expected Impact of Future Work
Implementing these improvements could significantly enhance the accuracy and robustness of group activity recognition models, making them more suitable for complex real-world scenarios.

These methodologies also pave the way for broader applications in multi-agent systems, behavioral analysis, and automated event summarization.

