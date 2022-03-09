
/**********************************************************
* README
* HW6: Artificial Neural Networks
* Comp 131 AI
* Carina Ye
* Dec 17, 2021
*********************************************************/

Files: 

README: this document, explains the basic ANN architecture
hw6.py: main implementation
ANN - Iris data.txt: input data file


Program Purpose:

This program implements an Artificial Neural Network to classify three classes of iris plants from scratch. The implementation follows class slides. The program reports the network accuracy on training and testing data sets and also supports user input.


Design:
    
1. Network architecture:
(1) Input layer: 4 neurons that represent the 4 iris features
(2) 1 Hidden layer: 4 neurons
(3) Output layer: 3 neurons since there are three iris classes

2. Steps to train the network:
(1) initialize parameters (weights and biases for the two layers)
(2) forward propagation
(3) backward propagation
(4) parameter updates using gradient descent

3. activation functions
For the hidden layer we used ReLU. We used softmax for the output layer. 

4. learning rate is set to 0.1

5. Model Performance:
After 500 iterations, training accuracy achieves 97.4% and testing accuracy achieves 98.5%.


Testing:

To test this program, run the "hw6.py" file. The program first reports the network accuracy on training and testing data sets. Then it ask for user inputs. Each enter should be as entered as four numbers seperated by commas. User can exit the program by entering "q". 



