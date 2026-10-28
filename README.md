# Apple Stock Predictor

A neural network built from scratch in NumPy that predicts Apple's (AAPL) next closing price from the previous 5 days.

## Run it

```bash
pip install numpy matplotlib
python main.py
```

Prints the loss every 100 epochs, then shows two graphs: training loss, and actual vs predicted prices.

## The setup

- **Input:** 5 closing prices in a row
- **Output:** the next day's closing price
- **Network:** `5 -> 8 -> 4 -> 1` (two hidden layers with ReLU)
- **Loss:** mean squared error
- **Training:** gradient descent, 6100 epochs, learning rate 0.001

## The maths

### 1. Scaling the data

Prices are around 340, which makes training unstable. So everything gets standardised first:

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

where $\mu$ is the mean and $\sigma$ is the standard deviation. After training, predictions are converted back to dollars with:

$$\hat{y} = \hat{y}_{scaled} \cdot \sigma_y + \mu_y$$

### 2. Forward pass

Each layer does a weighted sum, then an activation:

$$z_1 = XW_1 + b_1 \qquad a_1 = \text{ReLU}(z_1)$$

$$z_2 = a_1W_2 + b_2 \qquad a_2 = \text{ReLU}(z_2)$$

$$\hat{y} = a_2W_3 + b_3$$

- $W$ are the **weights** and $b$ are the **biases**. These are what the network learns.
- **ReLU** is just $\max(0, z)$. Negative numbers become 0, positives stay the same. Without it the whole network would collapse into one big linear equation.
- The last layer has no ReLU because we want it to output any number (a price), not just positives.

### 3. Weight initialisation

Weights start as random numbers scaled by:

$$W \sim \mathcal{N}(0,1) \cdot \sqrt{\frac{2}{n_{in}}}$$

where $n_{in}$ is the number of inputs to that layer. This is **He initialisation**. It stops the signal from blowing up or shrinking to nothing as it goes through the layers. Biases start at 0.

### 4. Loss (mean squared error)

How wrong is the network?

$$L = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2$$

where $m$ is the number of samples. Squaring means errors are always positive and big mistakes get punished a lot more than small ones.

### 5. Backpropagation

Now we need to know how much each weight contributed to the error, so we know which way to nudge it. That's the gradient $\frac{\partial L}{\partial W}$, found with the **chain rule** working backwards from the output.

**Output layer**

$$\frac{\partial L}{\partial \hat{y}} = \frac{2}{m}(\hat{y} - y)$$

$$\frac{\partial L}{\partial W_3} = a_2^T \cdot \frac{\partial L}{\partial \hat{y}} \qquad \frac{\partial L}{\partial b_3} = \sum \frac{\partial L}{\partial \hat{y}}$$

**Second hidden layer**

Pass the error back through $W_3$, then through the ReLU:

$$dA_2 = \frac{\partial L}{\partial \hat{y}} \cdot W_3^T$$

$$dZ_2 = dA_2 \odot \mathbb{1}(z_2 > 0)$$

The derivative of ReLU is 1 if $z > 0$ and 0 otherwise, so $\mathbb{1}(z_2 > 0)$ just acts as an on/off switch. $\odot$ means element-by-element multiplication.

$$\frac{\partial L}{\partial W_2} = a_1^T \cdot dZ_2 \qquad \frac{\partial L}{\partial b_2} = \sum dZ_2$$

**First hidden layer**

Same thing again, one layer further back:

$$dA_1 = dZ_2 \cdot W_2^T \qquad dZ_1 = dA_1 \odot \mathbb{1}(z_1 > 0)$$

$$\frac{\partial L}{\partial W_1} = X^T \cdot dZ_1 \qquad \frac{\partial L}{\partial b_1} = \sum dZ_1$$

### 6. Gradient descent (the update)

Step each parameter a little bit *against* its gradient, so the loss goes down:

$$W \leftarrow W - \alpha \cdot \frac{\partial L}{\partial W}$$

$$b \leftarrow b - \alpha \cdot \frac{\partial L}{\partial b}$$

$\alpha$ is the learning rate (0.001 here). Too big and it overshoots, too small and it takes forever.

Do steps 2 to 6 over and over for 6100 epochs and the loss drops.

## Heads up

Only 9 samples and no train/test split, so the model is basically memorising the data. It's a learning project, not a trading strategy.
