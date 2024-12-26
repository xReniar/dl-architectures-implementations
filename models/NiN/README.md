# NiN
<div style="text-align: center;">
    <img src="../../docs/architectures/NiN.png" alt="NiN architecture" width="50%">
</div>

## Implementation
As it's shown in the image this architecture have more repeated blocks called `NiN-block`, this is implemented with this function:
```python
def NiN_block(in_features:int, out_features:list, kernel_size:int, stride:int = 1, padding:int = 0):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features[0], kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_features[0], out_features[1], kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_features[1], out_features[2], kernel_size=1),
        nn.ReLU()
    )
```
Input features and output features needs to be specified as well as the kernel size, the stride and padding are not mandatory. Using this function simplifies the implementation of the network without duplicating code. The final layer is a `Global Average Pooling`, this can be done like this:
```python
nn.AdaptiveAvgPool2d((1,1))
```
The parameter `(1, 1)` specifies that the desired output should have a size of 1x1 for each channel of the feature map. In other words, the entire feature map of each channel will be reduced to a single scalar value, representing the average of all the values in that channel.
