# Go-ANN
An artificial neural network builder and trainer. Does not save an build network as of yet.

Net.go will allow multiple layers of nodes
Each layer gets a type of activation function and a number of nodes

How To Use
- Make a main.go
- import wrk "github.com/Go-ANN/BuilderTrainer"

- var x = wrk.Network{}
- x.AddLayer(wrk.[activation function], #) // input data. Activation function does not affect network on input layer
- x.AddLayer(wrk.[activation function], #) // hidden or final layer
- x.ConnectLayers()                        // puts in all the weight data between layers
- x.PutData([input])                       // fills the input layer with your data
- x.Propagation()                          // Maths the data through the network
- x.BackPropagation([expected])            // corrects the network based on expected values
- x.GetFinal()                             // returns the final layer data from the network

Another network in Development

inout ------
            \
inout ------ + bias -> activation -> inout
            /
inout ------

Backpropagation is the training method

act(x) is sigmoid activation of x

guess = act(mx + b)

Error = guess - data

Cost = Error^2

dCost
-----= 2*Error*Error' = 2*Error*( guess-data )' = 2*Error*( act( mx+b ) - data )' = 2*Error*( act(mx+b)' - data' )
 dm

dCost
-----= 2*Error*( act( mx+b ) * ( 1 - act( mx+b )) * ( mx+b )' - 0 )
 dm

dCost
-----= 2*Error*( act( mx+b ) * ( 1 - act( mx+b )) * mx' + b' - 0 ) = 2*Error*( act( mx+b ) * ( 1 - act( mx+b )) * x + 0 - 0 )
 dm
