package main

import (
	"fmt"

	goann "github.com/Go-ANN"
	dtm "github.com/Go-ANN/DataManagement"
	"github.com/Go-ANN/act"
)

func main() {
	var x goann.Network
	var inout uint
	inout = 3

	x.AddLayer(act.Linear{}, inout)
	x.AddLayer(act.Linear{}, 5)
	x.AddLayer(act.Sigmoid{}, 6)
	x.AddLayer(act.SoftSign{}, 4)
	x.AddLayer(act.ReLU{}, 5)
	x.AddLayer(act.ReLU{}, inout)
	x.ConnectLayers()

	var input []float64
	input = make([]float64, 0, 100)
	var dmb dtm.DataBuilder
	dmb.SetNumInput(inout)
	dmb.SetNumOutput(inout)
	dmb.AddxTrainValues(input)
	dmb.AddyTrainValues(input)

	/* for a := 0; a < 100; a++ {
		input = append(input, rand.Float64())
		input = append(input, rand.Float64())
		input = append(input, rand.Float64())
	} */
	input = append(input, 0., 1., 0.)
	input = append(input, 0., 0., 1.)
	input = append(input, 1., 1., 0.)
	input = append(input, 1., 0., 1.)
	input = append(input, 0., 1., 1.)
	input = append(input, 1., 0., 0.)
	input = append(input, 0., 0., 0.)
	input = append(input, 1., 1., 1.)

	input = append(input, 0., 1., 1.)
	input = append(input, 0., 1., 0.)
	input = append(input, 1., 1., 1.)
	input = append(input, 1., 1., 0.)
	input = append(input, 1., 0., 0.)
	input = append(input, 1., 0., 1.)
	input = append(input, 0., 0., 1.)
	input = append(input, 0., 0., 0.)

	dmb.AddxTrainValues(input[:8*3])
	dmb.AddyTrainValues(input[8*3:])

	var dm = dmb.Build()

	dm.Train(160, x)
	view(x, inout, input)
	x.AddNode(1)
	dm.Train(160, x)
	view(x, inout, input)
	x.AddNode(1)
	dm.Train(101, x)
	view(x, inout, input)
	x.RemoveNode(1, 2)
	x.AddNode(2)
	dm.Train(6101, x)
	view(x, inout, input)
	x.AddNode(3)
	dm.Train(601, x)
	view(x, inout, input)
	x.RemoveNode(3, 1)
	dm.Train(1603, x)
	view(x, inout, input)
	x.AddNode(2)
	dm.Train(1060, x)
	view(x, inout, input)

	fmt.Println("Con[0]: ", x.Connectors[0].GetWeight())
	fmt.Println("Con[1]: ", x.Connectors[1].GetWeight())
}

func view(x goann.Network, inout uint, input []float64) {
	var r = int(inout)
	for c := 0; c < 8; c++ {
		x.PutData(input[c*r : (c+1)*r])
		x.Propagation()
		fmt.Println(input[c*r:(c+1)*r], x.GetFinal())
	}
}
