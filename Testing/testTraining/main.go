package main

import (
	"fmt"

	dtm "github.com/Go-ANN/DataManagement"

	goann "github.com/Go-ANN"
	"github.com/Go-ANN/act"
)

func main() {
	var x goann.Network
	var inout uint = 3

	x.AddLayer(act.Linear{}, inout)
	x.AddLayer(act.Linear{}, 5)
	x.AddLayer(act.Sigmoid{}, 6)
	x.AddLayer(act.SoftSign{}, 4)
	x.AddLayer(act.ReLU{}, 5)
	x.AddLayer(act.ReLU{}, inout)
	x.ConnectLayers()

	var input []float64
	input = make([]float64, 0, 300)
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

	dm.Train(1600, x)
	view(x, inout, input)
	dm.Train(1601, x)
	view(x, inout, input)
	dm.Train(1061, x)
	view(x, inout, input)
	dm.Train(6101, x)
	view(x, inout, input)
	dm.Train(601, x)
	view(x, inout, input)
	dm.Train(1603, x)
	view(x, inout, input)
	dm.Train(1060, x)
	view(x, inout, input)

	var filepath string
	var filename string
	filepath = "C:\\users\\us3rs\\Desktop\\"
	filename = "testakufdgb.json"
	fmt.Println(x.SaveJSON(filepath, filename))
	//view(x, inout, []float64{0.5, 0., 0.5, 0.25, 0., 0.75, 1., 0.75, 0.25, 1.5, 0.25, 0.5, 0.25, 0.75, 1., 0.15, 0.1, 0.85, 0.45, 0.84, 0.22, 0.5, 0.5, 0.5})

	/*
		x.PutData(input)
		x.Propagation()
		fmt.Println(input, x.GetFinal())
		x.BackPropagation(input)

		for c := 0; c < 500000; c++ {
			for index := 0; index < int(inout); index++ {
				input[index] = rand.Float64()
			}
			x.PutData(input)
			x.Propagation()
			x.BackPropagation(input)
		}*/
}

func view(x goann.Network, inout uint, input []float64) {
	var r = int(inout)
	for c := 0; c < 8; c++ {
		x.PutData(input[c*r : (c+1)*r])
		x.Propagation()
		fmt.Println(input[c*r:(c+1)*r], x.GetFinal())
	}
}
