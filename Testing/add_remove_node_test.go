package Testing

import (
	"fmt"
	"testing"

	goann "github.com/Go-ANN"
	dtm "github.com/Go-ANN/DataManagement"
	"github.com/Go-ANN/act"
)

func TestAddRemoveAfterCreation(t *testing.T) {
	//Setup start
	var x goann.Network
	var inout uint = 3

	x.AddLayer(act.Linear{}, inout)
	x.AddLayer(act.Sigmoid{}, 6)
	x.AddLayer(act.ReLU{}, inout)
	x.ConnectLayers()

	var input []float64
	input = make([]float64, 0, 100)
	var dmb dtm.DataBuilder
	dmb.SetNumInput(inout)
	dmb.SetNumOutput(inout)
	dmb.AddxTrainValues(input)
	dmb.AddyTrainValues(input)

	input = append(input, 0., 1., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1.)

	input = append(input, 0., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0.)

	dmb.AddxTrainValues(input[:8*3])
	dmb.AddyTrainValues(input[8*3:])

	var dm = dmb.Build()
	//Setup is done

	want := x.MetaData.NodeCounts[1] + 1
	x.AddNode(1)
	got := x.MetaData.NodeCounts[1]
	if got != want {
		t.Errorf("got = %d, want = %d", got, want)
	}
	dm.Train(160, x)
	view(x, inout, input)

	want = x.MetaData.NodeCounts[1] - 1
	x.RemoveNode(1, 2)
	got = x.MetaData.NodeCounts[1]
	if got != want {
		t.Errorf("got = %d, want = %d", got, want)
	}
	wantStr := "no null"
	dm.Train(6101, x)
	view(x, inout, input)
	fmt.Printf("If %s : Then Pass confirmed\n", wantStr)
}

func view(x goann.Network, inout uint, input []float64) {
	var r = int(inout)
	for c := 0; c < 8; c++ {
		x.PutData(input[c*r : (c+1)*r])
		x.Propagation()
		fmt.Println(input[c*r:(c+1)*r], x.GetFinal())
	}
}
