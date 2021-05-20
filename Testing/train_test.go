package Testing

import (
	"testing"

	goann "github.com/Go-ANN"
	dtm "github.com/Go-ANN/DataManagement"
	"github.com/Go-ANN/act"
)

func TestTraining(t *testing.T) {
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

	notwanted := x.Connectors[4].GetWeight()
	dm.Train(160, x)
	view(x, inout, input)
	dm.Train(160, x)
	view(x, inout, input)
	dm.Train(101, x)
	view(x, inout, input)
	dm.Train(6101, x)
	view(x, inout, input)
	dm.Train(601, x)
	view(x, inout, input)
	dm.Train(1603, x)
	view(x, inout, input)
	dm.Train(1060, x)
	view(x, inout, input)
	got := x.Connectors[4].GetWeight()

	for i := 0; i < len(notwanted); i++ {
		if equal(notwanted[i], got[i]) {
			t.Errorf("\nnotwanted = %f,\n got = %f", notwanted, got)
			break
		}
	}

}

//func view(...) { ... } is in add_remove_node_test.go

func equal(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}
